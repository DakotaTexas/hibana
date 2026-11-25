# SPDX-License-Identifier: Apache-2.0
"""Inkling 2's prefill kernels, and the experts' products for every call.

Prefill: `route` and `place` pick and sort a call's (token, expert) pairs on the GPU; `quantize_nvfp4`,
`quantize_mxfp8` and `swiglu_quantize` quantize the experts' inputs as hf_quant_config.json requires; `grouped_mm4`
and `grouped_mm8` multiply every expert in one block-scaled tensor-core launch a projection (a GEMM an expert took
12 ms a visit). `conv`, `mix`, `gate`, `queries` and `attention` are the steps between.

Decode and verification calls of up to `PAIR_ROWS` rows: `pair_mm` multiplies each (row, expert) pair where it is;
wider ones sort as prefill does. Both run inside CUDA graphs.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


# Token and pair counts reach the kernels as runtime values that Triton does not specialize on (do_not_specialize):
# specialized, every new prompt length compiled fresh variants, about 7 s of a served request's first token.
FP32_MAX = tl.constexpr(3.4028234663852886e38)
SPLIT_ROWS = 64  # calls up to this many rows (decode, verification) quantize a row in many programs


@triton.jit
def _e2m1(x):
    """The E2M1 code (sign in bit 3) of FP32 x, rounded to nearest with ties to the even code and saturating, and its value."""
    a = tl.abs(x)
    code = tl.where(a <= 0.25, 0, tl.where(a < 0.75, 1, tl.where(a <= 1.25, 2, tl.where(a < 1.75, 3, tl.where(
        a <= 2.5, 4, tl.where(a < 3.5, 5, tl.where(a <= 5.0, 6, 7))))))).to(tl.uint32)
    value = tl.where(code < 4, code.to(tl.float32) * 0.5, tl.where(code == 4, 2.0, tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0))))
    negative = x < 0
    return tl.where(negative, code | 8, code), tl.where(negative, -value, value)


@triton.jit
def _nvfp4_block(x, encode, decode, outer, G: tl.constexpr):
    """A run of G blocks of 16 FP32 values -> (their packed codes [8 G] uint8, their E4M3 scales [G] FP32)."""
    x = tl.reshape(x, (G, 16))
    most = tl.max(tl.abs(x), 1)
    scale6 = tl.minimum(most * (encode * (1.0 / 6.0)), 448.0).to(tl.float8e4nv).to(tl.float32)
    scale4 = tl.minimum(most * (encode * 0.25), 448.0).to(tl.float8e4nv).to(tl.float32)
    code6, value6 = _e2m1(x * tl.minimum(tl.div_rn(1.0, scale6 * decode), FP32_MAX)[:, None])
    code4, value4 = _e2m1(x * tl.minimum(tl.div_rn(1.0, scale4 * decode), FP32_MAX)[:, None])
    error6 = tl.abs(value6 * scale6[:, None] * outer - x)
    error4 = tl.abs(value4 * scale4[:, None] * outer - x)
    for width in tl.static_range(3, -1, -1):  # summed pairwise, as the reference sums them
        low6, high6 = tl.split(tl.reshape(error6, (G, 1 << width, 2)))
        low4, high4 = tl.split(tl.reshape(error4, (G, 1 << width, 2)))
        error6 = low6 + high6
        error4 = low4 + high4
    four = tl.reshape(error4, (G,)) < tl.reshape(error6, (G,))
    low, high = tl.split(tl.reshape(tl.where(four[:, None], code4, code6), (G, 8, 2)))
    return tl.reshape((low | (high << 4)).to(tl.uint8), (8 * G,)), tl.where(four, scale4, scale6)


@triton.jit
def _quantize_nvfp4(X, Codes, Scales, Unit, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    amax = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, K, BLOCK):
        amax = tl.maximum(amax, tl.abs(tl.load(X + row * K + start + cols).to(tl.float32)))
    amax = tl.max(amax, 0)
    encode = tl.minimum(tl.div_rn(1536.0, amax), FP32_MAX)
    decode = tl.div_rn(1.0, encode)
    outer = amax * (1.0 / 1536.0)
    tl.store(Unit + row, outer)
    G: tl.constexpr = BLOCK // 16
    for start in range(0, K, BLOCK):
        codes, scales = _nvfp4_block(tl.load(X + row * K + start + cols).to(tl.float32), encode, decode, outer, G)
        tl.store(Codes + row * (K // 2) + start // 2 + tl.arange(0, BLOCK // 2), codes)
        tl.store(Scales + row * (K // 16) + start // 16 + tl.arange(0, G), scales.to(tl.float8e4nv))


@triton.jit
def _quantize_nvfp4_split(X, Codes, Scales, Unit, K: tl.constexpr, BLOCK: tl.constexpr, WIDE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    start = tl.program_id(1) * BLOCK
    wide = tl.arange(0, WIDE)
    amax = tl.max(tl.abs(tl.load(X + row * K + wide, wide < K, other=0.0).to(tl.float32)), 0)  # every program finds the row's
    encode = tl.minimum(tl.div_rn(1536.0, amax), FP32_MAX)
    decode = tl.div_rn(1.0, encode)
    outer = amax * (1.0 / 1536.0)
    if start == 0:
        tl.store(Unit + row, outer)
    G: tl.constexpr = BLOCK // 16
    codes, scales = _nvfp4_block(tl.load(X + row * K + start + tl.arange(0, BLOCK)).to(tl.float32), encode, decode, outer, G)
    tl.store(Codes + row * (K // 2) + start // 2 + tl.arange(0, BLOCK // 2), codes)
    tl.store(Scales + row * (K // 16) + start // 16 + tl.arange(0, G), scales.to(tl.float8e4nv))


def quantize_nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BF16 rows [M, K] as NVFP4, the way the checkpoint requires an NVFP4 expert's input: a row's values are codes
    x block scales x unit, unit = max |row| / 1536; of the two E4M3 block scales that put a block's largest value at 6
    and at 4, a block keeps the one whose codes reconstruct it with the smaller summed absolute error
    ("fouroversix_mae"; the rule and its arithmetic follow vLLM's `ops/nvfp4.py`, see THIRD_PARTY_NOTICES.md).
    -> (codes [M, K / 2] uint8, the even column in the low nibble; E4M3 block scales [M, K / 16]; unit FP32 [M])."""
    m, k = x.shape
    codes = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((m, k // 16), dtype=torch.float8_e4m3fn, device=x.device)
    unit = torch.empty(m, dtype=torch.float32, device=x.device)
    if m > SPLIT_ROWS:
        _quantize_nvfp4[(m,)](x, codes, scales, unit, K=k, BLOCK=256 if k % 256 == 0 else 128, num_warps=1, enable_fp_fusion=False)
    elif m:  # a program a 128 values: a decode row of 2,304 takes 1.6 us (6.7 as one program, bound by its instructions)
        _quantize_nvfp4_split[(m, k // 128)](x, codes, scales, unit, K=k, BLOCK=128, WIDE=triton.next_power_of_2(k), num_warps=1,
                                             enable_fp_fusion=False)
    return codes, scales, unit


@triton.jit
def _mxfp8_store(x, Values, Scales, at, count, BLOCK: tl.constexpr):
    """x [BLOCK / 32, 32] FP32, the values at flat positions `at`, as MXFP8: per 32 the smallest power-of-two scale that
    brings them within E4M3's 448, each value rounded to E4M3."""
    raw = tl.max(tl.abs(x), 1) * (1.0 / 448.0)
    exponent = ((raw.to(tl.int32, bitcast=True) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = tl.where(raw <= 2.0 ** -127, 0, tl.minimum(exponent, 254))
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    inverse = tl.where(exponent == 0, 0.0, tl.div_rn(1.0, scale))
    values = tl.minimum(tl.maximum(x * inverse[:, None], -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Values + at, tl.reshape(values, (BLOCK,)), at < count)
    groups = tl.program_id(0).to(tl.int64) * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    tl.store(Scales + groups, exponent.to(tl.uint8), groups * 32 < count)


@triton.jit(do_not_specialize=['count'])
def _quantize_mxfp8(X, Values, Scales, count, BLOCK: tl.constexpr):
    at = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.reshape(tl.load(X + at, at < count, other=0.0).to(tl.float32), (BLOCK // 32, 32))
    _mxfp8_store(x, Values, Scales, at, count, BLOCK)


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 rows [M, K] as MXFP8, the way the checkpoint requires an MXFP8 expert's input: per 32 of a row the smallest
    power-of-two scale that brings them within E4M3's 448, each value rounded to E4M3.
    -> (E4M3 values [M, K], the scales' exponent bytes [M, K / 32])."""
    m, k = x.shape
    values = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((m, k // 32), dtype=torch.uint8, device=x.device)
    if m:
        _quantize_mxfp8[(triton.cdiv(m * k, 1024),)](x, values, scales, m * k, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return values, scales


@triton.jit
def _route_sums(X, Router, r, valid, experts, H: tl.constexpr, BM: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr):
    """The router logits of rows r for `experts`, FP32 [BM, BE]: BF16 operands, so each product is exact, as in an FP32
    product of the same values; summed BK at a time in K order, whatever BM and BE."""
    cols = tl.arange(0, BK)
    logits = tl.zeros((BM, BE), tl.float32)
    for k in range(0, H, BK):
        x = tl.load(X + r[:, None] * H + k + cols[None, :], valid[:, None], other=0.0)
        logits = tl.dot(x, tl.load(Router + experts[None, :] * H + k + cols[:, None]), logits)
    return logits


@triton.jit(do_not_specialize=['rows'])
def _route_logits(X, Router, Logits, rows, H: tl.constexpr, E: tl.constexpr, BM: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr):
    r = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    valid = r < rows
    experts = tl.program_id(1) * BE + tl.arange(0, BE)
    logits = _route_sums(X, Router, r, valid, experts, H, BM, BE, BK)
    tl.store(Logits + r[:, None] * E + experts[None, :], logits, valid[:, None])


@triton.jit(do_not_specialize=['rows'])
def _route(X, Router, Logits, Bias, Ids, Weights, rows, H: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, SCALE: tl.constexpr,
           BM: tl.constexpr, BK: tl.constexpr, SUMMED: tl.constexpr):
    r = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    valid = r < rows
    experts = tl.arange(0, E)
    if SUMMED:  # `_route_logits`' sums
        logits = tl.load(Logits + r[:, None] * E + experts[None, :], valid[:, None], other=0.0)
    else:
        logits = _route_sums(X, Router, r, valid, experts, H, BM, E, BK)
    scores = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))  # torch.sigmoid's arithmetic
    biased = scores + tl.load(Bias + experts)[None, :]
    left = biased
    chosen = tl.zeros((BM, E), tl.int1)
    for _ in tl.static_range(TOPK):  # the largest left, the lowest expert of equals
        least = tl.max(left, 1)
        pick = tl.min(tl.where(left == least[:, None], experts[None, :], E), 1)
        hit = experts[None, :] == pick[:, None]
        chosen = chosen | hit
        left = tl.where(hit, float("-inf"), left)
    # in torch.topk(sorted=False)'s order: those above the smallest chosen by expert, then those equal to it; the
    # weights summed in that order
    above = chosen & (biased > least[:, None])
    slot = tl.where(above, tl.cumsum(above.to(tl.int32), 1), tl.sum(above.to(tl.int32), 1)[:, None] + tl.cumsum((chosen & ~above).to(tl.int32), 1)) - 1
    total = tl.zeros((BM,), tl.float32)
    for j in tl.static_range(TOPK):
        at = chosen & (slot == j)
        tl.store(Ids + r * TOPK + j, tl.sum(tl.where(at, experts[None, :], 0), 1).to(tl.int64), valid)
        total += tl.sum(tl.where(at, scores, 0.0), 1)
    for j in tl.static_range(TOPK):
        weight = tl.sum(tl.where(chosen & (slot == j), scores, 0.0), 1)
        tl.store(Weights + r * TOPK + j, tl.div_rn(weight, total + 1e-20) * SCALE, valid)


SPLIT_ROUTE_ROWS = 512  # calls up to this many rows route in two launches (`route`)


def route(x: torch.Tensor, router: torch.Tensor, bias: torch.Tensor, topk: int, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A wide call's routing in one launch: x [T, H] and router [E, H] BF16, bias [E] FP32 -> (ids int64 [T, topk],
    by expert; weights FP32 [T, topk] = scale x score / sum of the chosen scores), the top experts by sigmoid score
    plus bias."""
    t, h = x.shape
    experts = router.shape[0]
    ids = torch.empty((t, topk), dtype=torch.int64, device=x.device)
    weights = torch.empty((t, topk), dtype=torch.float32, device=x.device)
    if t > SPLIT_ROUTE_ROWS:
        # 32 rows a program: 50 us for 1,678 tokens (16: 73, 64: 61), where PyTorch's chain took 180
        _route[(triton.cdiv(t, 32),)](x, router, None, bias, ids, weights, t, H=h, E=experts, TOPK=topk, SCALE=scale, BM=32, BK=64,
                                      SUMMED=False, num_warps=4, enable_fp_fusion=False)
    elif t:
        # verification-sized calls: a program a row block's sums would stream the whole router alone (128 rows: 44 us);
        # the sums spread over expert blocks first, the same sums in the same order, then the choice (14 + 7 us)
        logits = torch.empty((t, experts), dtype=torch.float32, device=x.device)
        _route_logits[(triton.cdiv(t, 32), experts // 16)](x, router, logits, t, H=h, E=experts, BM=32, BE=16, BK=64, num_warps=4)
        _route[(triton.cdiv(t, 16),)](x, router, logits, bias, ids, weights, t, H=h, E=experts, TOPK=topk, SCALE=scale, BM=16, BK=64,
                                      SUMMED=True, num_warps=4, enable_fp_fusion=False)
    return ids, weights


class Plan(NamedTuple):
    """A wide call's (token, expert) pairs sorted by expert, from `place`: expert e owns sorted rows starts[e] ..
    starts[e + 1] (int32 [E + 1]); sorted row i is token[i]'s (int32 [P]); pair p (= token x topk + k) is sorted row
    where[p] (int64 [P]); tiles [n, 2] int32 is each (expert, first sorted row) block of `TILE_ROWS` rows the products
    run, expert -1 past the last."""
    starts: torch.Tensor
    token: torch.Tensor
    where: torch.Tensor
    tiles: torch.Tensor


TILE_ROWS = 128  # the expert products' rows a tile


@triton.jit
def _repeated(Ids, at, ids, pairs, TOPK: tl.constexpr):
    """Pairs naming an expert their token named in an earlier slot: expert sharing's weight-0 filler."""
    repeat = at < 0
    for i in tl.static_range(1, TOPK):
        earlier = tl.load(Ids + at - i, ((at % TOPK) >= i) & (at < pairs), other=-1).to(tl.int32)
        repeat = repeat | (earlier == ids)
    return repeat


@triton.jit(do_not_specialize=['pairs'])
def _place(Ids, Starts, Token, Where, Tiles, pairs, E: tl.constexpr, TOPK: tl.constexpr, BM: tl.constexpr, TILES: tl.constexpr,
           CHUNK: tl.constexpr, REPEATS: tl.constexpr):
    expert = tl.program_id(0)
    lanes = tl.arange(0, CHUNK)
    counts = tl.zeros((E,), tl.int32)
    for start in range(0, pairs, CHUNK):  # every program counts every expert's pairs
        at = start + lanes
        ids = tl.load(Ids + at, at < pairs, other=0).to(tl.int32)
        live = at < pairs
        if REPEATS:
            live = live & ~_repeated(Ids, at, ids, pairs, TOPK)
        counts += tl.histogram(ids, E, mask=live)
    experts = tl.arange(0, E)
    first = tl.sum(tl.where(experts < expert, counts, 0), 0)
    placed = first
    for start in range(0, pairs, CHUNK):  # this expert's pairs in pair order, so a token's rows stay in order
        at = start + lanes
        ids = tl.load(Ids + at, at < pairs, other=-1).to(tl.int32)
        mine = (ids == expert) & (at < pairs)
        if REPEATS:
            mine = mine & ~_repeated(Ids, at, ids, pairs, TOPK)
        dest = placed + tl.cumsum(mine.to(tl.int32), 0) - 1
        tl.store(Token + dest, at // TOPK, mine)
        tl.store(Where + at, dest.to(tl.int64), mine)
        placed += tl.sum(mine.to(tl.int32), 0)
    if expert == 0:
        ends = tl.cumsum(counts, 0)
        tl.store(Starts + experts + 1, ends)
        tl.store(Starts + experts, 0, experts == 0)
        blocks = (counts + BM - 1) // BM
        last = tl.cumsum(blocks, 0)  # tiles before the next expert's first
        tile = tl.arange(0, TILES)
        owner = tl.sum((last[None, :] <= tile[:, None]).to(tl.int32), 1)  # the expert whose tiles hold slot `tile`
        pick = experts[None, :] == owner[:, None]
        row = tl.sum(tl.where(pick, ends - counts, 0), 1) + (tile - tl.sum(tl.where(pick, last - blocks, 0), 1)) * BM
        tl.store(Tiles + tile * 2, tl.where(owner < E, owner, -1))
        tl.store(Tiles + tile * 2 + 1, row)


def place(ids: torch.Tensor, experts: int, repeats: bool = False) -> Plan:
    """The `Plan` of a wide call's routed ids [T, topk] (int64), on the GPU without a host sync: products are launched
    for the most tiles any routing can need, and tiles past the last exit. With `repeats` (expert sharing's routing),
    a token's repeats of an expert are left out: they are weight-0 filler, and `combine` skips them."""
    t, topk = ids.shape
    pairs = t * topk
    tiles = triton.cdiv(pairs, TILE_ROWS) + experts
    starts = torch.empty(experts + 1, dtype=torch.int32, device=ids.device)
    token = torch.empty(pairs, dtype=torch.int32, device=ids.device)
    where = torch.empty(pairs, dtype=torch.int64, device=ids.device)
    table = torch.empty((triton.next_power_of_2(tiles), 2), dtype=torch.int32, device=ids.device)
    _place[(experts,)](ids, starts, token, where, table, pairs, E=experts, TOPK=topk, BM=TILE_ROWS, TILES=table.shape[0], CHUNK=2048,
                       REPEATS=repeats, num_warps=8)
    return Plan(starts, token, where, table[:tiles])


@triton.jit
def _tile(Tiles, Starts, Token, BM: tl.constexpr, GATHER: tl.constexpr, ACROSS: tl.constexpr):
    """A product program's tile: (expert, its sorted rows [BM] int64, which are live, each one's input row, its output tile).
    The programs run output tiles fastest (ACROSS) or row tiles fastest: for 1,678 tokens an NVFP4 down projection
    1.31 against 1.61 ms (MXFP8 1.92 against 2.95), the up projection 2.21 against 2.03."""
    tile = tl.program_id(1) if ACROSS else tl.program_id(0)
    expert = tl.load(Tiles + tile * 2)
    first = tl.load(Tiles + tile * 2 + 1)
    rows = first + tl.arange(0, BM)
    valid = rows < tl.load(Starts + tl.maximum(expert, 0) + 1)
    source = rows
    if GATHER:  # a token's row, quantized once, read where each of its pairs needs it
        source = tl.load(Token + rows, valid, other=0)
    return expert, rows.to(tl.int64), valid, source.to(tl.int64), tl.program_id(0) if ACROSS else tl.program_id(1)


# up projections of up to this many pairs (decode, verification) run output tiles fastest: real 128-row verify calls
# 1.527 -> 1.492 ms (NVFP4), 64 rows 1.279 -> 1.253; down projections always do
ACROSS_PAIRS = 1024


@triton.jit
def _grouped_mm8(A, AS, W, WS, Factor, Tiles, Starts, Token, Y, N: tl.constexpr, K: tl.constexpr, GATHER: tl.constexpr, ACROSS: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    expert, rows, valid, source, column = _tile(Tiles, Starts, Token, BM, GATHER, ACROSS)
    if expert < 0:
        return
    out = column * BN + tl.arange(0, BN)
    cols = expert.to(tl.int64) * N + out
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        ks = k * BK + tl.arange(0, BK)
        groups = k * (BK // 32) + tl.arange(0, BK // 32)
        a = tl.load(A + source[:, None] * K + ks[None, :], valid[:, None] & (ks[None, :] < K), other=0.0)
        a_scale = tl.load(AS + source[:, None] * (K // 32) + groups[None, :], valid[:, None] & (groups[None, :] < K // 32), other=127)
        w = tl.load(W + cols[None, :] * K + ks[:, None], (out[None, :] < N) & (ks[:, None] < K), other=0.0)
        w_scale = tl.load(WS + cols[:, None] * (K // 32) + groups[None, :], (out[:, None] < N) & (groups[None, :] < K // 32), other=127)
        acc = tl.dot_scaled(a, a_scale, "e4m3", w, w_scale, "e4m3", acc)
    acc *= tl.load(Factor + expert)
    tl.store(Y + rows[:, None] * N + out[None, :], acc.to(tl.bfloat16), valid[:, None] & (out[None, :] < N))


def grouped_mm8(values: torch.Tensor, scales: torch.Tensor, plan: Plan, gather: bool,
                weight: torch.Tensor, scale: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """`grouped_mm4` for MXFP8 experts (weight [E, N, K] E4M3, exponent bytes [E, N, K / 32]) on MXFP8 rows
    (`quantize_mxfp8`'s). -> BF16 [P, N]."""
    experts, n, k = weight.shape
    pairs = plan.token.shape[0]
    out = torch.empty((pairs, n), dtype=torch.bfloat16, device=values.device)
    if pairs:
        # 128 x 128 tiles, 256 columns a step (512 does not fit the shared memory): 6.1 ms a projection against 7.9
        # with 64 outputs a tile and 8.2 with both sides decoded into BF16 operands
        across = not gather or pairs <= ACROSS_PAIRS
        grid = (triton.cdiv(n, 128), plan.tiles.shape[0]) if across else (plan.tiles.shape[0], triton.cdiv(n, 128))
        _grouped_mm8[grid](
            values, scales, weight, scale, factor, plan.tiles, plan.starts, plan.token, out, N=n, K=k, GATHER=gather, ACROSS=across, BM=TILE_ROWS, BN=128,
            BK=256, num_warps=8, num_stages=2)
    return out


@triton.jit
def _grouped_mm4(A, AS, Unit, W, WS, Factor, Tiles, Starts, Token, Y, N: tl.constexpr, K: tl.constexpr, GATHER: tl.constexpr,
                 ACROSS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    expert, rows, valid, source, column = _tile(Tiles, Starts, Token, BM, GATHER, ACROSS)
    if expert < 0:
        return
    out = column * BN + tl.arange(0, BN)
    cols = expert.to(tl.int64) * N + out  # this expert's rows of W
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        packed = k * (BK // 2) + tl.arange(0, BK // 2)
        groups = k * (BK // 16) + tl.arange(0, BK // 16)
        a = tl.load(A + source[:, None] * (K // 2) + packed[None, :], valid[:, None] & (packed[None, :] < K // 2), other=0)
        a_scale = tl.load(AS + source[:, None] * (K // 16) + groups[None, :], valid[:, None] & (groups[None, :] < K // 16), other=0.0)
        w = tl.load(W + cols[None, :] * (K // 2) + packed[:, None], (out[None, :] < N) & (packed[:, None] < K // 2), other=0)
        w_scale = tl.load(WS + cols[:, None] * (K // 16) + groups[None, :], (out[:, None] < N) & (groups[None, :] < K // 16), other=0.0)
        acc = tl.dot_scaled(a, a_scale, "e2m1", w, w_scale, "e2m1", acc)
    acc *= tl.load(Factor + expert)
    acc *= tl.load(Unit + source, valid, other=0.0)[:, None]
    tl.store(Y + rows[:, None] * N + out[None, :], acc.to(tl.bfloat16), valid[:, None] & (out[None, :] < N))


def grouped_mm4(codes: torch.Tensor, scales: torch.Tensor, unit: torch.Tensor, plan: Plan, gather: bool,
                weight: torch.Tensor, scale: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Every NVFP4 expert's projection of its rows in one launch, over the sorted rows of `plan`: the input rows are
    `quantize_nvfp4`'s, one a token read through plan.token (`gather`) or already one a sorted row. weight
    [E, N, K / 2] uint8 with E4M3 block scales [E, N, K / 16] as bytes, factor [E] FP32, as the checkpoint stores
    them. Both sides stay packed and multiply on the tensor cores' block-scaled product.
    -> BF16 [P, N] = unit x factor_e x (codes x scales) @ W_e^T."""
    n = weight.shape[1]
    k = weight.shape[2] * 2
    pairs = plan.token.shape[0]
    out = torch.empty((pairs, n), dtype=torch.bfloat16, device=codes.device)
    if pairs:
        # 128 x 128 tiles: 1,401 tokens take 6.0 ms a visit for the whole MoE (64-wide tiles 6.2 to 6.3)
        across = not gather or pairs <= ACROSS_PAIRS
        grid = (triton.cdiv(n, 128), plan.tiles.shape[0]) if across else (plan.tiles.shape[0], triton.cdiv(n, 128))
        _grouped_mm4[grid](
            codes, scales, unit, weight, scale.view(torch.float8_e4m3fn), factor, plan.tiles, plan.starts, plan.token, out,
            N=n, K=k, GATHER=gather, ACROSS=across, BM=TILE_ROWS, BN=128, BK=512, num_warps=8, num_stages=2)
    return out


@triton.jit
def _swiglu(Hidden, at, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    """The experts' activation at flat positions `at` of [P, I] as FP32 of its BF16 rounding, from hidden [P, 2 I] BF16
    (gate and up alternating): BF16(softcap(g) sigmoid(g) softcap(u)), softcap_c(z) = c tanh(z / c)."""
    pair = tl.load(Hidden.to(tl.pointer_type(tl.uint32)) + at)  # (gate, up) in one load
    g = (pair << 16).to(tl.float32, bitcast=True)
    u = (pair & 0xFFFF0000).to(tl.float32, bitcast=True)
    gate = tl.div_rn(libdevice.tanh(tl.div_rn(g, GATE_CAP)) * GATE_CAP, 1.0 + libdevice.exp(-g))
    return (gate * (libdevice.tanh(tl.div_rn(u, UP_CAP)) * UP_CAP)).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _swiglu_nvfp4_part(Codes, Scales, act, row, start, encode, decode, outer, I: tl.constexpr, BLOCK: tl.constexpr):
    codes, scales = _nvfp4_block(act, encode, decode, outer, BLOCK // 16)
    tl.store(Codes + row * (I // 2) + start // 2 + tl.arange(0, BLOCK // 2), codes)
    tl.store(Scales + row * (I // 16) + start // 16 + tl.arange(0, BLOCK // 16), scales.to(tl.float8e4nv))


@triton.jit
def _swiglu_nvfp4(Hidden, Codes, Scales, Unit, I: tl.constexpr, MAIN: tl.constexpr, TAIL: tl.constexpr, GATE_CAP: tl.constexpr,
                  UP_CAP: tl.constexpr):
    # a row's activation stays in registers, as a power-of-two run and its tail (1,152 = 1,024 + 128), between its
    # amax and its codes: 13,432 rows take 304 us, an activation kernel then `quantize_nvfp4` 557
    row = tl.program_id(0).to(tl.int64)
    main = _swiglu(Hidden, row * I + tl.arange(0, MAIN), GATE_CAP, UP_CAP)
    amax = tl.max(tl.abs(main), 0)
    if TAIL:
        tail = _swiglu(Hidden, row * I + MAIN + tl.arange(0, TAIL), GATE_CAP, UP_CAP)
        amax = tl.maximum(amax, tl.max(tl.abs(tail), 0))
    encode = tl.minimum(tl.div_rn(1536.0, amax), FP32_MAX)
    decode = tl.div_rn(1.0, encode)
    outer = amax * (1.0 / 1536.0)
    tl.store(Unit + row, outer)
    _swiglu_nvfp4_part(Codes, Scales, main, row, 0, encode, decode, outer, I, MAIN)
    if TAIL:
        _swiglu_nvfp4_part(Codes, Scales, tail, row, MAIN, encode, decode, outer, I, TAIL)


@triton.jit(do_not_specialize=['count'])
def _swiglu_mxfp8(Hidden, Values, Scales, count, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr, BLOCK: tl.constexpr):
    at = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.reshape(_swiglu(Hidden, tl.minimum(at, count - 1), GATE_CAP, UP_CAP), (BLOCK // 32, 32))
    _mxfp8_store(x, Values, Scales, at, count, BLOCK)


def swiglu_quantize(hidden: torch.Tensor, nvfp4: bool, gate_cap: float, up_cap: float) -> tuple[torch.Tensor, ...]:
    """`quantize_nvfp4` (nvfp4) or `quantize_mxfp8` of the experts' activation (`_swiglu`) in one pass: hidden
    [P, 2 I] BF16, gate and up alternating."""
    p, i = hidden.shape[0], hidden.shape[1] // 2
    if nvfp4:
        codes = torch.empty((p, i // 2), dtype=torch.uint8, device=hidden.device)
        scales = torch.empty((p, i // 16), dtype=torch.float8_e4m3fn, device=hidden.device)
        unit = torch.empty(p, dtype=torch.float32, device=hidden.device)
        if p:
            main = 1 << (i.bit_length() - 1)
            _swiglu_nvfp4[(p,)](hidden, codes, scales, unit, I=i, MAIN=main, TAIL=i - main, GATE_CAP=gate_cap, UP_CAP=up_cap, num_warps=4,
                                enable_fp_fusion=False)
        return codes, scales, unit
    values = torch.empty((p, i), dtype=torch.float8_e4m3fn, device=hidden.device)
    scales = torch.empty((p, i // 32), dtype=torch.uint8, device=hidden.device)
    if p:
        _swiglu_mxfp8[(triton.cdiv(p * i, 1024),)](hidden, values, scales, p * i, GATE_CAP=gate_cap, UP_CAP=up_cap, BLOCK=1024, num_warps=4,
                                                   enable_fp_fusion=False)
    return values, scales


@triton.jit
def _swiglu_store(Y, acc, rows, valid, cols, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    """Store the activation of acc's gate/up column pairs (output columns `cols` // 2) as `_swiglu` computes it from
    the BF16 product."""
    gate, up = tl.split(tl.reshape(acc.to(tl.bfloat16).to(tl.float32), (BM, BN // 2, 2)))
    gate_cap = libdevice.tanh(tl.div_rn(gate, GATE_CAP)) * GATE_CAP
    up_cap = libdevice.tanh(tl.div_rn(up, UP_CAP)) * UP_CAP
    act = tl.div_rn(gate_cap, 1.0 + libdevice.exp(-gate)) * up_cap
    tl.store(Y + rows[:, None] * (N // 2) + cols[None, :], act.to(tl.bfloat16), valid[:, None])


@triton.jit
def _pair_mm(A, AS, Unit, Ids, W, WS, Factor, Y, SHARE: tl.constexpr, FP4: tl.constexpr, N: tl.constexpr, K: tl.constexpr, KM: tl.constexpr,
             KT: tl.constexpr, VB: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, STEPS: tl.constexpr,
             GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr, TOPK: tl.constexpr, REPEATS: tl.constexpr):
    pair = tl.program_id(0)
    row = (pair // SHARE).to(tl.int64)
    expert = tl.load(Ids + pair)
    if REPEATS:  # expert sharing's weight-0 filler (a repeat of an expert its row named in an earlier slot) is left unwritten
        repeat = False
        for i in tl.static_range(1, TOPK):
            repeat = repeat | (tl.load(Ids + pair - i, (pair % TOPK) >= i, other=-1) == expert)
        if repeat:
            return
    lead = tl.arange(0, BM) == 0  # the product's other rows stay empty
    spread = tl.zeros((BM, 1), tl.int64)
    KB: tl.constexpr = K // VB  # bytes a row (VB values a byte)
    # K = KM + KT (powers of two): a step multiplies BN whole rows of W in two products, the main KM values and the
    # tail, so it reads one contiguous span of codes and one of scales
    main_bytes, tail_bytes = tl.arange(0, KM // VB), KM // VB + tl.arange(0, KT // VB)
    main_groups, tail_groups = tl.arange(0, KM // G), KM // G + tl.arange(0, KT // G)
    a_main = tl.load(A + row * KB + main_bytes[None, :] + spread, lead[:, None], other=0.0)
    a_tail = tl.load(A + row * KB + tail_bytes[None, :] + spread, lead[:, None], other=0.0)
    if FP4:
        s_main = tl.load(AS + row * (K // G) + main_groups[None, :] + spread, lead[:, None], other=0.0)
        s_tail = tl.load(AS + row * (K // G) + tail_groups[None, :] + spread, lead[:, None], other=0.0)
    else:  # an MXFP8 scale's exponent byte; 127 is 1
        s_main = tl.load(AS + row * (K // G) + main_groups[None, :] + spread, lead[:, None], other=127)
        s_tail = tl.load(AS + row * (K // G) + tail_groups[None, :] + spread, lead[:, None], other=127)
    factor = tl.load(Factor + expert)
    unit = tl.load(Unit + row)  # MXFP8 calls pass the factors here
    for step in range(STEPS):
        out = (tl.program_id(1) * STEPS + step) * BN + tl.arange(0, BN)
        cols = expert * N + out  # this expert's rows of W
        w_main = tl.load(W + cols[None, :] * KB + main_bytes[:, None])
        w_tail = tl.load(W + cols[None, :] * KB + tail_bytes[:, None])
        ws_main = tl.load(WS + cols[:, None] * (K // G) + main_groups[None, :])
        ws_tail = tl.load(WS + cols[:, None] * (K // G) + tail_groups[None, :])
        if FP4:
            acc = tl.dot_scaled(a_main, s_main, "e2m1", w_main, ws_main, "e2m1")
            acc = tl.dot_scaled(a_tail, s_tail, "e2m1", w_tail, ws_tail, "e2m1", acc)
        else:
            acc = tl.dot_scaled(a_main, s_main, "e4m3", w_main, ws_main, "e4m3")
            acc = tl.dot_scaled(a_tail, s_tail, "e4m3", w_tail, ws_tail, "e4m3", acc)
        acc *= factor
        if FP4:
            acc *= unit
        if GATE_CAP > 0:
            # on row 0 alone (exactly: the max of it and -inf): on the whole tile, 16-row MXFP8 calls took 1.4x as long
            lead_row = tl.max(tl.where(lead[:, None], acc, float("-inf")), 0)[None, :]
            half = (tl.program_id(1) * STEPS + step) * (BN // 2) + tl.arange(0, BN // 2)
            _swiglu_store(Y, lead_row, tl.full((1,), pair, tl.int64), tl.full((1,), True, tl.int1), half, N, 1, BN, GATE_CAP, UP_CAP)
        else:
            tl.store(Y + pair.to(tl.int64) * N + out[None, :] + spread, acc.to(tl.bfloat16), lead[:, None])


# Decode and verification calls up to this many rows multiply each pair where it is (`pair_mm`); wider ones sort their
# pairs by expert as prefill does, reading each expert once. A visit's NVFP4 experts on real verify calls, pairs
# against sorted: 24 rows 1.32 against 1.39 ms, 64 rows 2.48 against 1.99 (sharing experts K2 6:24: 1.76 against 1.11)
PAIR_ROWS = 24


def pair_mm(rows: tuple, ids: torch.Tensor, share: int, weight: torch.Tensor, scale: torch.Tensor, factor: torch.Tensor,
            swiglu: tuple[float, float] | None = None, repeats: bool = False) -> torch.Tensor:
    """`grouped_mm4` / `grouped_mm8` for the few rows of a decode or verification call, a program a (pair, output
    tile) with no sorting (it runs in a captured graph): pair p multiplies row p // share of `rows` (what
    `quantize_nvfp4` or `quantize_mxfp8` returned) by expert ids.flatten()[p] (int64 [T, k]; a row repeats an expert
    only as expert sharing's weight-0 filler, whose rows `repeats` leaves unwritten, as `place` does, for `combine` to
    skip). -> BF16 [P, N]; with `swiglu` =
    (gate cap, up cap), the activation of the gate/up column pairs instead, BF16 [P, N / 2], bitwise `_swiglu` of
    the BF16 product (a launch less a visit). Rows that share an expert each read it, the later
    ones from cache: with real verification routing a call reads about the distinct experts' bytes once."""
    fp4 = weight.dtype == torch.uint8
    n, k = weight.shape[1], weight.shape[2] * (2 if fp4 else 1)
    main = 1 << (k.bit_length() - 1)
    # whole weight rows a step: one row's eight NVFP4 up projections 111 -> 104 us cold (a pure read of their bytes
    # takes 102-104; steps of 512 values, 128 rows apart in memory, read at 214 GB/s), MXFP8 194 -> 186 us
    bn, steps = (32 if fp4 else 16), 4
    out = torch.empty((ids.numel(), n // 2 if swiglu else n), dtype=torch.bfloat16, device=ids.device)
    codes, scales, unit = rows if fp4 else (*rows, factor)
    gate_cap, up_cap = swiglu or (0.0, 0.0)
    _pair_mm[(ids.numel(), n // (bn * steps))](codes, scales, unit, ids, weight, scale.view(torch.float8_e4m3fn) if fp4 else scale, factor, out,
                                               SHARE=share, FP4=fp4, N=n, K=k, KM=main, KT=k - main, VB=2 if fp4 else 1, G=16 if fp4 else 32, BM=16, BN=bn,
                                               STEPS=steps, GATE_CAP=gate_cap, UP_CAP=up_cap, TOPK=ids.shape[1], REPEATS=repeats, num_warps=4,
                                               num_stages=2)
    return out


@triton.jit
def _conv(Full, Taps, Dynamic, Add, Y, C: tl.constexpr, HEAD: tl.constexpr, DYNAMIC: tl.constexpr, ADD: tl.constexpr,
          OUT_F32: tl.constexpr, NORM: tl.constexpr, POST: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < C
    own = tl.load(Full + (t + 3) * C + cols, mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK,), tl.float32)
    for j in tl.static_range(4):  # the row j steps back
        value = own if j == 0 else tl.load(Full + (t + 3 - j) * C + cols, mask, other=0.0).to(tl.float32)
        if DYNAMIC:  # taps by backward distance: the layer's per-channel taps plus the row's per-head ones
            tap = tl.load(Taps + cols * 4 + j, mask, other=0.0) + tl.load(Dynamic + t * (C // HEAD) * 4 + (cols // HEAD) * 4 + j, mask, other=0.0).to(tl.float32)
        else:  # taps oldest first
            tap = tl.load(Taps + cols * 4 + (3 - j), mask, other=0.0)
        acc += value * tap
    out = ((acc + own) / NORM).to(tl.bfloat16)
    if POST == 1:  # each head clipped to an L2 norm of LIMIT (keys)
        heads = tl.reshape(out.to(tl.float32), (BLOCK // HEAD, HEAD))
        scale = tl.minimum(LIMIT / (tl.sqrt(tl.sum(heads * heads, 1)) + 1e-6), 1.0)
        out = tl.reshape(heads * scale[:, None], (BLOCK,)).to(tl.bfloat16)
    elif POST == 2:  # each head RMS-normalized, eps LIMIT (values)
        heads = tl.reshape(out.to(tl.float32), (BLOCK // HEAD, HEAD))
        scale = tl.rsqrt(tl.sum(heads * heads, 1) / HEAD + LIMIT)
        out = tl.reshape(heads * scale[:, None], (BLOCK,)).to(tl.bfloat16)
    if ADD:
        total = tl.load(Add + t * C + cols, mask, other=0.0).to(tl.float32) + out.to(tl.float32)
        if OUT_F32:
            tl.store(Y + t * C + cols, total, mask)
        else:
            tl.store(Y + t * C + cols, total.to(tl.bfloat16), mask)
    elif OUT_F32:
        tl.store(Y + t * C + cols, out.to(tl.float32), mask)
    else:
        tl.store(Y + t * C + cols, out, mask)


def conv(full: torch.Tensor, taps: torch.Tensor, dynamic: torch.Tensor | None = None, add: torch.Tensor | None = None,
         out_dtype: torch.dtype = torch.bfloat16, clip: float | None = None, rms_eps: float | None = None) -> torch.Tensor:
    """c[t] = BF16((sum_j tap_j full[t + 3 - j] + full[t + 3]) / sqrt(taps + 1)) over a sequence: full [T + 3, C] BF16 is
    the stream's three inputs before the forward, then its rows. taps [C, 4] FP32, oldest first; with `dynamic`
    [T, C / 128, 4] (a full-attention layer's key or value stream) taps are [C, 4] by backward distance and the row's
    per-head taps are added. -> c; or with `add` [T, C] (FP32 or BF16) add + c, as `out_dtype`; or, each head of 128
    apart, c clipped to an L2 norm of `clip` (keys) or RMS-normalized with `rms_eps` (values), BF16."""
    t, c = full.shape[0] - 3, full.shape[1]
    out = torch.empty((t, c), dtype=out_dtype, device=full.device)
    if t:
        post, limit = (1, clip) if clip is not None else (2, rms_eps) if rms_eps is not None else (0, 0.0)
        _conv[(t, triton.cdiv(c, 1024))](full, taps, taps if dynamic is None else dynamic, full if add is None else add, out, C=c, HEAD=128,
                                         DYNAMIC=dynamic is not None, ADD=add is not None, OUT_F32=out_dtype == torch.float32,
                                         NORM=(9.0 if dynamic is not None else 5.0) ** 0.5, POST=post, LIMIT=limit, BLOCK=1024, num_warps=4)
    return out


@triton.jit
def _mix_part(Full, Taps, Add, Residual, States, Queries, row, cols, count, H: tl.constexpr, SLOTS: tl.constexpr, Q: tl.constexpr,
              S: tl.constexpr, QP: tl.constexpr, CONV: tl.constexpr, ADD: tl.constexpr):
    """A run of the row's columns: (its sources [S, n], the queries' dots with them [QP, S], the own vector's square)."""
    source = tl.arange(0, S)  # the row's sources, padded to S: its `count` block states, then its own vector
    if CONV:  # the row's own vector is the stream's convolution joined to its residual
        own = tl.load(Full + (row + 3) * H + cols).to(tl.float32)
        acc = own * tl.load(Taps + cols * 4 + 3)
        for j in tl.static_range(1, 4):
            acc += tl.load(Full + (row + 3 - j) * H + cols).to(tl.float32) * tl.load(Taps + cols * 4 + (3 - j))
        x = ((acc + own) / 2.23606797749979).to(tl.bfloat16).to(tl.float32)
        if ADD:
            x += tl.load(Add + row * H + cols).to(tl.float32)
        tl.store(Residual + row * H + cols, x)
    else:
        x = tl.load(Add + row * H + cols).to(tl.float32)
    src = tl.load(States + (row * SLOTS + source[:, None]) * H + cols[None, :], source[:, None] < count, other=0.0)
    src = tl.where(source[:, None] == count, x[None, :], src)
    query = tl.arange(0, QP)
    dots = tl.zeros((QP, S), tl.float32)
    for i in tl.static_range(Q):  # a query at a time, to keep registers for the sources
        dot = tl.sum(tl.load(Queries + i * H + cols)[None, :] * src, 1)
        dots = tl.where(query[:, None] == i, dot[None, :], dots)
    return src, dots, tl.sum(x * x, 0)


@triton.jit
def _mix_out(Weight, Mixed, src, a, scale, row, rows, cols, H: tl.constexpr, Q: tl.constexpr, QP: tl.constexpr, NORMED: tl.constexpr):
    query = tl.arange(0, QP)
    for i in tl.static_range(Q):
        mixed = tl.sum(tl.sum(tl.where(query[:, None] == i, a, 0.0), 0)[:, None] * src, 0)
        at = (i * rows + row) * H + cols
        if NORMED:
            tl.store(Mixed + at, (mixed * tl.sum(tl.where(query == i, scale, 0.0), 0) * tl.load(Weight + i * H + cols)).to(tl.bfloat16))
        else:
            tl.store(Mixed + at, mixed)


@triton.jit
def _mix_square(src, a, Q: tl.constexpr, QP: tl.constexpr):
    """[QP]: each mix's sum of squares over a run of columns."""
    query = tl.arange(0, QP)
    square = tl.zeros((QP,), tl.float32)
    for i in tl.static_range(Q):
        mixed = tl.sum(tl.sum(tl.where(query[:, None] == i, a, 0.0), 0)[:, None] * src, 0)
        square = tl.where(query == i, tl.sum(mixed * mixed, 0), square)
    return square


@triton.jit(do_not_specialize=['count', 'rows'])
def _mix(Full, Taps, Add, Residual, States, Queries, Weight, Mixed, count, rows, H: tl.constexpr, SLOTS: tl.constexpr, Q: tl.constexpr,
         S: tl.constexpr, QP: tl.constexpr, CONV: tl.constexpr, ADD: tl.constexpr, NORMED: tl.constexpr, EPS: tl.constexpr,
         MAIN: tl.constexpr, TAIL: tl.constexpr):
    # the row's sources stay in registers, as a power-of-two run of columns and its tail (2,304 = 2,048 + 256), from
    # their one read to the mixes: a 1,678-token prefill's 120 mixes take 39 ms, 46 with passes that re-read them
    row = tl.program_id(0).to(tl.int64)
    main = tl.arange(0, MAIN)
    src, dots, own_square = _mix_part(Full, Taps, Add, Residual, States, Queries, row, main, count, H, SLOTS, Q, S, QP, CONV, ADD)
    if TAIL:
        tail = MAIN + tl.arange(0, TAIL)
        src2, dots2, own_square2 = _mix_part(Full, Taps, Add, Residual, States, Queries, row, tail, count, H, SLOTS, Q, S, QP, CONV, ADD)
        dots += dots2
        own_square += own_square2
    source = tl.arange(0, S)
    unit = tl.where(source == count, tl.rsqrt(own_square / H + EPS), 1.0)  # the own vector enters RMS-normalized
    logits = tl.where(source[None, :] <= count, dots * unit[None, :] * (H ** -0.5), float("-inf"))
    weights = tl.exp(logits - tl.max(logits, 1)[:, None])
    a = weights / tl.sum(weights, 1)[:, None] * unit[None, :]  # each raw source's coefficient, [QP, S]
    scale = tl.zeros((QP,), tl.float32) + 1.0
    if NORMED:
        square = _mix_square(src, a, Q, QP)
        if TAIL:
            square += _mix_square(src2, a, Q, QP)
        scale = tl.rsqrt(square / H + EPS)
    _mix_out(Weight, Mixed, src, a, scale, row, rows, main, H, Q, QP, NORMED)
    if TAIL:
        _mix_out(Weight, Mixed, src2, a, scale, row, rows, tail, H, Q, QP, NORMED)


def mix(states: torch.Tensor, count: int, queries: torch.Tensor, weight: torch.Tensor | None, *, own: torch.Tensor | None = None,
        full: torch.Tensor | None = None, taps: torch.Tensor | None = None, add: torch.Tensor | None = None):
    """Prefill's sublayer inputs: for each of `queries` [Q, H] (FP32, Q <= 4) the softmax mix, by that query, of the
    rows' first `count` normalized block states (states [T, slots, H] FP32) and the RMS norm of their own vector.
    The own vector is `own` [T, H] (BF16 or FP32), or the convolution of a hidden stream joined to its residual
    (`full` [T + 3, H] BF16 and `taps` as `conv`'s, plus `add` when given), which is then returned too, FP32.
    -> [Q, T, H]: each mix RMS-normalized through its row of `weight` [Q, H] as BF16, or FP32 as it is without one;
    with a convolution, (that, the residual [T, H])."""
    t, slots, h = states.shape
    q = queries.shape[0]
    out = torch.empty((q, t, h), dtype=torch.float32 if weight is None else torch.bfloat16, device=states.device)
    residual = torch.empty((t, h), dtype=torch.float32, device=states.device) if full is not None else None
    if t:
        dummy = states
        main = 1 << (h.bit_length() - 1)
        _mix[(t,)](dummy if full is None else full, dummy if taps is None else taps, (own if full is None else add) if (own is not None or add is not None) else dummy,
                   dummy if residual is None else residual, states, queries, queries if weight is None else weight, out, count, t,
                   H=h, SLOTS=slots, Q=q, S=triton.next_power_of_2(count + 1), QP=triton.next_power_of_2(q), CONV=full is not None,
                   ADD=add is not None, NORMED=weight is not None, EPS=1e-6, MAIN=main, TAIL=h - main, num_warps=4)
    return out if residual is None else (out, residual)


@triton.jit
def _gate_merged(x, m, low, Forget, Gain, cols, rank, R: tl.constexpr):
    """[BM, BK] (1 - tanh(forget)) x + tanh(gain) m at `cols`, the two gates read from the rank-R `low` [BM, R] BF16."""
    forget = tl.dot(low, tl.load(Forget + cols[None, :] * R + rank[:, None]))
    gain = tl.dot(low, tl.load(Gain + cols[None, :] * R + rank[:, None]))
    return (1.0 - libdevice.tanh(forget)) * x + libdevice.tanh(gain) * m


@triton.jit
def _gate_rows(X, Message, Down, Forget, Gain, Weight, Y, rows, gate, first, H: tl.constexpr, R: tl.constexpr, SHARED: tl.constexpr,
               EPS: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    r = (first + tl.arange(0, BM)).to(tl.int64)
    valid = (r < rows)[:, None]
    xs = X + ((r if SHARED else gate * rows + r) * H)[:, None]
    ms = Message + ((gate * rows + r) * H)[:, None]
    cols = tl.arange(0, BK)
    rank = tl.arange(0, R)
    # the rows' RMS of x and of the message, then the gates' rank-R read of the normalized pair (BF16 operands, the
    # result rounded to BF16 as the BF16 product returns it)
    sx = tl.zeros((BM,), tl.float32)
    sm = tl.zeros((BM,), tl.float32)
    for k in range(0, H, BK):
        x = tl.load(xs + k + cols[None, :], valid, other=0.0)
        m = tl.load(ms + k + cols[None, :], valid, other=0.0)
        sx += tl.sum(x * x, 1)
        sm += tl.sum(m * m, 1)
    rx = tl.rsqrt(sx / H + EPS)[:, None]
    rm = tl.rsqrt(sm / H + EPS)[:, None]
    down = Down + gate * R * 2 * H + rank[None, :] * (2 * H)
    low = tl.zeros((BM, R), tl.float32)
    for k in range(0, H, BK):
        x = tl.load(xs + k + cols[None, :], valid, other=0.0)
        m = tl.load(ms + k + cols[None, :], valid, other=0.0)
        low = tl.dot((x * rx).to(tl.bfloat16), tl.load(down + k + cols[:, None]), low)
        low = tl.dot((m * rm).to(tl.bfloat16), tl.load(down + H + k + cols[:, None]), low)
    low = low.to(tl.bfloat16)
    # the merge, twice: once for its RMS, once to write it normalized
    forget = Forget + gate * H * R
    gain = Gain + gate * H * R
    square = tl.zeros((BM,), tl.float32)
    for k in range(0, H, BK):
        x = tl.load(xs + k + cols[None, :], valid, other=0.0)
        m = tl.load(ms + k + cols[None, :], valid, other=0.0)
        merged = _gate_merged(x, m, low, forget, gain, k + cols, rank, R)
        square += tl.sum(merged * merged, 1)
    scale = tl.rsqrt(square / H + EPS)[:, None]
    ys = Y + ((gate * rows + r) * H)[:, None]
    for k in range(0, H, BK):
        x = tl.load(xs + k + cols[None, :], valid, other=0.0)
        m = tl.load(ms + k + cols[None, :], valid, other=0.0)
        merged = _gate_merged(x, m, low, forget, gain, k + cols, rank, R)
        weight = tl.load(Weight + gate * H + k + cols)[None, :]
        tl.store(ys + k + cols[None, :], (merged * scale * weight).to(tl.bfloat16), valid)


@triton.jit(do_not_specialize=['rows', 'tasks'])
def _gate(X, Message, Down, Forget, Gain, Weight, Y, rows, tasks, H: tl.constexpr, R: tl.constexpr, SHARED: tl.constexpr, EPS: tl.constexpr,
          BM: tl.constexpr, BK: tl.constexpr):
    # a fixed set of programs walks the (gate, row block) tasks, each block's four passes over its rows back to back
    blocks = tl.cdiv(rows, BM)
    for task in range(tl.program_id(0), tasks, tl.num_programs(0)):
        _gate_rows(X, Message, Down, Forget, Gain, Weight, Y, rows, (task // blocks).to(tl.int64), (task % blocks) * BM, H, R, SHARED, EPS, BM, BK)


def gate(x: torch.Tensor, message: torch.Tensor, down: torch.Tensor, forget: torch.Tensor, gain: torch.Tensor, weight: torch.Tensor,
         eps: float) -> torch.Tensor:
    """G message gates and the RMS norm of the sublayer each feeds, one launch: message [G, T, H] FP32; x [G, T, H] or,
    one input for every gate, [1, T, H]; down [G, R, 2 H], forget and gain [G, H, R] BF16; weight [G, H] FP32.
    -> BF16 [G, T, H]: rms((1 - tanh(forget low)) x + tanh(gain low) message) * weight, low = BF16(down [rms(x), rms(message)])."""
    g, t, h = message.shape
    out = torch.empty((g, t, h), dtype=torch.bfloat16, device=x.device)
    if t:
        # blocks of 16 rows walked by two programs an SM: 1.18 ms for a sliding layer's three gates over 1,678 tokens,
        # 1.38 with a program a block, 2.24 as two batched products between a norm and a merge kernel
        tasks = g * triton.cdiv(t, 16)
        _gate[(min(tasks, 96),)](x, message, down, forget, gain, weight, out, t, tasks, H=h, R=down.shape[1], SHARED=x.shape[0] != g, EPS=eps,
                                 BM=16, BK=256, num_warps=8, num_stages=1)
    return out


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))


@triton.jit
def _fp8_rows(ptr, offsets, mask, D: tl.constexpr):
    """[n, D] FP16 from E4M3 rows at byte `offsets`: each pair of values converted by one instruction."""
    half = tl.arange(0, D // 2)
    pair = tl.load(ptr.to(tl.pointer_type(tl.uint16)) + tl.multiple_of(offsets // 2, D // 2)[:, None] + half[None, :], mask=mask[:, None], other=0)
    lo, hi = tl.inline_asm_elementwise("{ .reg .b32 pair; cvt.rn.f16x2.e4m3x2 pair, $2; mov.b32 {$0, $1}, pair; }", "=h,=h,h", [pair],
                                       dtype=(tl.float16, tl.float16), is_pure=True, pack=1)
    return tl.interleave(lo, hi)


@triton.jit
def _queries(PQ, QScale, RScale, Proj, Positions, Qout, Rel, STRIDE: tl.constexpr, H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
             E: tl.constexpr, CLIP_Q: tl.constexpr, CLIP_R: tl.constexpr, TAU: tl.constexpr, FLOOR: tl.constexpr, ALPHA: tl.constexpr,
             HP: tl.constexpr, BE: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    head = tl.arange(0, HP)
    live = head < H
    tau = 1.0
    if TAU:  # a full-attention layer scales with the position's logarithm past FLOOR tokens
        tau = 1.0 + ALPHA * tl.log(tl.maximum((tl.load(Positions + t) + 1).to(tl.float32) / FLOOR, 1.0))
    d = tl.arange(0, D)
    q = tl.load(PQ + t * STRIDE + head[:, None] * D + d[None, :], live[:, None], other=0.0).to(tl.float32)
    q *= tl.load(QScale + head[:, None] * D + d[None, :], live[:, None], other=0.0)
    q *= (tl.minimum(CLIP_Q / (tl.sqrt(tl.sum(q * q, 1)) + 1e-6), 1.0) * tau)[:, None]
    tl.store(Qout + (t * H + head[:, None]) * D + d[None, :], q.to(tl.bfloat16), live[:, None])
    j = tl.arange(0, R)
    r = tl.load(PQ + t * STRIDE + H * D + head[:, None] * R + j[None, :], live[:, None], other=0.0).to(tl.float32)
    r *= tl.load(RScale + head[:, None] * R + j[None, :], live[:, None], other=0.0)
    r = (r * (tl.minimum(CLIP_R / (tl.sqrt(tl.sum(r * r, 1)) + 1e-6), 1.0) * tau)[:, None]).to(tl.bfloat16)
    for start in range(0, E, BE):  # the bias by backward distance: the relative states through their projection
        e = start + tl.arange(0, BE)
        bias = tl.dot(r, tl.load(Proj + j[:, None] * E + e[None, :]))
        tl.store(Rel + (t * H + head[:, None]) * E + e[None, :], bias.to(tl.bfloat16), live[:, None])


def queries(pq: torch.Tensor, q_scale: torch.Tensor, r_scale: torch.Tensor, proj: torch.Tensor, positions: torch.Tensor | None,
            clip_q: float, clip_r: float, floor: float, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The attention's query side from the query projection pq [T, >= 24 x (128 + 16)] BF16 (queries, then relative
    states): q = clip(q x q_scale) and the relative bias clip(r x r_scale) @ proj [16, extent], both scaled by
    tau(position) in a full-attention layer (`positions` [T] int64, else None). One launch for the PyTorch chain of
    scales, norms and the bias product. -> (q BF16 [T, 24, 128], bias BF16 [T, 24, extent])."""
    t = pq.shape[0]
    heads, dim = q_scale.shape
    rank, extent = proj.shape
    q = torch.empty((t, heads, dim), dtype=torch.bfloat16, device=pq.device)
    rel = torch.empty((t, heads, extent), dtype=torch.bfloat16, device=pq.device)
    if t:
        _queries[(t,)](pq, q_scale, r_scale, proj, pq if positions is None else positions, q, rel, STRIDE=pq.shape[1], H=heads, D=dim, R=rank,
                       E=extent, CLIP_Q=clip_q, CLIP_R=clip_r, TAU=positions is not None, FLOOR=floor, ALPHA=alpha,
                       HP=triton.next_power_of_2(heads), BE=256, num_warps=4)
    return q, rel


@triton.jit
def _attention_block(q, m_i, l_i, acc, K, V, Rel, Index, n, hi, kv_head, row, valid, pos, k_start, qk_scale, H: tl.constexpr, G: tl.constexpr,
                     D: tl.constexpr, E: tl.constexpr, WINDOW: tl.constexpr, FP8: tl.constexpr, BLOCK_N: tl.constexpr, NEAR: tl.constexpr):
    """One key block through the online softmax. Far (NEAR False): every key visible to every row and at least E
    behind it, so no mask and no bias."""
    LOG2E: tl.constexpr = 1.4426950408889634
    offs = tl.arange(0, D)
    keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
    key_valid = keys < hi
    at = tl.load(Index + keys, mask=key_valid, other=0).to(tl.int64) * (H // G) * D + kv_head * D  # the key's row of the pool
    if FP8:
        k = _fp8_rows(K, at, key_valid, D).to(q.dtype).T
    else:
        k = tl.load(K + at[None, :] + offs[:, None], mask=key_valid[None, :], other=0.0)
    s = tl.dot(q, k) * qk_scale
    if NEAR:
        behind = pos[:, None] - (k_start + keys)[None, :]
        visible = (behind >= 0) & key_valid[None, :]
        if WINDOW > 0:
            visible &= behind < WINDOW
        bias = tl.load(Rel + row[:, None] * E + behind, mask=visible & (behind < E) & valid[:, None], other=0.0)
        s = tl.where(visible, s + bias.to(tl.float32) * LOG2E, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    else:
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = m_new
    alpha = tl.exp2(m_i - m_safe)
    p = tl.exp2(s - m_safe[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    if FP8:
        v = _fp8_rows(V, at, key_valid, D)
    else:
        v = tl.load(V + at[:, None] + offs[None, :], mask=key_valid[:, None], other=0.0).to(tl.float16)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)  # weights and values as FP16: 11 bits of each weight
    return m_new, l_i, acc


@triton.jit(do_not_specialize=['tokens', 'keys_total', 'q_start', 'k_start'])
def _attention(Q, K, V, Rel, PQ, Index, Out, tokens, keys_total, q_start, k_start, scale, STRIDE: tl.constexpr, GATE: tl.constexpr,
               SINK: tl.constexpr, H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, E: tl.constexpr, WINDOW: tl.constexpr, FP8: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    TOKENS: tl.constexpr = BLOCK_M // G
    LOG2E: tl.constexpr = 1.4426950408889634
    first = tl.program_id(0) * TOKENS
    kv_head = tl.program_id(1)
    rows = tl.arange(0, BLOCK_M)
    tok = first + rows // G
    valid = (rows < TOKENS * G) & (tok < tokens)
    head = kv_head * G + rows % G
    row = (tok.to(tl.int64) * H + head)  # the row's (token, query head)
    pos = q_start + tok
    # key j of the sequence's list is the token at position k_start + j: the tile's rows see up to their own, and with a
    # window the last WINDOW of them
    lo = 0
    far = 0  # key blocks before this one are E or more behind the tile's first token: no bias, no mask
    if WINDOW > 0:
        lo = tl.maximum(q_start + first - (WINDOW - 1) - k_start, 0) // BLOCK_N
        far = lo
    else:
        far = tl.maximum(q_start + first - E + 1 - k_start, 0) // BLOCK_N
    hi = tl.minimum(q_start + tl.minimum(first + TOKENS, tokens) - k_start, keys_total)
    offs = tl.arange(0, D)
    q = tl.load(Q + row[:, None] * D + offs[None, :], mask=valid[:, None], other=0.0)
    if FP8:  # E4M3 keys convert to FP16 in one instruction a pair, to BF16 in three: FP16 queries (exact from BF16 unless
        q = q.to(tl.float16)  # |q| < 2^-14, a score change under 1e-5) took a 32,000-key chunk from 18.5 to 14.5 ms
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    # a full-attention tile at 32k keys spends over 90% of its blocks here: without the bias gather and masks a chunk
    # of 2,048 tokens over 32,000 keys took 18.5 ms instead of 23.8 (FP8 pool), 14.7 instead of 37.3 (BF16)
    for n in range(lo, far):
        m_i, l_i, acc = _attention_block(q, m_i, l_i, acc, K, V, Rel, Index, n, hi, kv_head, row, valid, pos, k_start, scale * LOG2E,
                                         H, G, D, E, WINDOW, FP8, BLOCK_N, False)
    for n in range(far, tl.cdiv(hi, BLOCK_N)):
        m_i, l_i, acc = _attention_block(q, m_i, l_i, acc, K, V, Rel, Index, n, hi, kv_head, row, valid, pos, k_start, scale * LOG2E,
                                         H, G, D, E, WINDOW, FP8, BLOCK_N, True)
    # the sink takes its share of the softmax, then the output gate
    mine = PQ + tok.to(tl.int64) * STRIDE  # the row's query projection: its sink logit and output gate
    sink = tl.load(mine + SINK + head, mask=valid, other=0.0).to(tl.float32) * LOG2E
    total = l_i + tl.exp2(sink - tl.where(m_i == float("-inf"), 0.0, m_i))
    gate = tl.load(mine[:, None] + GATE + head[:, None] * D + offs[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    out = acc / total[:, None] * _sigmoid(gate)
    tl.store(Out + row[:, None] * D + offs[None, :], out.to(tl.bfloat16), mask=valid[:, None])


def attention(q: torch.Tensor, rel: torch.Tensor, pq: torch.Tensor, gate_at: int, sink_at: int, k: torch.Tensor, v: torch.Tensor,
              index: torch.Tensor, q_start: int, k_start: int, window: int | None, scale: float) -> torch.Tensor:
    """One sequence's prefill attention, scores never written out: q [T, 24, 128] BF16 for the tokens at positions
    q_start ..; rel [T, 24, extent] BF16, the bias by backward distance; pq [T, width] BF16, the rows' query
    projection, which holds each head's output gate (128 values from column `gate_at`; its sigmoid multiplies the
    output) and sink logit (from `sink_at`). Key j is the token at position k_start + j, at row index[j] (int32) of k
    and v [rows, 8, 128], BF16 or E4M3. A row attends the keys up to its own position, the last `window` of them in a
    sliding-window layer; the sink logit joins the softmax and its share is dropped. -> BF16 [T, 24, 128].

    A tile of 42 tokens (their three query heads of one KV head) streams its key blocks through two tensor-core
    products with an online softmax between, as vLLM's SM12x attention does (see THIRD_PARTY_NOTICES.md)."""
    tokens, heads, dim = q.shape
    group = heads // k.shape[1]
    out = torch.empty_like(q)
    if tokens:
        rows = tokens * group
        block_m = 16 if rows <= 16 else 32 if rows <= 32 else 64 if rows <= 64 else 128
        # 2,048-token chunk: E4M3 key blocks of 128 take 0.9 ms a sliding visit, 7.6 over 17,000 keys (64: 1.7, 9.7); BF16
        # blocks of 128 in one stage 1.0 ms, but past 16k keys two stages of 64 win (32,000 keys: 14.7 against 17.6)
        fp8 = k.dtype == torch.float8_e4m3fn
        far = not window and q_start - k_start >= 16384
        block_n, stages = (128, 2) if fp8 else (64, 2) if far else (128, 1)
        if block_m < 128:
            block_n, stages = 32, 2
        _attention[(triton.cdiv(tokens, block_m // group), k.shape[1])](
            q, k, v, rel, pq, index, out, tokens, index.shape[0], q_start, k_start, scale, STRIDE=pq.shape[1], GATE=gate_at, SINK=sink_at,
            H=heads, G=group, D=dim, E=rel.shape[2], WINDOW=window or 0, FP8=fp8,
            BLOCK_M=block_m, BLOCK_N=block_n, num_warps=8 if block_m >= 128 else 4, num_stages=stages)
    return out


@triton.jit
def _softcap(X, Y, stride, N: tl.constexpr, INV: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = cols < N
    x = tl.load(X + row * stride + cols, live, other=0.0).to(tl.float32) * INV
    # libdevice's tanh flushes subnormal inputs to zero here; PyTorch's returns them, as tanh(x) = x there
    tl.store(Y + row * N + cols, tl.where(tl.abs(x) < 1.1754943508222875e-38, x, libdevice.tanh(x)) * CAP, live)


def softcap(logits: torch.Tensor, n: int, cap: float) -> torch.Tensor:
    """The head's BF16 logits [R, >= n] -> FP32 [R, n] = cap tanh(logits[:, :n] / cap), in one pass: the bits of
    `torch.tanh(logits[:, :n].float() / cap) * cap` (PyTorch divides by a scalar as a product with its FP32
    reciprocal), which took four passes over FP32 rows (128 rows: 3.41 -> 0.66 ms a verify round)."""
    rows = logits.shape[0]
    out = torch.empty((rows, n), dtype=torch.float32, device=logits.device)
    if rows:
        inv = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(cap, dtype=torch.float32))
        _softcap[(triton.cdiv(n, 4096), rows)](logits, out, logits.stride(0), N=n, INV=inv, CAP=cap, BLOCK=4096, num_warps=8,
                                               enable_fp_fusion=False)
    return out
