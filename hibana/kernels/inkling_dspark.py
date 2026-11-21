"""Kernels of Inkling's DSpark drafter (`model/inkling/dspark.py`): low-row projections, the recurrent head's gate and
its vocabulary step. Each streams its weights in K order (230-255 GB/s against cuBLAS's 200-230) and, where cuBLAS
keeps its 16-row tiles, gives the same bits; every elementwise step keeps PyTorch's BF16 round points, so drafts stay
the argmax of the same logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

MAX_ROWS = 32


@triton.jit
def _linear(X, W, Y, R, M, N, K, SX, SW, SY, SR, MODE: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr):
    """MODE 0: Y = BF16(x W^T). 1: Y (FP32) = BF16(R) + BF16(x W^T), the drafter's residual add (R may be Y).
    2: W holds [gate; up] (N rows each); Y = BF16(silu(BF16(x gate^T)) * BF16(x up^T))."""
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    if MODE == 2:
        acc_up = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        keep = (cols[None, :] < N) & (ks[:, None] < K)
        x = tl.load(X + rows[:, None] * SX + ks[None, :], (rows[:, None] < M) & (ks[None, :] < K), other=0)
        w = tl.load(W + cols[None, :].to(tl.int64) * SW + ks[:, None], keep, other=0)
        acc = tl.dot(x, w, acc)
        if MODE == 2:
            u = tl.load(W + (cols[None, :] + N).to(tl.int64) * SW + ks[:, None], keep, other=0)
            acc_up = tl.dot(x, u, acc_up)
    out = (rows[:, None] < M) & (cols[None, :] < N)
    if MODE == 0:
        tl.store(Y + rows[:, None] * SY + cols[None, :], acc.to(Y.dtype.element_ty), out)
    elif MODE == 1:
        r = tl.load(R + rows[:, None] * SR + cols[None, :], out, other=0).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + rows[:, None] * SY + cols[None, :], r + acc.to(tl.bfloat16).to(tl.float32), out)
    else:
        g = acc.to(tl.bfloat16).to(tl.float32)
        u = acc_up.to(tl.bfloat16).to(tl.float32)
        silu = tl.math.div_rn(g, 1.0 + libdevice.exp(-g))  # PyTorch's silu: x / (1 + expf(-x)), IEEE division
        tl.store(Y + rows[:, None] * SY + cols[None, :], (silu * u).to(tl.bfloat16), out)


# (block of output columns, K step, warps, stages) by mode: the fastest of a sweep at 8-32 rows on GB10, cold weights
_TILES = {0: (32, 256, 4, 4), 1: (32, 256, 4, 4), 2: (16, 256, 4, 3)}


def linear(x: torch.Tensor, weight: torch.Tensor, *, residual: torch.Tensor | None = None, swiglu: bool = False) -> torch.Tensor:
    """x [M, K] BF16 (rows unit-stride), weight [N, K] BF16 contiguous -> BF16 [M, N]; with `residual` (FP32 [M, N],
    updated in place and returned) the residual add; with `swiglu`, weight [2 N, K] -> BF16 [M, N]."""
    m, k = x.shape
    n = weight.shape[0] // 2 if swiglu else weight.shape[0]
    if m > (MAX_ROWS if swiglu else 16) or m == 1:
        # cuBLAS changes its tiling past 16 rows (past 32 for gate_up) and takes a GEMV for one row (a commit of one
        # token: 62 us against 44 here): there it keeps its own bits, which the drafts were measured with
        y = F.linear(x, weight)
        if swiglu:
            gate, up = y.float().chunk(2, dim=-1)
            return (F.silu(gate) * up).to(x.dtype)
        return y if residual is None else residual.copy_(residual.to(x.dtype).float() + y.float())
    mode = 2 if swiglu else 1 if residual is not None else 0
    if mode == 1:
        y = r = residual
    else:
        y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        r = y
    bn, bk, warps, stages = _TILES[mode]
    if m:
        _linear[(triton.cdiv(n, bn),)](x, weight, y, r, m, n, k, x.stride(0), weight.stride(0), y.stride(0), r.stride(0),
                                        MODE=mode, BN=bn, BK=bk, MP=max(16, triton.next_power_of_2(m)),
                                        num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return y


@triton.jit
def _gate(CTX, SC, MM, STATE, LATENT, RANK: tl.constexpr):
    """The recurrent head's step after its projection: raw = BF16(context + mm) (FP32 add), then PyTorch's BF16 ops
    keep = sigmoid(raw_0), state = keep * state + (1 - keep) * tanh(raw_1), latent = tanh(raw_2), each rounded."""
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, RANK)
    ctx = CTX + row * SC + c
    mm = MM + row * 3 * RANK + c
    g = (tl.load(ctx) + tl.load(mm)).to(tl.bfloat16).to(tl.float32)
    cand = (tl.load(ctx + RANK) + tl.load(mm + RANK)).to(tl.bfloat16).to(tl.float32)
    out = (tl.load(ctx + 2 * RANK) + tl.load(mm + 2 * RANK)).to(tl.bfloat16).to(tl.float32)
    state = tl.load(STATE + row * RANK + c).to(tl.float32)
    keep = tl.math.div_rn(1.0, 1.0 + libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    cand = libdevice.tanh(cand).to(tl.bfloat16).to(tl.float32)
    kept = (keep * state).to(tl.bfloat16).to(tl.float32)
    update = ((1.0 - keep).to(tl.bfloat16).to(tl.float32) * cand).to(tl.bfloat16).to(tl.float32)
    tl.store(STATE + row * RANK + c, (kept + update).to(tl.bfloat16))
    tl.store(LATENT + row * RANK + c, libdevice.tanh(out).to(tl.bfloat16))


def gate(context: torch.Tensor, mm: torch.Tensor, state: torch.Tensor, latent: torch.Tensor) -> None:
    """context [B, 3 r] FP32 (rows strided), mm [B, 3 r] FP32: state [B, r] BF16 advances in place, latent [B, r] BF16."""
    b, r = state.shape
    _gate[(b,)](context, context.stride(0), mm, state, latent, r, num_warps=2, enable_fp_fusion=False)


@triton.jit
def _vocab_step(LAT, F, FB, BASE, SB, PV, PI, V, R: tl.constexpr, CAP: tl.constexpr, BN: tl.constexpr, GROUPS: tl.constexpr,
                MP: tl.constexpr, B):
    """Each request's best (value, index) over this program's GROUPS * BN vocabulary rows of
    BF16(tanh(BF16(base + BF16(fold_bias + latent fold^T)) / CAP) * CAP): PyTorch's addmm, add, softcap and argmax."""
    pid = tl.program_id(0)
    rows = tl.arange(0, MP)
    ks = tl.arange(0, R)
    lat = tl.load(LAT + rows[:, None] * R + ks[None, :], rows[:, None] < B, other=0)
    best = tl.full((MP,), float("-inf"), tl.float32)
    where = tl.zeros((MP,), tl.int32)
    inv = 1.0 / CAP
    for g in range(GROUPS):
        cols = (pid * GROUPS + g) * BN + tl.arange(0, BN)
        ok = cols < V
        w = tl.load(F + cols[None, :].to(tl.int64) * R + ks[:, None], ok[None, :], other=0)
        acc = tl.dot(lat, w)
        acc = acc + tl.load(FB + cols, ok, other=0).to(tl.float32)[None, :]
        fold = acc.to(tl.bfloat16).to(tl.float32)
        live = (rows[:, None] < B) & ok[None, :]
        base = tl.load(BASE + rows[:, None] * SB + cols[None, :], live, other=0).to(tl.float32)
        logit = (base + fold).to(tl.bfloat16).to(tl.float32)
        # tensor / python scalar is a multiply by the reciprocal in PyTorch's CUDA kernels
        capped = (libdevice.tanh(logit * inv) * CAP).to(tl.bfloat16).to(tl.float32)
        capped = tl.where(live, capped, float("-inf"))
        value, index = tl.max(capped, axis=1, return_indices=True, return_indices_tie_break_left=True)
        better = value > best
        best = tl.where(better, value, best)
        where = tl.where(better, index + (pid * GROUPS + g) * BN, where)
    tl.store(PV + pid * MP + rows, best)
    tl.store(PI + pid * MP + rows, where)


@triton.jit
def _vocab_pick(PV, PI, TOK, ST, P, MP: tl.constexpr, PB: tl.constexpr, B):
    """The first index of the largest value over the programs' partial bests (programs in vocabulary order)."""
    row = tl.program_id(0)
    best = float("-inf")
    where = 0
    for p0 in range(0, P, PB):
        ps = p0 + tl.arange(0, PB)
        v = tl.load(PV + ps * MP + row, ps < P, other=float("-inf"))
        i = tl.load(PI + ps * MP + row, ps < P, other=0)
        m = tl.max(v, 0)
        first = tl.min(tl.where(v == m, i, 2147483647), 0)
        if m > best:
            best = m
            where = first
    tl.store(TOK + row * ST, where.to(tl.int64))


_STEP_BN, _STEP_GROUPS = 64, 4  # 782 programs over the 200k rows: 433 us a step, the fold alone reads in 401


def vocab_step(latent: torch.Tensor, fold_weight: torch.Tensor, fold_bias: torch.Tensor, base: torch.Tensor, softcap: float,
               out: torch.Tensor, scratch: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
    """latent [B, r] BF16, fold_weight [V, r] BF16, fold_bias [V] BF16, base [B, >= V] BF16 (rows strided) -> the
    tokens of softcap(base + fold_bias + latent fold^T), written to `out` (int64 [B], may be strided)."""
    b, r = latent.shape
    v = fold_weight.shape[0]
    if b > 16:  # one tensor-core tile of requests; more draft only eagerly, in PyTorch
        logits = base[:, :v] + torch.addmm(fold_bias, latent, fold_weight.T)
        out.copy_((torch.tanh(logits.float() / softcap) * softcap).to(latent.dtype).argmax(-1))
        return
    per = _STEP_BN * _STEP_GROUPS
    programs = triton.cdiv(v, per)
    mp = 16
    pv, pi = vocab_scratch(v, latent.device) if scratch is None else scratch
    _vocab_step[(programs,)](latent, fold_weight, fold_bias, base, base.stride(0), pv, pi, v, R=r, CAP=softcap, BN=_STEP_BN,
                             GROUPS=_STEP_GROUPS, MP=mp, B=b, num_warps=4, num_stages=3, enable_fp_fusion=False)
    _vocab_pick[(b,)](pv, pi, out, out.stride(0), programs, MP=mp, PB=256, B=b, num_warps=4)


def vocab_scratch(v: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    programs = triton.cdiv(v, _STEP_BN * _STEP_GROUPS)
    return (torch.empty(programs * 16, dtype=torch.float32, device=device), torch.empty(programs * 16, dtype=torch.int32, device=device))


# The drafter's norms sum in `kernels.rms`'s order (per thread, then a warp butterfly, then the warps in order), so
# fusing them keeps its bits.
@triton.jit
def _warp_sums(s, WARPS: tl.constexpr):
    """[WARPS * 32] thread values -> [WARPS]: each warp's xor-butterfly sum (lane 0's)."""
    s = tl.sum(tl.reshape(s, (WARPS, 2, 16)), axis=1)
    s = tl.sum(tl.reshape(s, (WARPS, 2, 8)), axis=1)
    s = tl.sum(tl.reshape(s, (WARPS, 2, 4)), axis=1)
    s = tl.sum(tl.reshape(s, (WARPS, 2, 2)), axis=1)
    return tl.sum(tl.reshape(s, (WARPS, 2, 1)), axis=1).reshape((WARPS,))


@triton.jit
def _in_order(s, WARPS: tl.constexpr):
    """((s_0 + s_1) + s_2) + ... over the warps, as the kernel's loop adds them."""
    w = tl.arange(0, WARPS)
    total = tl.sum(tl.where(w == 0, s, 0.0))
    for i in tl.static_range(1, WARPS):
        total = total + tl.sum(tl.where(w == i, s, 0.0))
    return total


@triton.jit
def _mix_chunk(T0, T1, T2, T3, S0, S1, S2, S3, row, c, a0, a1, a2, a3):
    """Columns `c` of row `row`'s mix of the four taps, FP32."""
    x = tl.load(T0 + row * S0 + c).to(tl.float32) * a0
    x = x + tl.load(T1 + row * S1 + c).to(tl.float32) * a1
    x = x + tl.load(T2 + row * S2 + c).to(tl.float32) * a2
    return x + tl.load(T3 + row * S3 + c).to(tl.float32) * a3


@triton.jit
def _mix_norm(T0, T1, T2, T3, S0, S1, S2, S3, ROWS, ALPHA, WEIGHT, OUT, T, EPS, HAS_ROWS: tl.constexpr, H: tl.constexpr):
    """out[m, t] = BF16(rms(sum_i alpha[m, i] f_i[row t]) * weight[m]) for the four taps; grid (T, mixes)."""
    t = tl.program_id(0)
    m = tl.program_id(1)
    row = (tl.load(ROWS + t) if HAS_ROWS else t).to(tl.int64)
    a0 = tl.load(ALPHA + m * 4)
    a1 = tl.load(ALPHA + m * 4 + 1)
    a2 = tl.load(ALPHA + m * 4 + 2)
    a3 = tl.load(ALPHA + m * 4 + 3)
    lane = tl.arange(0, 256)
    ss = tl.zeros((256,), tl.float32)
    for i in tl.static_range(9):
        c = lane + 256 * i
        x = _mix_chunk(T0, T1, T2, T3, S0, S1, S2, S3, row, c, a0, a1, a2, a3)
        ss = tl.fma(x, x, ss)
    total = _in_order(_warp_sums(ss, 8), 8)
    scale = libdevice.rsqrt(tl.math.div_rn(total, 2304.0) + EPS)
    out = OUT + (m * T + t).to(tl.int64) * H
    for i in tl.static_range(9):
        c = lane + 256 * i
        x = _mix_chunk(T0, T1, T2, T3, S0, S1, S2, S3, row, c, a0, a1, a2, a3)
        tl.store(out + c, (x * scale * tl.load(WEIGHT + m * H + c)).to(tl.bfloat16))


def mix_norm(taps, alpha: torch.Tensor, weight: torch.Tensor, eps: float, rows: torch.Tensor | None = None) -> torch.Tensor:
    """taps: four BF16 [N, 2304] (rows unit-stride), alpha FP32 [mixes, 4], weight FP32 [mixes, 2304] -> BF16
    [mixes, T, 2304]: the RMS norm of each mix of the taps' rows (`rows`, int64 [T], or all N), as the drafter's
    `mix` (PyTorch's mul and sum over the taps, then `kernels.rms`) gives it."""
    n, h = taps[0].shape
    t = n if rows is None else rows.shape[0]
    mixes = alpha.shape[0]
    out = torch.empty((mixes, t, h), dtype=torch.bfloat16, device=taps[0].device)
    if t:
        _mix_norm[(t, mixes)](*taps, *(x.stride(0) for x in taps), taps[0] if rows is None else rows, alpha, weight, out, t, eps,
                              HAS_ROWS=rows is not None, H=h, num_warps=8, enable_fp_fusion=False)
    return out


@triton.jit
def _norm_rotate(X, SX, COL0, W, POS, COS, SIN, CS, OUT, V, SV, VCOL0, PAGES, KC, VC, EPS,
                 HEADS: tl.constexpr, CACHE: tl.constexpr):
    """One head of one row: BF16(rms(x) * w) over its 128 columns (`kernels.rms`), then the draft RoPE with
    `kernels/draft_rope.py`'s BF16 round points; to OUT [T, HEADS, 128], or with CACHE to the K page and the row's V
    head to the V page."""
    t = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    j = tl.arange(0, 64)
    src = X + t * SX + COL0 + head * 128
    a = tl.load(src + j).to(tl.float32)
    b = tl.load(src + 64 + j).to(tl.float32)
    total = _in_order(_warp_sums(tl.join(a * a, b * b).permute(1, 0).reshape((128,)), 4), 4)
    scale = libdevice.rsqrt(tl.math.div_rn(total, 128.0) + EPS)
    a = (a * scale * tl.load(W + j)).to(tl.bfloat16).to(tl.float32)
    b = (b * scale * tl.load(W + 64 + j)).to(tl.bfloat16).to(tl.float32)
    pos = tl.load(POS + t)
    c = tl.load(COS + pos * CS + j).to(tl.bfloat16).to(tl.float32)
    s = tl.load(SIN + pos * CS + j).to(tl.bfloat16).to(tl.float32)
    lo = ((a * c).to(tl.bfloat16).to(tl.float32) - (b * s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    hi = ((b * c).to(tl.bfloat16).to(tl.float32) + (a * s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    if CACHE:
        page = tl.load(PAGES + t).to(tl.int64)
        dst = (page * HEADS + head) * 128
        tl.store(KC + dst + j, lo)
        tl.store(KC + dst + 64 + j, hi)
        vsrc = V + t * SV + VCOL0 + head * 128
        tl.store(VC + dst + j, tl.load(vsrc + j))
        tl.store(VC + dst + 64 + j, tl.load(vsrc + 64 + j))
    else:
        dst = (t * HEADS + head) * 128
        tl.store(OUT + dst + j, lo)
        tl.store(OUT + dst + 64 + j, hi)


def norm_rotate(x: torch.Tensor, col0: int, heads: int, weight: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                eps: float, *, values: tuple[torch.Tensor, int] | None = None, caches=None, pages: torch.Tensor | None = None):
    """Heads [col0, col0 + 128 heads) of x [T, *] BF16 rows -> rms-normed (FP32 weight [128]) and rotated at
    `positions`: a new BF16 [T, heads, 128], or with `caches` (K, V pages [P, 1, heads, 128]) written to `pages` [T]
    with the V heads at columns values[1] of values[0]."""
    t = x.shape[0]
    cache = caches is not None
    out = None
    if cache:
        v, vcol0 = values
        kc, vc = caches
    else:
        out = torch.empty((t, heads, 128), dtype=torch.bfloat16, device=x.device)
        v, vcol0, kc, vc, pages = x, 0, out, out, positions
    if t:
        _norm_rotate[(t, heads)](x, x.stride(0), col0, weight, positions, cos, sin, cos.stride(0), out if out is not None else kc,
                                 v, v.stride(0), vcol0, pages, kc, vc, eps, HEADS=heads, CACHE=cache, num_warps=2,
                                 enable_fp_fusion=False)
    return out
