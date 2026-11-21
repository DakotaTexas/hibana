"""Inkling 2's kernels (`csrc/inkling.cu`), compiled on first use. For calls of a few rows (decode, verification):
the relative-position attention over paged K/V, the BF16 projections and the steps between. `rms` and `combine`
serve prefill too, whose other kernels (and every call's expert products) are `kernels/inkling_prefill.py`."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent / "csrc" / "inkling.cu"
MOE_ROWS = 64  # calls up to this many rows (decode, verification) multiply their (row, expert) pairs unsorted
# Blocks a (row group, KV head) divides its keys among (K/V cold, attention and merge). A sliding layer's 1,024 keys
# (1,027 for four verified rows) in 12 splits fill one 96-key chunk a block: 17.8 us for one row, 23.2 for four (16
# splits: 18.3, 27.3). A full layer: 1,024 keys 18.9 us in 16 splits, 22.9 in 32; 32,000 keys 310 us either way, at
# 216 GB/s; fewer splits leave the merge less to read.
LOCAL_SPLITS, GLOBAL_SPLITS = 12, 16
GEMV_ROWS = 4  # rows `gemv` stages in a launch


@lru_cache(maxsize=1)
def _module():
    from torch.utils.cpp_extension import load

    from hibana.engine.config import current
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", current().cuda_arch)
    revision = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:16]
    return load(name=f"hibana_inkling_{revision}", sources=[str(_SRC)], extra_cuda_cflags=["-O3"], verbose=False)


def attention(q: torch.Tensor, rel: tuple[torch.Tensor, torch.Tensor] | None, sink: torch.Tensor, caches: tuple[torch.Tensor, torch.Tensor],
              indptr: torch.Tensor, indices: torch.Tensor, row_seq: torch.Tensor, row_len: torch.Tensor,
              window: int | None, scale: float, gate: torch.Tensor | None = None, gate_offset: int = 0, steps: int = 1) -> torch.Tensor:
    """q [R, heads, 128] BF16 -> [R, heads, 128] BF16; row r sees the first `row_len[r]` pages of sequence
    `row_seq[r]` (its own key last), the last `window` of them in a sliding-window layer. `rel` = (the rows'
    relative-position states [R, heads, d_rel] FP32, the bias projection [extent, d_rel] BF16), or None for no bias.
    With `gate` (BF16 rows), the output is multiplied by the sigmoid of the row's values from `gate_offset`. Rows
    come `steps` to a sequence (verification: the pending token and its drafts), which share their key loads."""
    none = q.new_empty(0)
    states, proj = rel if rel is not None else (none, none)
    return _module().attention(q, states, proj, sink, caches[0], caches[1], indptr, indices, row_seq, row_len,
                               window or 0, LOCAL_SPLITS if window else GLOBAL_SPLITS, scale, none if gate is None else gate, gate_offset, steps)


def mix(states: torch.Tensor, count: int, partial: torch.Tensor, queries: torch.Tensor, outputs: int, weight: torch.Tensor,
        gate: tuple | None = None, first_gate: int = 0) -> torch.Tensor:
    """A decode row's sublayer inputs -> BF16 [outputs, R, H]: each the softmax mix, by its query, of the row's first
    `count` normalized block states and rms(`partial`); with `gate` = (down, forget, gain, message sources, message
    queries) merged with its message by gate `first_gate + j`; then RMS-normalized with `weight[j]`."""
    none = partial.new_empty(0)
    down, forget, gain, sources, queries_m = gate if gate is not None else (none,) * 5
    return _module().mix(states, count, partial, queries, outputs, down, forget, gain, sources, queries_m, first_gate, weight)


def rms(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6, *, into: torch.Tensor | None = None, slot: int = 0) -> torch.Tensor:
    """x [R, W] BF16 or FP32 -> BF16(x rsqrt(mean(x^2) + eps) * weight), one launch; with `into` (FP32 [R, slots, W]),
    the FP32 norm without a weight, written to into[:, slot]."""
    none = x.new_empty(0)
    return _module().rms(x, none if weight is None else weight, eps, into is None, none if into is None else into, slot)


def conv(x: torch.Tensor, taps: torch.Tensor, state: torch.Tensor, slots: torch.Tensor, visit: int, offset: int, *, hidden: torch.Tensor | None = None,
         residual: torch.Tensor | None = None, windows: torch.Tensor | None = None, steps: int = 1) -> torch.Tensor:
    """c = BF16((conv(x) + x) / sqrt(5)) on a visit's hidden stream. With `residual`: BF16(residual + c), the next
    hidden. Otherwise the FP32 residual: `hidden` + c, or c alone (a layer that opens a block). Rows come `steps` to a
    sequence, whose stored window is at `slots[sequence]`: it advances in place (decode), or with `windows`
    [sequences, steps, visits, 3, width] stays and the window after each row goes there (verification)."""
    none = x.new_empty(0)
    windows = none if windows is None else windows
    if residual is not None:
        return _module().conv(x, taps, state, slots, visit, offset, none, residual, 2, windows, steps)
    return _module().conv(x, taps, state, slots, visit, offset, none if hidden is None else hidden, none, 0 if hidden is None else 1, windows, steps)


def prep(pq, pk, pv, q_scale, r_scale, k_taps, v_taps, positions, state, slots, visit: int, caches, pages, full: bool,
         tau_floor: float, tau_alpha: float, qk_clip: float, rel_clip: float, windows: torch.Tensor | None = None, steps: int = 1):
    """The projections of decode or verification rows -> (q, relative-position states, sink), their K and V written
    to `pages` of the pools; `slots`, `windows` and `steps` as `conv`'s."""
    return _module().prep(pq, pk, pv, q_scale, r_scale, k_taps, v_taps, positions, state, slots, visit, caches[0], caches[1],
                          pages, full, tau_floor, tau_alpha, qk_clip, rel_clip, pq.new_empty(0) if windows is None else windows, steps)


_route_done: dict[torch.device, torch.Tensor] = {}


def route(x: torch.Tensor, router: torch.Tensor, bias: torch.Tensor, topk: int, scale: float):
    """x [R, H] and router [E, H] BF16 -> (ids int64 [R, topk], weights FP32 [R, topk]): the top experts by sigmoid score plus bias, weights scale * score / sum."""
    done = _route_done.get(x.device)
    if done is None:  # the kernel's counters a group of rows, zero between calls; kept for the graphs that captured them
        done = _route_done[x.device] = torch.zeros(MOE_ROWS, dtype=torch.int32, device=x.device)
    return _module().route(x, router, bias, done, topk, scale)


def gemv(x: torch.Tensor, weight: torch.Tensor, first: int | None = None, second: int | None = None,
         scales: torch.Tensor | None = None) -> torch.Tensor:
    """x [M, K] BF16 (M <= GEMV_ROWS) and a matrix [N, K] -> x @ weight.T, BF16 [M, N], accumulated in FP32. The matrix
    may be up to three stacked ones, from rows `first` and `second`, each BF16 or E4M3 with its FP32 scale in
    `scales`; with x [S, M, K] the second reads x[1] and the third x[2]."""
    n = weight.shape[0]
    return _module().gemv(x, weight, n if first is None else first, n if second is None else second,
                          x.new_empty(0, dtype=torch.float32) if scales is None else scales)


def combine(down: torch.Tensor, where: torch.Tensor | None, weights: torch.Tensor) -> torch.Tensor:
    """down [P, H] BF16 with token t's k-th expert output at row where[t, k] (None: at row t topk + k); weights [T, k] FP32
    -> the weighted sums, BF16 [T, H]."""
    return _module().combine(down, down.new_empty(0) if where is None else where, weights)
