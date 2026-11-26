"""Fused kernels of the Qwen vision tower (model/qwen_vision.py).

`rope_qk` rotates q and k straight out of the qkv projection into the patch-major layout flash attention reads, with the
torch path's FP32 arithmetic exactly (explicit rounding, no FMA contraction), so its output is bitwise the same.
`add_layer_norm` is a block's residual add and the LayerNorm after it in one pass.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from hibana.kernels.qsa.decode_prep import _add, _mul, _sub


@triton.jit
def _rope_kernel(qkv_ptr, cos_ptr, sin_ptr, out_ptr, n, HEADS: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
                 LANES: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)[:, None]
    lane = tl.arange(0, LANES)[None, :]
    mask = (t < n) & (lane < HALF)
    c1, c2 = tl.load(cos_ptr + t * D + lane, mask=mask), tl.load(cos_ptr + t * D + HALF + lane, mask=mask)
    s1, s2 = tl.load(sin_ptr + t * D + lane, mask=mask), tl.load(sin_ptr + t * D + HALF + lane, mask=mask)
    for col in tl.static_range(2 * HEADS):  # q heads, then k heads
        src = qkv_ptr + t * (3 * HEADS * D) + col * D
        x1 = tl.load(src + lane, mask=mask).to(tl.float32)
        x2 = tl.load(src + HALF + lane, mask=mask).to(tl.float32)
        dst = out_ptr + (col // HEADS) * n * HEADS * D + t * (HEADS * D) + (col % HEADS) * D
        tl.store(dst + lane, _sub(_mul(x1, c1), _mul(x2, s1)).to(tl.bfloat16), mask=mask)  # x1 cos + (-x2) sin
        tl.store(dst + HALF + lane, _add(_mul(x2, c2), _mul(x1, s2)).to(tl.bfloat16), mask=mask)  # x2 cos + x1 sin


def rope_qk(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """qkv [n, 3, heads, d] BF16 (contiguous), cos/sin [n, d] FP32 -> rotated q and k, [2, n, heads, d] BF16: the
    patch-major layout flash attention reads, so attention takes them (and v, a view of `qkv`) without copies."""
    n, three, heads, d = qkv.shape
    assert three == 3 and qkv.is_contiguous() and cos.is_contiguous() and sin.is_contiguous() and d % 2 == 0
    out = torch.empty((2, n, heads, d), dtype=torch.bfloat16, device=qkv.device)
    block = 16
    _rope_kernel[(triton.cdiv(n, block),)](qkv, cos, sin, out, n, HEADS=heads, D=d, HALF=d // 2,
                                           LANES=triton.next_power_of_2(d // 2), BLOCK=block)
    return out


@triton.jit
def _add_ln_kernel(h_ptr, y_ptr, w_ptr, b_ptr, h_out_ptr, x_out_ptr, eps, W: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < W
    s = (tl.load(h_ptr + row * W + cols, mask=m, other=0.0).to(tl.float32)
         + tl.load(y_ptr + row * W + cols, mask=m, other=0.0).to(tl.float32)).to(tl.bfloat16)
    tl.store(h_out_ptr + row * W + cols, s, mask=m)
    x = s.to(tl.float32)
    mean = tl.sum(x, axis=0) / W
    d = tl.where(m, x - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(d * d, axis=0) / W + eps)
    out = d * rstd * tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32) + tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(x_out_ptr + row * W + cols, out.to(tl.bfloat16), mask=m)


def add_layer_norm(h: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    """(h + y, LayerNorm(h + y)) for BF16 [n, w] rows in one pass: the residual add and the next norm."""
    n, w = h.shape
    h_out, x_out = torch.empty_like(h), torch.empty_like(h)
    _add_ln_kernel[(n,)](h, y, weight, bias, h_out, x_out, eps, W=w, BLOCK=triton.next_power_of_2(w), num_warps=8)
    return h_out, x_out
