# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 attention: 24 query and 8 KV heads of 128, a learned relative-position bias over the last 1,024 distances,
a per-query sink logit, a sigmoid output gate and a short causal convolution on the key and value streams.

    q = clip(scale * W_q x_q) [* tau]        k = clip(conv(W_k x_k))        v = rms(conv(W_v x_v))
    logit[i, j] = q_i . k_j / sqrt(128) + (clip(scale * W_r x_q) [* tau] @ P)[i - j]   for 0 <= i - j < 1024
    out = W_o (sigmoid(W_g x_q) * softmax([logits, W_s x_q])[:-1] @ v)

Sliding-window layers attend the last 1,024 keys with static convolution taps; full-attention layers attend every key,
scale by tau(position) and add per-token taps. `forward` is prefill (`FUSED` off: the PyTorch reference), `decode` is
decode and verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from hibana.kernels import inkling as kernels
from hibana.kernels import inkling_prefill as prefill
from hibana.model.flashnext.linear import TrunkLinear
from hibana.model.inkling import config as c

ATTENTION_BYTES = 1 << 30  # most FP32 logits one attention call materializes (rows are chunked to it)


def rms(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = c.RMS_NORM_EPS) -> torch.Tensor:
    """FP32 RMS normalization over the last dimension, with the layer's gain when it has one."""
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x if weight is None else x * weight


def l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Scale FP32 vectors longer than `threshold` down to it."""
    return x * (threshold / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)


def log_scale(positions: torch.Tensor) -> torch.Tensor:
    """tau = 1 + alpha log(max((position + 1) / floor, 1)), FP32 [T]."""
    return 1.0 + c.LOG_SCALING_ALPHA * torch.log(((positions + 1).float() / c.LOG_SCALING_FLOOR).clamp(min=1.0))


def padded(weight: torch.Tensor) -> torch.Tensor:
    """The projection [N, K] BF16 with zero rows up to a multiple of 16 outputs (the FP8 GEMM's shape rule); callers
    read the first N outputs."""
    pad = -weight.shape[0] % 16
    return torch.cat([weight, weight.new_zeros((pad, weight.shape[1]))]) if pad else weight.contiguous()


def padded_linear(weight: torch.Tensor, precision: str) -> TrunkLinear:
    return TrunkLinear.from_bf16(padded(weight), precision)


def project(linear: TrunkLinear, x: torch.Tensor) -> torch.Tensor:
    """`linear(x)` for decode-sized calls: a BF16 or FP8 matrix times up to `kernels.GEMV_ROWS` rows runs
    `kernels.gemv` (BF16 at one row: 232-245 GB/s against 160 through cuBLAS's GEMV; FP8 keeps the activations in
    BF16, where `TrunkLinear`'s GEMV quantizes them to E4M3)."""
    if linear.w3 is None and x.dim() == 2 and x.shape[0] <= kernels.GEMV_ROWS:
        return kernels.gemv(x.contiguous(), linear.weight, scales=None if linear.scale is None else linear.scale.reshape(1))
    return linear(x)


@dataclass
class Sequence:
    """One request's rows in a forward: `count` rows from `row`, the first at token position `start`; `pages` are the
    request's KV pages for positions 0 .. start + count - 1 (None without a pool) and `slot` its state slot."""
    row: int
    count: int
    start: int
    pages: torch.Tensor | None
    slot: int


@dataclass
class Layout:
    """The sequences of a forward and where their convolution windows live."""
    seqs: list[Sequence]
    state: torch.Tensor | None  # [slots, visits, CONV_KERNEL - 1, CONV_WIDTH] BF16, oldest row first; None without a pool
    checkpoints: list[tuple[int, int, int]]  # (sequence, position, slot): windows to store inside this forward
    # decode and verification: each sequence's planned pages (int32 index pointers and page ids, its rows' own last),
    # the rows' own pages, the sequences' state slots, and each row's sequence and visible keys (all int32)
    index: tuple[torch.Tensor, torch.Tensor] | None = None
    out_pages: torch.Tensor | None = None
    rows: torch.Tensor | None = None
    row_seq: torch.Tensor | None = None
    row_len: torch.Tensor | None = None
    steps: int = 1  # rows a sequence: 1 in decode, the pending token and its drafts in verification
    windows: torch.Tensor | None = None  # verification: [sequences, steps, visits, 3, width], the windows after each row


FUSED = True  # False: prefill's queries, convolutions and attention as PyTorch operators (the tests' reference)


def stream_rows(x: torch.Tensor, layout: Layout, visit: int, stream: str) -> list[torch.Tensor]:
    """Per sequence, [3 + n, C] BF16: the stream's three inputs before this forward (zeros before the sequence's first
    token), then its rows; stores each sequence's new window and the forward's checkpoints."""
    off, width = c.CONV_STREAMS[stream]
    keep = c.CONV_KERNEL - 1
    state = layout.state
    out = []
    for i, seq in enumerate(layout.seqs):
        prev = state[seq.slot, visit, :, off:off + width] if state is not None else x.new_zeros((keep, width))
        full = torch.cat([prev, x[seq.row:seq.row + seq.count]], dim=0)
        out.append(full)
        if state is not None:
            for owner, position, slot in layout.checkpoints:
                if owner == i:
                    at = position - seq.start
                    state[slot, visit, :, off:off + width] = full[at:at + keep]
            state[seq.slot, visit, :, off:off + width] = full[-keep:]
    return out


def _joined(x: torch.Tensor, add: torch.Tensor | None, out_dtype: torch.dtype | None) -> torch.Tensor:
    return x if add is None else (add.float() + x.float()).to(out_dtype or x.dtype)


def _per_head(x: torch.Tensor, clip: float | None, rms_eps: float | None) -> torch.Tensor:
    """A key or value stream's rows [T, 8 x 128]: each head clipped to an L2 norm of `clip`, or RMS-normalized."""
    if clip is None and rms_eps is None:
        return x
    heads = x.view(x.shape[0], c.NUM_KV_HEADS, c.HEAD_DIM)
    heads = l2_clip(heads.float(), clip) if clip is not None else rms(heads, eps=rms_eps)
    return heads.to(x.dtype).view(x.shape)


def static_conv(x: torch.Tensor, taps: torch.Tensor, layout: Layout, visit: int, stream: str, add: torch.Tensor | None = None,
                out_dtype: torch.dtype | None = None, clip: float | None = None, rms_eps: float | None = None) -> torch.Tensor:
    """c = BF16((conv(x) + x) / sqrt(kernel + 1)) with per-channel taps [C, 4] (FP32, the current token last); with
    `add`, add + c as `out_dtype`; with `clip` or `rms_eps` (a key or value stream), c with each head clipped or
    RMS-normalized."""
    fulls = stream_rows(x, layout, visit, stream)
    if FUSED and x.is_cuda:
        parts = [prefill.conv(full, taps, add=None if add is None else add[seq.row:seq.row + seq.count], out_dtype=out_dtype or x.dtype,
                              clip=clip, rms_eps=rms_eps) for full, seq in zip(fulls, layout.seqs)]
        return parts[0] if len(parts) == 1 else torch.cat(parts)
    win = torch.cat([full.unfold(0, c.CONV_KERNEL, 1) for full in fulls])
    out = (((win.float() * taps).sum(-1) + x.float()) / math.sqrt(c.CONV_KERNEL + 1)).to(x.dtype)
    return _joined(_per_head(out, clip, rms_eps), add, out_dtype)


def dynamic_conv(x: torch.Tensor, static: torch.Tensor, dynamic: torch.Tensor, layout: Layout, visit: int, stream: str,
                 clip: float | None = None, rms_eps: float | None = None) -> torch.Tensor:
    """A full-attention layer's key or value convolution: taps = static [heads, head_dim, 4] + the row's own
    `dynamic` [T, heads, 4], both indexed by backward distance; (conv(x) + x) / sqrt(2 kernel + 1), as BF16, then each
    head clipped (`clip`) or RMS-normalized (`rms_eps`)."""
    t = x.shape[0]
    fulls = stream_rows(x, layout, visit, stream)
    if FUSED and x.is_cuda:
        dynamic = dynamic.contiguous()
        parts = [prefill.conv(full, static.view(-1, c.CONV_KERNEL), dynamic=dynamic[seq.row:seq.row + seq.count], clip=clip, rms_eps=rms_eps)
                 for full, seq in zip(fulls, layout.seqs)]
        return parts[0] if len(parts) == 1 else torch.cat(parts)
    win = torch.cat([full.unfold(0, c.CONV_KERNEL, 1) for full in fulls]).flip(-1).float().view(t, c.NUM_KV_HEADS, c.HEAD_DIM, c.CONV_KERNEL)
    taps = static + dynamic.float().view(t, c.NUM_KV_HEADS, 1, c.CONV_KERNEL)
    return _per_head((((win * taps).sum(-1).view(t, -1) + x.float()) / math.sqrt(2 * c.CONV_KERNEL + 1)).to(x.dtype), clip, rms_eps)


def relative_attention(q: torch.Tensor, rel: torch.Tensor, sink: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       q_pos: torch.Tensor, k_start: int, window: int | None) -> torch.Tensor:
    """The attention as PyTorch operators (the reference: `prefill.attention` and the decode kernel serve). One
    sequence. q [n, 24, 128] BF16 at positions `q_pos`; rel [n, 24, extent] bias by backward distance; sink [n, 24];
    k, v [N, 8, 128] BF16 for positions k_start .. k_start + N - 1. -> FP32 [n, 24, 128]."""
    n, heads, dim = q.shape
    keys, groups = k.shape[0], k.shape[1]
    per = heads // groups
    extent = rel.shape[-1]
    kt = k.permute(1, 2, 0).float()  # [8, 128, N]
    vt = v.transpose(0, 1).float()  # [8, N, 128]
    k_pos = torch.arange(k_start, k_start + keys, device=q.device)
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    step = max(1, ATTENTION_BYTES // (heads * keys * 4))
    for a in range(0, n, step):
        b = min(n, a + step)
        rows = b - a
        qg = q[a:b].view(rows, groups, per, dim).transpose(0, 1).reshape(groups, rows * per, dim).float()
        logits = torch.bmm(qg, kt).view(groups, rows, per, keys).transpose(0, 1).reshape(rows, heads, keys)
        logits *= dim ** -0.5
        dist = q_pos[a:b, None] - k_pos[None, :]  # [rows, N]
        bias = rel[a:b].float().gather(-1, dist.clamp(0, extent - 1)[:, None, :].expand(rows, heads, keys))
        logits += bias.masked_fill_((dist >= extent)[:, None, :], 0.0)
        hidden = dist < 0 if window is None else (dist < 0) | (dist >= window)
        logits.masked_fill_(hidden[:, None, :], float("-inf"))
        # softmax over the keys and the sink, the sink's share dropped
        sinks = sink[a:b].float().unsqueeze(-1)
        top = torch.maximum(logits.amax(-1, keepdim=True), sinks)
        probs = logits.sub_(top).exp_()
        probs /= probs.sum(-1, keepdim=True) + (sinks - top).exp_()
        pg = probs.view(rows, groups, per, keys).transpose(0, 1).reshape(groups, rows * per, keys)
        out[a:b] = torch.bmm(pg, vt).view(groups, rows, per, dim).transpose(0, 1).reshape(rows, heads, dim)
    return out


class Attention:
    def __init__(self, w, prefix: str, local: bool, device: torch.device, precision: str):
        """`precision`: bf16 keeps the checkpoint's projections, fp8 quantizes them at load (`TrunkLinear`)."""
        self.local = local
        get = lambda name: w.get(f"{prefix}.{name}").to(device)  # noqa: E731
        stacked = lambda *names: padded(torch.cat([get(name) for name in names]))  # noqa: E731
        self.q_size = c.NUM_ATTENTION_HEADS * c.HEAD_DIM
        self.r_size = c.NUM_ATTENTION_HEADS * c.D_REL
        # everything the query input projects to, as one matrix (q | relative states | output gate | sink), then the
        # key and value projections, a full-attention layer's with its taps' projections under them
        parts = [stacked("wq_du.weight", "wr_du.weight", "gate_attn.weight", "attn_sink_proj.weight"),
                 stacked("wk_dv.weight", *(() if local else ("k_sconv.weight_proj",))),
                 stacked("wv_dv.weight", *(() if local else ("v_sconv.weight_proj",)))]
        self.qkv, self.qkv_scales = None, None
        if precision == "bf16":  # one matrix, its three projections views of it: decode multiplies them in one launch
            self.qkv = torch.cat(parts)
            parts = list(self.qkv.split([part.shape[0] for part in parts]))
        self.query, self.wk, self.wv = (TrunkLinear.from_bf16(part, precision) for part in parts)
        if precision == "fp8":  # the same for the E4M3 copies, each keeping its own scale
            linears = (self.query, self.wk, self.wv)
            self.qkv = torch.cat([lin.weight for lin in linears])
            self.qkv_scales = torch.cat([lin.scale.reshape(1) for lin in linears])
            for lin, view in zip(linears, self.qkv.split([lin.weight.shape[0] for lin in linears])):
                lin.weight = view
        self.wo = padded_linear(get("wo_ud.weight"), precision)
        self.q_scale = get("q_l2_clip.scale").float().view(c.NUM_ATTENTION_HEADS, c.HEAD_DIM).contiguous()
        self.r_scale = get("rel_l2_clip.scale").float().view(c.NUM_ATTENTION_HEADS, c.D_REL).contiguous()
        self.rel_proj = get("rel_logits_proj.proj").contiguous()  # [d_rel, extent]
        self.rel_proj_t = self.rel_proj.T.contiguous()  # by distance, for decode's attention kernel
        if local:
            self.k_taps = get("k_sconv.weight").float().view(c.KV_SIZE, c.CONV_KERNEL).contiguous()
            self.v_taps = get("v_sconv.weight").float().view(c.KV_SIZE, c.CONV_KERNEL).contiguous()
        else:
            shape = (c.NUM_KV_HEADS, c.HEAD_DIM, c.CONV_KERNEL)
            self.k_taps = get("k_sconv.static_w").float().view(c.CONV_KERNEL, c.KV_SIZE).T.reshape(shape).contiguous()
            self.v_taps = get("v_sconv.static_w").float().view(c.CONV_KERNEL, c.KV_SIZE).T.reshape(shape).contiguous()

    @property
    def bytes(self) -> int:
        return sum(lin.bytes for lin in (self.query, self.wk, self.wv, self.wo)) + self.rel_proj.numel() * 2

    def decode(self, x: torch.Tensor, positions: torch.Tensor, layout: Layout, visit: int, kv_pool) -> torch.Tensor:
        """`forward` for decode and verification rows, on the kernels. x [S, T, H] BF16: the query, key and value
        inputs (S = 3), or the one all three read (S = 1). The sequences' planned pages are `layout.index` and a row
        at position p sees its sequence's first p + 1 keys (`layout.row_len`)."""
        sets, t = x.shape[:2]
        nq, nk = self.query.out_features, self.wk.out_features
        if self.qkv is not None and t <= kernels.GEMV_ROWS:
            p = kernels.gemv(x if sets == 3 else x[0], self.qkv, nq, nq + nk, self.qkv_scales)
            pq, pk, pv = p[:, :nq], p[:, nq:nq + nk], p[:, nq + nk:]
        else:
            pq, pk, pv = project(self.query, x[0]), project(self.wk, x[sets // 2]), project(self.wv, x[-1])
        caches = kv_pool.cache(visit)
        q, rel, sink = kernels.prep(pq, pk, pv, self.q_scale, self.r_scale, self.k_taps, self.v_taps,
                                    positions, layout.state, layout.rows, visit, caches, layout.out_pages, not self.local,
                                    c.LOG_SCALING_FLOOR, c.LOG_SCALING_ALPHA, c.QK_CLIP, c.REL_CLIP, layout.windows, layout.steps)
        indptr, indices = layout.index
        attn = kernels.attention(q, (rel, self.rel_proj_t), sink, caches, indptr, indices, layout.row_seq, layout.row_len,
                                 None if not self.local else c.SLIDING_WINDOW, c.HEAD_DIM ** -0.5, pq, self.q_size + self.r_size, layout.steps)
        return project(self.wo, attn.view(t, self.q_size))

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, positions: torch.Tensor, layout: Layout,
                visit: int, kv_pool) -> torch.Tensor:
        t = xq.shape[0]
        heads, dim = c.NUM_ATTENTION_HEADS, c.HEAD_DIM
        fused = FUSED and xq.is_cuda
        window = c.SLIDING_WINDOW if self.local else None
        pq = self.query(xq)
        gate_at, sink_at = self.q_size + self.r_size, 2 * self.q_size + self.r_size
        if fused:  # one launch: the scales, clips, tau and the bias product
            q, rel = prefill.queries(pq, self.q_scale, self.r_scale, self.rel_proj, None if self.local else positions,
                                     c.QK_CLIP, c.REL_CLIP, c.LOG_SCALING_FLOOR, c.LOG_SCALING_ALPHA)
        else:
            q, r, gate, sink = pq[:, :sink_at + heads].split([self.q_size, self.r_size, self.q_size, heads], dim=-1)
            q = l2_clip(q.reshape(t, heads, dim).float() * self.q_scale, c.QK_CLIP)
            r = l2_clip(r.reshape(t, heads, c.D_REL).float() * self.r_scale, c.REL_CLIP)
            if not self.local:
                tau = log_scale(positions)[:, None, None]
                q, r = q * tau, r * tau
            q = q.to(xq.dtype)
            rel = r.to(xq.dtype) @ self.rel_proj  # [T, 24, extent]
        k, v = self.wk(xk), self.wv(xv)
        if self.local:
            k = static_conv(k, self.k_taps, layout, visit, "k", clip=c.QK_CLIP)
            v = static_conv(v, self.v_taps, layout, visit, "v", rms_eps=c.RMS_NORM_EPS)
        else:
            taps = c.NUM_KV_HEADS * c.CONV_KERNEL
            k = dynamic_conv(k[:, :c.KV_SIZE].contiguous(), self.k_taps, k[:, c.KV_SIZE:c.KV_SIZE + taps], layout, visit, "k", clip=c.QK_CLIP)
            v = dynamic_conv(v[:, :c.KV_SIZE].contiguous(), self.v_taps, v[:, c.KV_SIZE:c.KV_SIZE + taps], layout, visit, "v", rms_eps=c.RMS_NORM_EPS)
        k, v = k.view(t, c.NUM_KV_HEADS, dim), v.view(t, c.NUM_KV_HEADS, dim)
        gated = torch.empty_like(q)
        for seq in layout.seqs:
            rows = slice(seq.row, seq.row + seq.count)
            if kv_pool is None or seq.pages is None:  # the forward's own rows are the keys
                keys, values, first, index = k[rows].contiguous(), v[rows].contiguous(), seq.start, None
            else:
                kv_pool.write(visit, seq.pages[seq.start:seq.start + seq.count], k[rows], v[rows])
                first = max(0, seq.start - window + 1) if window else 0
                index = seq.pages[first:seq.start + seq.count]
                keys, values = kv_pool.cache(visit)
            if fused:
                if index is None:
                    index = torch.arange(seq.count, dtype=torch.int32, device=q.device)
                gated[rows] = prefill.attention(q[rows], rel[rows], pq[rows], gate_at, sink_at, keys.view(-1, c.NUM_KV_HEADS, dim),
                                                values.view(-1, c.NUM_KV_HEADS, dim), index.to(torch.int32).contiguous(), seq.start, first,
                                                window, dim ** -0.5)
                continue
            if index is not None:
                keys, values = keys[index.long(), 0].to(xq.dtype), values[index.long(), 0].to(xq.dtype)
            attn = relative_attention(q[rows], rel[rows], sink[rows], keys, values, positions[rows], first, window)
            gated[rows] = (attn * torch.sigmoid(gate[rows].reshape(-1, heads, dim).float())).to(xq.dtype)
        return self.wo(gated.view(t, self.q_size))
