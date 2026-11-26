# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini's text model for the engine: a looped decoder of 24 layers (48 visits a token) with block attention
residuals, the final norm and the softcapped head.

Layers come in blocks of six; every sublayer input is a softmax mix, by a learned query, of the normalized earlier
block states and the running block (`attn_res`). The core (layers 6-17) runs three passes, and from the second on
each core sublayer input merges a message from the previous pass through a rank-64 gate (`MessageGate`).

A visit is the target's layer: it keeps its own K/V and its own convolution windows. Prefill runs `prefill_visit`,
decode and verification `decode_visit`; `Layer.forward` is the same layer in PyTorch, the tests' reference.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from hibana.distributed import get_tp_info
from hibana.engine.config import current
from hibana.engine.context import ForwardContext, scratch_prefill_context
from hibana.engine.contracts import ForwardResult
from hibana.kernels import inkling as kernels
from hibana.kernels import inkling_prefill as prefill
from hibana.model.inkling import config as c
from hibana.model.inkling.attention import Attention, Layout, Sequence, padded_linear, project, rms, static_conv, stream_rows
from hibana.model.inkling.moe import Moe
from hibana.target import INKLING, Target
from hibana.weights.safetensors import SnapshotWeights

STATE_NAME = "inkling_conv"


def attn_res(blocks: list[torch.Tensor], partial: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """The block attention residual: `blocks` are the RMS-normalized FP32 states [T, H] of the finished blocks,
    `partial` the running one, `query` [Q, H] FP32. -> [T, Q, H] FP32, each query's softmax mix of the normalized states."""
    stacked = torch.stack([*blocks, rms(partial)], dim=1)  # [T, N, H]
    weights = torch.softmax(stacked @ query.T * stacked.shape[-1] ** -0.5, dim=1)  # [T, N, Q]
    return torch.einsum("tnq,tnh->tqh", weights, stacked)


class MessageGate:
    """(1 - forget) * hidden + gain * message, both gates tanh of a rank-64 read of the normalized pair."""

    def __init__(self, w, prefix: str, device: torch.device):
        self.down = w.get(f"{prefix}.down_weight").to(device)  # [rank, 2H] BF16
        self.forget16 = w.get(f"{prefix}.forget_weight").to(device)  # [H, rank] BF16, computed with in FP32
        self.gain16 = w.get(f"{prefix}.gain_weight").to(device)
        self.forget, self.gain = self.forget16.float(), self.gain16.float()

    def __call__(self, hidden: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([rms(hidden), rms(message)], dim=-1).to(self.down.dtype)
        low = F.linear(pair, self.down).float()
        return (1 - torch.tanh(F.linear(low, self.forget))) * hidden + torch.tanh(F.linear(low, self.gain)) * message


class Layer:
    def __init__(self, w, index: int, device: torch.device, precision: str):
        p = f"{c.PREFIX}.layers.{index}"
        self.index = index
        self.local = index in c.LOCAL_LAYERS
        self.boundary = index % c.BLOCK_LAYERS == 0
        norm = lambda name: w.get(f"{p}.{name}.weight").to(device).float()  # noqa: E731
        # a sliding layer normalizes one attention input; a full-attention layer its query, key and value inputs apart
        self.attn_norms = torch.stack([norm("attn_norm")] * 3 if self.local else [norm(f"{n}_input_norm") for n in "qkv"]).contiguous()
        self.mlp_norm = norm("mlp_norm").view(1, c.HIDDEN_SIZE).contiguous()
        self.attn = Attention(w, f"{p}.attn", self.local, device, precision)
        self.moe = Moe(w, f"{p}.mlp", device)
        self.attn_taps = w.get(f"{p}.attn_sconv.weight").to(device).float().view(c.HIDDEN_SIZE, c.CONV_KERNEL).contiguous()
        self.mlp_taps = w.get(f"{p}.mlp_sconv.weight").to(device).float().view(c.HIDDEN_SIZE, c.CONV_KERNEL).contiguous()
        queries = f"{c.PREFIX}.attn_res"
        self.attn_query = w.get(f"{queries}.attn_queries.{index - 1}").to(device).float().view(-1, c.HIDDEN_SIZE).contiguous() if index else None
        self.mlp_query = w.get(f"{queries}.mlp_queries.{index}").to(device).float().view(1, c.HIDDEN_SIZE).contiguous()
        self.gates = self.gate_stack = None
        if c.LOOP_START <= index < c.LOOP_END:  # the query, key, value and MLP inputs' message gates
            self.gates = [MessageGate(w, f"{p}.controllers.{name}", device) for name in ("attn", "k", "v", "mlp")]
            # the four stacked, as the gate kernels read them (decode's `kernels.mix`, prefill's `prefill.gate`)
            self.gate_stack = tuple(torch.stack([getattr(g, name) for g in self.gates]).contiguous() for name in ("down", "forget16", "gain16"))

    def gate(self, x: torch.Tensor, message: torch.Tensor, first: int, weight: torch.Tensor) -> torch.Tensor:
        """Prefill's message gates `first` .. `first + G` on FP32 inputs: message [G, T, H]; x [G, T, H], or [1, T, H]
        when the gates share an input. -> BF16 [G, T, H], each merge through its row of `weight`'s RMS norm, one launch."""
        gates = slice(first, first + message.shape[0])
        return prefill.gate(x, message, *(t[gates] for t in self.gate_stack), weight, c.RMS_NORM_EPS)

    def forward(self, hidden: torch.Tensor, blocks: list[torch.Tensor], messages: torch.Tensor | None, positions: torch.Tensor,
                layout: Layout, visit: int, kv_pool) -> torch.Tensor:
        """The layer as PyTorch operators between its attention and its experts: the reference the tests hold
        `decode_visit` and `prefill_visit` to. hidden: the running block [T, H] BF16; `blocks` gains this layer's
        input when it opens a block; `messages` [T, 4, H] FP32 in the passes after the first. -> the running block
        after this layer."""
        dtype = hidden.dtype
        if self.attn_query is None:
            inputs = [hidden] * 3
        else:
            mixed = attn_res(blocks, hidden, self.attn_query)
            inputs = [mixed[:, 0]] * 3 if self.local else list(mixed.unbind(1))
        if messages is not None:
            inputs = [gate(x, messages[:, i]) for i, (gate, x) in enumerate(zip(self.gates, inputs))]
        if self.local and messages is None:
            xq = xk = xv = rms(inputs[0], self.attn_norms[0]).to(dtype)
        else:
            xq, xk, xv = (rms(x, weight).to(dtype) for x, weight in zip(inputs, self.attn_norms))
        attn = self.attn.forward(xq, xk, xv, positions, layout, visit, kv_pool)
        attn = static_conv(attn, self.attn_taps, layout, visit, "attn").float()
        if self.boundary:
            blocks.append(rms(hidden))
            residual = attn
        else:
            residual = hidden.float() + attn
        x = attn_res(blocks, residual, self.mlp_query)[:, 0]
        if messages is not None:
            x = self.gates[3](x, messages[:, 3])
        mlp = self.moe(rms(x, self.mlp_norm[0]).to(dtype))
        mlp = static_conv(mlp, self.mlp_taps, layout, visit, "mlp")
        return (residual + mlp.float()).to(dtype)


def decode_visit(layer: Layer, hidden: torch.Tensor, states: torch.Tensor, count: int, message: tuple | None, positions: torch.Tensor,
                 layout: Layout, visit: int, kv_pool) -> torch.Tensor:
    """`Layer.forward` for decode or verification rows, on the kernels (`kernels/inkling.py`). `states` [rows, slots, H]
    holds the rows' first `count` normalized block states (a layer that opens a block adds one); `message` =
    (the two message sources [rows, 2, H], the layer's message queries [4, H]) in the passes after the first."""
    gate = None if message is None else (*layer.gate_stack, *message)
    if layer.attn_query is None:
        inputs = kernels.rms(hidden, layer.attn_norms[0]).unsqueeze(0)
    else:  # a sliding-window layer's query, key and value read one input unless a message gates each
        inputs = kernels.mix(states, count, hidden, layer.attn_query, 1 if layer.local and gate is None else 3, layer.attn_norms, gate, 0)
    attn = layer.attn.decode(inputs, positions, layout, visit, kv_pool)
    conv = dict(state=layout.state, slots=layout.rows, visit=visit, windows=layout.windows, steps=layout.steps)
    residual = kernels.conv(attn, layer.attn_taps, offset=c.CONV_STREAMS["attn"][0], hidden=None if layer.boundary else hidden, **conv)
    if layer.boundary:
        kernels.rms(hidden, into=states, slot=count)
        count += 1
    x = kernels.mix(states, count, residual, layer.mlp_query, 1, layer.mlp_norm, gate, 3)[0]
    return kernels.conv(layer.moe(x, layout.rows, layout.steps), layer.mlp_taps, offset=c.CONV_STREAMS["mlp"][0], residual=residual, **conv)


def conv_mix(x: torch.Tensor, taps: torch.Tensor, layout: Layout, visit: int, stream: str, add: torch.Tensor | None, states: torch.Tensor,
             count: int, queries: torch.Tensor, weight: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """A hidden stream's convolution joined to its residual and, in the same launch, the mix that reads it
    (`prefill.mix`), sequence by sequence -> (the mixes [Q, T, H], the residual [T, H] FP32)."""
    parts = []
    for full, seq in zip(stream_rows(x, layout, visit, stream), layout.seqs):
        rows = slice(seq.row, seq.row + seq.count)
        parts.append(prefill.mix(states[rows], count, queries, weight, full=full, taps=taps, add=None if add is None else add[rows]))
    if len(parts) == 1:
        return parts[0]
    return torch.cat([mixed for mixed, _ in parts], dim=1), torch.cat([residual for _, residual in parts])


def prefill_visit(layer: Layer, hidden: torch.Tensor, states: torch.Tensor, count: int, message: torch.Tensor | None, positions: torch.Tensor,
                  layout: Layout, visit: int, kv_pool) -> torch.Tensor:
    """`Layer.forward` for prefill rows with its sublayer inputs, convolutions and residuals on kernels. `states` as
    `decode_visit`'s; `message` [4, T, H] FP32, the layer's four messages, in the passes after the first."""
    if layer.attn_query is None:
        x = kernels.rms(hidden, layer.attn_norms[0])
        inputs = (x, x, x)
    elif message is None:
        mixed = prefill.mix(states, count, layer.attn_query, layer.attn_norms, own=hidden)
        inputs = (mixed[0],) * 3 if layer.local else mixed.unbind(0)
    else:
        inputs = layer.gate(prefill.mix(states, count, layer.attn_query, None, own=hidden), message[:3], 0, layer.attn_norms).unbind(0)
    attn = layer.attn.forward(*inputs, positions, layout, visit, kv_pool)
    if layer.boundary:
        kernels.rms(hidden, into=states, slot=count)
        count += 1
    # the attention stream's convolution, its residual and the MLP's input in one launch
    x, residual = conv_mix(attn, layer.attn_taps, layout, visit, "attn", None if layer.boundary else hidden, states, count,
                           layer.mlp_query, layer.mlp_norm if message is None else None)
    x = x[0] if message is None else layer.gate(x, message[3:], 3, layer.mlp_norm)[0]
    return static_conv(layer.moe(x), layer.mlp_taps, layout, visit, "mlp", add=residual, out_dtype=hidden.dtype)


class InklingExtras:
    """The page lists a prefill's attention reads (every visit's K/V sit in the request's pages), as the model's side
    inputs. Decode and verification read the pages the engine planned (`ForwardContext.decode_wrapper`,
    `paged_prefill_wrapper`), in a graph or not."""
    variants = ("dense",)

    def variant(self, seqs):
        return "dense"

    def for_prefill(self, chunks):
        return [pages for _, _, _, pages in chunks]

    def for_decode(self, reqs, seqs):
        return None

    def for_verify(self, reqs, drafts, seqs):
        return None

    def buffers(self, rows):
        return None

    def fill(self, side, reqs, drafts, seqs, rows, columns=None):
        """Decode and verification need no side buffers."""


class Inkling:
    def __init__(self, snap: Path, device: torch.device | str = "cuda", log=print, target: Target = INKLING, **_ignored):
        from hibana.cache.ledger import check_load_headroom
        if get_tp_info().size != 1:
            raise ValueError("Inkling serves on one Spark")
        check_load_headroom(target.weight_load_peak_bytes)
        self.device = torch.device(device)
        cfg = current()
        self.target = target
        self.eos = int(target.load_model_config(snap)["eos_token_id"])
        self.w3_identity = None
        self.precision = cfg.inkling_trunk  # bf16 | fp8: the attention projections and the head
        started = time.perf_counter()
        w = SnapshotWeights(snap)
        get = lambda name: w.get(f"{c.PREFIX}.{name}").to(self.device)  # noqa: E731
        self.embed = get("embed.weight")
        self.embed_norm = get("embed_norm.weight").float()
        self.layers: list[Layer] = []
        for i in range(c.NUM_LAYERS):
            self.layers.append(Layer(w, i, self.device, self.precision))
            if log and (i + 1) % 6 == 0:
                log(f"  loaded {i + 1}/{c.NUM_LAYERS} layers ({time.perf_counter() - started:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB)")
        # each core layer's message queries, [layers, 4, H]: query, key, value, MLP
        q_mlp = get("middle_cycle_cross_pass_queries").float().unflatten(0, (-1, 2))
        k_v = get("middle_cycle_cross_pass_kv_queries").float().unflatten(0, (-1, 2))
        self.message_queries = torch.cat([q_mlp[:, :1], k_v, q_mlp[:, 1:]], dim=1).contiguous()
        self.output_query = get("attn_res.output_query").float().view(1, c.HIDDEN_SIZE).contiguous()
        self.final_norm = get("norm.weight").float().contiguous()
        self.lm_head = padded_linear(get("unembed.weight")[:target.vocab_size], self.precision)
        w.close()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        # Bytes one decode token streams: every visit's attention weights and routed experts, and the head.
        visit_bytes = {i: layer.attn.bytes + layer.moe.expert_bytes * c.EXPERTS_PER_TOKEN // c.NUM_EXPERTS for i, layer in enumerate(self.layers)}
        self.stream_bytes = sum(visit_bytes[i] for i, _ in c.VISITS) + self.lm_head.bytes
        self._extras = InklingExtras()
        # The bundled DSpark drafter, when the server speculates with it (HIBANA_PROPOSER=dspark): its context K/V are
        # rows of the KV pool's pages (`register_pages`), so it loads with the model.
        self.draft = None
        if cfg.spec not in (None, "", "0") and cfg.proposer in (None, "dspark"):
            from hibana.model.inkling.dspark import InklingDraft
            # as long as the engine serves (HIBANA_MAX_CONTEXT), not the target's default: its rotary table and page
            # lists end there
            self.draft = InklingDraft(snap, self, cfg.max_context or target.context_length)
        self.state_slots = 0  # the state pool's slots, once registered
        # the SSD store keys stored prefixes by the arithmetic that produced them: the trunk's format and expert sharing
        # (K/V, convolution windows and the drafter's context all differ between them)
        self.arithmetic_identity = {"trunk": self.precision}
        if self.layers[0].moe.keep:
            self.arithmetic_identity["moe_sharing"] = f"{cfg.moe_keep}:{cfg.moe_budget_per_request}:{cfg.moe_budget}"
        if log:
            allocated = torch.cuda.memory_allocated() / 2**30 if self.device.type == "cuda" else 0.0
            log(f"  weights loaded in {time.perf_counter() - started:.0f}s, {allocated:.1f} GiB allocated ({self.stream_bytes / 1e9:.2f} GB per decode token)")

    def linears(self):
        """The attention projections and the head (`weights/linear.py`); the experts are the MoE's own."""
        for layer in self.layers:
            for name in ("query", "wk", "wv", "wo"):
                yield f"layers.{layer.index}.attn.{name}", getattr(layer.attn, name)
        yield "lm_head", self.lm_head

    def forward_extras(self):
        return self._extras

    def register_pages(self, kv_pool) -> None:
        """The drafter's context K/V, when it is loaded."""
        if self.draft is not None:
            self.draft.register(kv_pool)

    def register_state(self, pool) -> torch.Tensor:
        """Every visit's convolution windows: the last three inputs of its four streams, oldest first."""
        self.state_slots = pool.num_slots + len(pool.scratch)
        return pool.register_aux(STATE_NAME, c.CONV_STATE_SHAPE, torch.bfloat16)

    def _layout(self, ctx: ForwardContext, rows: int) -> Layout:
        state = self.register_state(ctx.state_pool) if ctx.state_pool is not None else None
        pages = ctx.extras
        if ctx.mode in ("decode", "verify"):  # on the kernels, over the pages the engine planned (FlashInfer's wrapper holds them)
            decode = ctx.mode == "decode"
            wrapper = ctx.decode_wrapper if decode else ctx.paged_prefill_wrapper
            steps = 1 if decode else ctx.verify_steps
            row_seq = torch.arange(rows, dtype=torch.int32, device=self.device)
            return Layout([], state, [], (wrapper._paged_kv_indptr_buf, wrapper._paged_kv_indices_buf),
                          ctx.out_pages if decode else ctx.pages, ctx.slots if decode else ctx.seq_slots,
                          row_seq if decode else row_seq // steps, (ctx.positions + 1).to(torch.int32), steps,
                          None if decode else ctx.verify_aux_windows[STATE_NAME])
        if ctx.batched:
            starts = ctx.cu_seqlens.tolist()
            seqs = [Sequence(starts[i], n, ctx.seq_prefix_lens[i], None if pages is None else pages[i], slot)
                    for i, (n, slot) in enumerate(zip(ctx.seq_lens, ctx.seq_slots.tolist()))]
        else:
            own = None
            if pages is not None:
                own = pages[0]
            elif ctx.pages is not None:
                own = ctx.pages if ctx.prefix_pages is None else torch.cat([ctx.prefix_pages, ctx.pages])
            seqs = [Sequence(0, rows, ctx.prefix_len, own, ctx.slot)]
        owners = ctx.checkpoint_seqs or [0] * len(ctx.checkpoint_slots)
        return Layout(seqs, state, list(zip(owners, ctx.checkpoint_positions, ctx.checkpoint_slots)))

    def embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = F.embedding(input_ids, self.embed)
        return kernels.rms(x, self.embed_norm)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, ctx: ForwardContext, capture: bool = False) -> ForwardResult:
        layout = self._layout(ctx, input_ids.shape[0])
        hidden = self.embed_ids(input_ids)
        if ctx.media_rows is not None:
            raise NotImplementedError("Inkling serves text only")
        out: dict = {"input_ids": input_ids, "positions": ctx.positions}
        if capture:
            out["embed"] = hidden
            out["layers"] = []
        out["norm"] = self._visits(hidden, ctx, layout, out, capture)
        return out

    def _visits(self, hidden: torch.Tensor, ctx: ForwardContext, layout: Layout, out: dict, capture: bool) -> torch.Tensor:
        """The 48 visits -> the final norm. Decode and verification rows (`layout.index`) run `decode_visit`, prefill
        rows `prefill_visit`; the rows' normalized block states live in `states`, `count` of them so far."""
        prefill_rows = layout.index is None
        states = torch.empty((hidden.shape[0], c.NUM_LAYERS // c.BLOCK_LAYERS + 1, c.HIDDEN_SIZE), dtype=torch.float32, device=hidden.device)
        count = 0
        prelude = message = sources = None
        for visit, (index, loop_pass) in enumerate(c.VISITS):
            layer = self.layers[index]
            if index == c.LOOP_START:
                if loop_pass == 0:
                    prelude = (hidden, count)
                else:  # the previous pass's two block states (the second still running) are this pass's message sources
                    # copies, not `.contiguous()`: one row's slice is a view of the slot this pass rewrites
                    if prefill_rows:
                        sources = (states[:, count - 1:count].clone(), hidden)
                    else:
                        sources = states[:, count - 1:count + 1].clone()
                        kernels.rms(hidden, into=sources, slot=1)
                    hidden, count = prelude
            if loop_pass:
                queries = self.message_queries[index - c.LOOP_START]
                message = prefill.mix(sources[0], 1, queries, None, own=sources[1]) if prefill_rows else (sources, queries)
            if visit in ctx.feature_request.layers:
                out.setdefault("features", {})[visit] = hidden.clone()
            step = prefill_visit if prefill_rows else decode_visit
            hidden = step(layer, hidden, states, count, message if loop_pass else None, ctx.positions, layout, visit, ctx.kv_pool)
            count += layer.boundary
            if capture:
                out["layers"].append((hidden, None))
        weight = self.final_norm.view(1, -1)
        if prefill_rows:
            return prefill.mix(states, count, self.output_query, weight, own=hidden)[0]
        return kernels.mix(states, count, hidden, self.output_query, 1, weight)[0]

    def prefill(self, input_ids: torch.Tensor, capture: bool = False) -> dict:
        """Stand-alone prefill without pools (parity tools)."""
        return self.forward(input_ids, scratch_prefill_context(input_ids.shape[0], self.device), capture)

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return prefill.softcap(project(self.lm_head, hidden), self.target.vocab_size, c.LOGIT_SOFTCAP)

    @torch.inference_mode()
    def head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The head's FP32 logits before the softcap: what a DFlash2 selector scores, softcapping after its correction."""
        return project(self.lm_head, hidden)[:, :self.target.vocab_size].float()
