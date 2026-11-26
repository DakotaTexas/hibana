# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini's bundled DSpark drafter (`mtp.safetensors`): a three-layer block drafter and a recurrent head.

A round drafts one block: the pending token and seven mask rows, which attend each other and the context, every
committed position's K/V projected from the target's features at visits 5, 23 and 41 and the final norm. Layers 0
and 2 see the last 1,024 positions, layer 1 all of them. The recurrent head then corrects each position's logits
from the previous token. Context K/V are per-page rows of the KV pool, so a cached prefix brings its draft context.
The equations follow vLLM's implementation (Apache-2.0, see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from hibana.kernels import inkling as kernels, inkling_dspark as draft_kernels
from hibana.model.inkling import config as c
from hibana.model.inkling.attention import project
from hibana.model.parallel_draft import frequencies

DRAFT_FILE = "mtp.safetensors"


class InklingDraft:
    def __init__(self, snap: Path, target, max_context: int):
        """`target`: the loaded `Inkling` model, whose embeddings and vocabulary head the draft shares."""
        cfg = json.loads((snap / "config.json").read_text())["mtp_config"]
        self.target, self.device = target, target.device
        self.hidden, self.heads, self.kv_heads, self.head_dim = (cfg["hidden_size"], cfg["num_attention_heads"],
                                                                 cfg["num_key_value_heads"], cfg["head_dim"])
        self.layers, self.block, self.rank = cfg["num_hidden_layers"], cfg["block_size"], cfg["markov_rank"]
        self.features = tuple(cfg["target_layer_ids"][:-1])  # visits whose input the draft reads; the final norm is last
        expected = (c.HIDDEN_SIZE, c.NUM_ATTENTION_HEADS, c.NUM_KV_HEADS, c.HEAD_DIM)
        if ((self.hidden, self.heads, self.kv_heads, self.head_dim) != expected or cfg["target_layer_ids"][-1] != c.NUM_VISITS
                or cfg.get("markov_head_type") != "rnn_hidden"):
            raise ValueError(f"unexpected Inkling draft: {cfg}")
        self.local = tuple(i in cfg["local_layer_ids"] for i in range(self.layers))
        self.window = cfg["sliding_window_size"]
        self.eps = cfg["rms_norm_eps"]
        with safe_open(str(snap / DRAFT_FILE), framework="pt", device="cpu") as source:
            w = {key: source.get_tensor(key).clone().to(self.device) for key in source.keys()}
        self.alpha = torch.softmax(w.pop("fuse_weight").float(), dim=-1)  # [layers + 1, taps]: each layer's mix, then the head's
        self.mask = w.pop("mask_embed")
        self.norm = w.pop("norm.weight").float()
        self.blocks = []
        for i in range(self.layers):
            p = f"layers.{i}."
            a = p + "self_attn."
            self.blocks.append({
                "input_norm": w.pop(p + "input_layernorm.weight").float(), "post_norm": w.pop(p + "post_attention_layernorm.weight").float(),
                "qkv": torch.cat([w.pop(a + "q_proj.weight"), w.pop(a + "k_proj.weight"), w.pop(a + "v_proj.weight")]).contiguous(),
                "context_kv": torch.cat([w.pop(a + "k_proj_ctx.weight"), w.pop(a + "v_proj_ctx.weight")]).contiguous(),
                "o": w.pop(a + "o_proj.weight"), "q_norm": w.pop(a + "q_norm.weight").float(), "k_norm": w.pop(a + "k_norm.weight").float(),
                "context_k_norm": w.pop(a + "k_norm_ctx.weight").float(),
                "gate_up": torch.cat([w.pop(p + "mlp.gate_proj.weight"), w.pop(p + "mlp.up_proj.weight")]).contiguous(),
                "down": w.pop(p + "mlp.down_proj.weight"),
            })
        m = "markov_head."
        self.head_norm = w.pop(m + "dflare_norm.weight").float()
        self.context_norms = torch.stack([w.pop(f"layers.{i}.dflare_norm.weight").float() for i in range(self.layers)])
        self.token_embed = w.pop(m + "markov_w1.weight")  # [padded vocabulary, rank]
        rnn = w.pop(m + "rnn_proj.weight").float()  # [3 rank, 2 rank + 2 H]: over [state | previous token | hidden | f_rnn]
        self.rnn_state, self.rnn_context = rnn[:, :2 * self.rank].contiguous(), rnn[:, 2 * self.rank:].contiguous()
        self.rnn_bias = w.pop(m + "rnn_proj.bias").float()
        # The head's latent reaches the logits through `hidden_out` and the vocabulary head: folded once, in FP64.
        out_w, out_b = w.pop(m + "hidden_out.weight").double(), w.pop(m + "hidden_out.bias").double()
        if w:
            raise ValueError(f"unread draft tensors: {sorted(w)}")
        vocab = target.target.vocab_size
        head = target.lm_head.dequantize()[:vocab]
        self.fold_weight = torch.empty((vocab, self.rank), dtype=torch.bfloat16, device=self.device)
        self.fold_bias = torch.empty(vocab, dtype=torch.bfloat16, device=self.device)
        for a in range(0, vocab, 8192):
            rows = head[a:a + 8192].double()
            self.fold_weight[a:a + 8192] = rows @ out_w
            self.fold_bias[a:a + 8192] = rows @ out_b
        self.cos, self.sin = frequencies({"head_dim": self.head_dim, "rope_parameters": {"rope_type": "default", "rope_theta": cfg["rope_theta"]}},
                                         max_context + self.block, self.device)
        self.caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._fixed: dict[int, dict] = {}
        self._scratch = None  # the vocabulary step's partial bests

    @property
    def bytes(self) -> int:
        tensors = [self.mask, self.token_embed, self.rnn_state, self.rnn_context, self.fold_weight, self.fold_bias, self.cos, self.sin,
                   self.context_norms]
        tensors += [t for block in self.blocks for t in block.values()]
        return sum(t.numel() * t.element_size() for t in tensors)

    def register(self, kv_pool) -> None:
        """The draft's K/V as per-page rows of the target's pool: [pages, 1, KV heads, head dim] BF16 a layer."""
        shape = (self.kv_heads, self.head_dim)
        self.caches = [tuple(kv_pool.register_page_aux(f"dspark_{t}.{i}", shape, torch.bfloat16).unsqueeze(1) for t in "kv")
                       for i in range(self.layers)]

    def rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """The RMS norm of x [..., W] (FP32 or BF16) with its gain, as BF16, in one launch."""
        return kernels.rms(x.reshape(-1, x.shape[-1]).contiguous(), weight, self.eps).view(x.shape)

    def write_context(self, taps: list[torch.Tensor], positions: torch.Tensor, pages: torch.Tensor, rows: torch.Tensor | None = None) -> None:
        """Committed positions' context K/V into their pages: taps, the target's features at the draft's three visits
        and the target's final norm (BF16 [N, H] each), at `rows` (int64 [T], or all N) at `positions` [T]. A
        layer's mix and norm are one kernel for all layers, then per layer the projection and one kernel for the
        key's norm, its rotation and both pages (as PyTorch operators the commit is about 50 launches and three
        host syncs a round)."""
        mixed = draft_kernels.mix_norm(taps, self.alpha[:self.layers], self.context_norms, self.eps, rows)
        kv_size = self.kv_heads * self.head_dim
        for i, (block, (keys, values)) in enumerate(zip(self.blocks, self.caches)):
            kv = draft_kernels.linear(mixed[i], block["context_kv"])
            draft_kernels.norm_rotate(kv, 0, self.kv_heads, block["context_k_norm"], positions, self.cos, self.sin, self.eps,
                                      values=(kv, kv_size), caches=(keys, values), pages=pages)

    def head_context(self, taps: list[torch.Tensor], rows: torch.Tensor | None = None) -> torch.Tensor:
        """The recurrent head's read of the last committed positions' features (`taps` at `rows`) -> BF16 [B, H]."""
        return draft_kernels.mix_norm(taps, self.alpha[self.layers:], self.head_norm[None], self.eps, rows)[0]

    def _constants(self, batch: int) -> dict:
        """A block's fixed inputs for `batch` requests, made once (outside graph capture: its warm-up runs first)."""
        if batch not in self._fixed:
            rows = batch * self.block
            self._fixed[batch] = {
                "offsets": torch.arange(self.block, device=self.device),
                "row_seq": torch.arange(rows, dtype=torch.int32, device=self.device) // self.block,
                "no_sink": torch.full((rows, self.heads), -1e30, dtype=torch.float32, device=self.device),
                "masks": self.mask.expand(batch, self.block, -1).contiguous(),
            }
        return self._fixed[batch]

    @torch.inference_mode()
    def forward(self, anchors: torch.Tensor, positions: torch.Tensor, pages: torch.Tensor, indptr: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """One block a request -> the mask rows' final hidden states, BF16 [B, block - 1, H].

        anchors, positions [B]: the pending token and its position; pages [B, block] int32: where the block's own K/V
        go (the request's reserved pages); indptr, indices int32: each request's pages by position, the block's last."""
        batch, block = anchors.shape[0], self.block
        rows = batch * block
        fixed = self._constants(batch)
        x = fixed["masks"].clone()
        x[:, 0] = self.target.embed_ids(anchors)
        x = x.view(rows, self.hidden)
        row_positions = (positions[:, None] + fixed["offsets"]).reshape(-1)
        seen = (positions.to(torch.int32) + block).repeat_interleave(block)  # every row sees the context and the whole block
        flat_pages = pages.reshape(-1)
        q_size, kv_size = self.heads * self.head_dim, self.kv_heads * self.head_dim
        residual = x.float()
        h = self.rms(residual, self.blocks[0]["input_norm"])
        for i, (block_w, (keys, values)) in enumerate(zip(self.blocks, self.caches)):
            qkv = draft_kernels.linear(h, block_w["qkv"])
            q = draft_kernels.norm_rotate(qkv, 0, self.heads, block_w["q_norm"], row_positions, self.cos, self.sin, self.eps)
            draft_kernels.norm_rotate(qkv, q_size, self.kv_heads, block_w["k_norm"], row_positions, self.cos, self.sin, self.eps,
                                      values=(qkv, q_size + kv_size), caches=(keys, values), pages=flat_pages)
            attn = kernels.attention(q, None, fixed["no_sink"], (keys, values), indptr, indices, fixed["row_seq"], seen,
                                     self.window + block if self.local[i] else None, self.head_dim ** -0.5, steps=block)
            draft_kernels.linear(attn.view(rows, q_size), block_w["o"], residual=residual)
            h = self.rms(residual, block_w["post_norm"])
            draft_kernels.linear(draft_kernels.linear(h, block_w["gate_up"], swiglu=True), block_w["down"], residual=residual)
            h = self.rms(residual, self.blocks[i + 1]["input_norm"] if i + 1 < self.layers else self.norm)
        return h.view(batch, block, self.hidden)[:, 1:]

    @torch.inference_mode()
    def draft(self, hidden: torch.Tensor, head_context: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """The greedy drafts [B, d] from the first d mask rows' hidden states [B, d, H], position by position:

            raw = rnn [state | embed(previous) | hidden_j | head_context] + bias          (FP32, rounded to BF16)
            state = sigmoid(raw_0) state + (1 - sigmoid(raw_0)) tanh(raw_1),  latent = tanh(raw_2)   (BF16 steps)
            logits_j = softcap(head(hidden_j) + fold latent)

        The gate and the vocabulary step (fold, softcap, argmax) are one kernel each (`kernels/inkling_dspark.py`)."""
        batch, depth = hidden.shape[:2]
        dtype = hidden.dtype
        base = project(self.target.lm_head, hidden.reshape(batch * depth, self.hidden)).view(batch, depth, -1)
        context = torch.cat([hidden, head_context[:, None].expand(-1, depth, -1)], dim=-1).float() @ self.rnn_context.T + self.rnn_bias
        state = torch.zeros((batch, self.rank), dtype=dtype, device=self.device)
        latent = torch.empty_like(state)
        tokens = torch.empty((batch, depth), dtype=torch.int64, device=self.device)
        if self._scratch is None:
            self._scratch = draft_kernels.vocab_scratch(self.fold_bias.shape[0], self.device)
        previous = anchors
        for j in range(depth):
            mm = torch.cat([state, F.embedding(previous, self.token_embed)], dim=-1).float() @ self.rnn_state.T
            draft_kernels.gate(context[:, j], mm, state, latent)
            draft_kernels.vocab_step(latent, self.fold_weight, self.fold_bias, base[:, j], c.LOGIT_SOFTCAP, tokens[:, j], self._scratch)
            previous = tokens[:, j]
        return tokens
