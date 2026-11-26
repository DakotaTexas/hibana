# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 routed experts: 128 softcapped-SwiGLU experts of width 1,152, the top 8 by sigmoid score.

    scores = sigmoid(W_r x)                top 8 of (scores + bias)
    weight = 8 * score / sum(the 8 scores)
    expert(x) = W_2 (softcap_4(g) * sigmoid(g) * softcap_25(u)),   [g; u] = W_13 x,   softcap_c(z) = c tanh(z / c)

The experts run on the checkpoint's packed tensors (NVFP4 in layers 0-19, MXFP8 in 20-23), their inputs quantized as
hf_quant_config.json requires. Prefill and wide decode or verification calls sort the (token, expert) pairs by
expert; calls of up to `PAIR_ROWS` rows multiply each pair where it is (`kernels/inkling_prefill.py`).
HIBANA_MOE_KEEP / HIBANA_MOE_BUDGET (the fast and throughput profiles: keep 4, 12:48; off in exact) share experts
across a decode or verification call through Nemotron's `route_budget`, which ranks the same way.
"""

from __future__ import annotations

import torch

from hibana.kernels import inkling as kernels
from hibana.kernels import inkling_prefill as prefill
from hibana.model.inkling import config as c


LOGIT_GEMM_ROWS = 8  # sharing's router logits from this many rows on a cuBLAS product (2-7 rows: a slower split-K one)


def expert_names(prefix: str, name: str) -> list[str]:
    return [f"{prefix}.experts.{name}_weight{kind}" for kind in ("", "_scale", "_amax", "_outer_scale_denominator")]


class Moe:
    pair_calls = True  # False: every call sorts its pairs by expert, as prefill's do

    def __init__(self, w, prefix: str, device: torch.device):
        self.router16 = w.get(f"{prefix}.gate.weight").to(device).contiguous()  # [E, H] BF16, as the route kernels read it
        self.bias = w.get(f"{prefix}.gate.bias").to(device).float()
        raw13, raw2 = self._read(w, expert_names(prefix, "w13"), device), self._read(w, expert_names(prefix, "w2"), device)
        # the checkpoint's packed tensors, as every kernel reads them: weights, block scales (bytes), expert factors;
        # w13's rows alternate gate, up
        self.packed = (*self._pack(raw13), *self._pack(raw2))
        from hibana.engine.config import current
        cfg = current()
        self.keep = cfg.moe_keep if cfg.moe_keep < c.EXPERTS_PER_TOKEN else 0  # 0: every row's top 8
        self.budget, self.per_request = (cfg.moe_budget, cfg.moe_budget_per_request) if self.keep else (0, 0)

    @staticmethod
    def _read(w, names: list[str], device):
        return [w.get(name).to(device) if name in w else None for name in names]

    @staticmethod
    def _pack(raw):
        weight, scale, amax, denominator = raw
        factor = torch.ones(weight.shape[0], device=weight.device) if amax is None else (amax.float() / denominator.float()).reshape(-1)
        return weight.contiguous(), scale.view(torch.uint8).contiguous(), factor.contiguous()

    @property
    def expert_bytes(self) -> int:
        """Every packed expert's bytes; a decode token reads `EXPERTS_PER_TOKEN` of the `NUM_EXPERTS`."""
        return sum(t.numel() * t.element_size() for t in self.packed)

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return prefill.route(x.contiguous(), self.router16, self.bias, c.EXPERTS_PER_TOKEN, c.ROUTE_SCALE)

    def share(self, x: torch.Tensor, slots: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Budgeted routing of a decode or verification call: rows request-major, `steps` a request, request q real
        iff slots[q] != 0 (a graph's padding sits on the dummy slot 0). The FP32 logits of the BF16 operands: the
        decode route kernel's sums under `LOGIT_GEMM_ROWS` rows (2-4 us where the FP32 product took 7-14), a cuBLAS
        BF16 product with FP32 accumulation from there (4-5 us at 64-128 rows against 13-18), which sums in another
        order than the plain routes."""
        from hibana.model.nemotron.moe import route_budget
        x = x.contiguous()
        if x.shape[0] < LOGIT_GEMM_ROWS:
            logits = kernels.route_logits(x, self.router16)
        else:
            logits = torch.mm(x, self.router16.T, out_dtype=torch.float32)
        ids, weights, _ = route_budget(logits, self.bias, c.EXPERTS_PER_TOKEN, 0, self.keep,
                                       self.budget, slots, steps, per_request=self.per_request, scale=c.ROUTE_SCALE,
                                       id_dtype=torch.int64)
        return ids, weights

    def __call__(self, x: torch.Tensor, slots: torch.Tensor | None = None, steps: int = 1) -> torch.Tensor:
        """x [M, H] BF16 -> the routed experts' weighted sum, BF16. `slots` and `steps` (decode and verification:
        the requests' state slots, rows a request) let HIBANA_MOE_KEEP share experts."""
        nvfp4 = self.packed[0].dtype == torch.uint8
        x = x.contiguous()
        rows = x.shape[0]
        shared = bool(self.keep) and slots is not None
        if shared:
            ids, weights = self.share(x, slots, steps)
        elif rows <= kernels.MOE_ROWS and self.pair_calls:
            ids, weights = kernels.route(x, self.router16, self.bias, c.EXPERTS_PER_TOKEN, c.ROUTE_SCALE)
        else:
            ids, weights = self.route(x)
        if self.pair_calls and rows <= prefill.PAIR_ROWS:
            quantize = prefill.quantize_nvfp4 if nvfp4 else prefill.quantize_mxfp8
            act = prefill.pair_mm(quantize(x), ids, c.EXPERTS_PER_TOKEN, *self.packed[:3], swiglu=(c.GATE_SOFTCAP, c.UP_SOFTCAP), repeats=shared)
            down = prefill.pair_mm(quantize(act), ids, 1, *self.packed[3:], repeats=shared)
            return kernels.combine(down, None, weights)
        # pairs sorted by expert on the GPU (no host sync): each expert multiplies one run of rows, one launch a projection
        plan = prefill.place(ids, c.NUM_EXPERTS, repeats=shared)
        if nvfp4:
            hidden = prefill.grouped_mm4(*prefill.quantize_nvfp4(x), plan, True, *self.packed[:3])
            down = prefill.grouped_mm4(*prefill.swiglu_quantize(hidden, True, c.GATE_SOFTCAP, c.UP_SOFTCAP), plan, False, *self.packed[3:])
        else:
            hidden = prefill.grouped_mm8(*prefill.quantize_mxfp8(x), plan, True, *self.packed[:3])
            down = prefill.grouped_mm8(*prefill.swiglu_quantize(hidden, False, c.GATE_SOFTCAP, c.UP_SOFTCAP), plan, False, *self.packed[3:])
        return kernels.combine(down, plan.where.view_as(ids), weights.contiguous())
