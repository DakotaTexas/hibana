# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 routed experts: 128 softcapped-SwiGLU experts of width 1,152, the top 8 by sigmoid score.

    scores = sigmoid(W_r x)                top 8 of (scores + bias)
    weight = 8 * score / sum(the 8 scores)
    expert(x) = W_2 (softcap_4(g) * sigmoid(g) * softcap_25(u)),   [g; u] = W_13 x,   softcap_c(z) = c tanh(z / c)

The experts run on the checkpoint's packed tensors (NVFP4 in layers 0-19, MXFP8 in 20-23), their inputs quantized as
hf_quant_config.json requires. Prefill sorts the (token, expert) pairs by expert; decode and verification multiply
each pair where it is, inside CUDA graphs (`kernels/inkling_prefill.py`). HIBANA_MOE_KEEP / HIBANA_MOE_BUDGET (off
by default) share experts across a decode or verification call through Nemotron's `route_budget`, which ranks the
same way.
"""

from __future__ import annotations

import torch

from hibana.kernels import inkling as kernels
from hibana.kernels import inkling_prefill as prefill
from hibana.model.inkling import config as c


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
        iff slots[q] != 0 (a graph's padding sits on the dummy slot 0)."""
        from hibana.model.nemotron.moe import route_budget
        ids, weights, _ = route_budget(x.float() @ self.router16.float().T, self.bias, c.EXPERTS_PER_TOKEN, 0, self.keep,
                                       self.budget, slots, steps, per_request=self.per_request, scale=c.ROUTE_SCALE,
                                       id_dtype=torch.int64)
        return ids, weights

    def __call__(self, x: torch.Tensor, slots: torch.Tensor | None = None, steps: int = 1) -> torch.Tensor:
        """x [M, H] BF16 -> the routed experts' weighted sum, BF16. `slots` and `steps` (decode and verification:
        the requests' state slots, rows a request) let HIBANA_MOE_KEEP share experts."""
        nvfp4 = self.packed[0].dtype == torch.uint8
        if x.shape[0] <= kernels.MOE_ROWS and self.pair_calls:
            x = x.contiguous()
            if self.keep and slots is not None:
                ids, weights = self.share(x, slots, steps)
            else:
                ids, weights = kernels.route(x, self.router16, self.bias, c.EXPERTS_PER_TOKEN, c.ROUTE_SCALE)
            quantize = prefill.quantize_nvfp4 if nvfp4 else prefill.quantize_mxfp8
            act = prefill.pair_mm(quantize(x), ids, c.EXPERTS_PER_TOKEN, *self.packed[:3], swiglu=(c.GATE_SOFTCAP, c.UP_SOFTCAP))
            down = prefill.pair_mm(quantize(act), ids, 1, *self.packed[3:])
            return kernels.combine(down, None, weights)
        shared = bool(self.keep) and slots is not None
        ids, weights = self.share(x, slots, steps) if shared else self.route(x)
        # pairs sorted by expert on the GPU (no host sync): each expert multiplies one run of rows, one launch a projection
        plan = prefill.place(ids, c.NUM_EXPERTS, repeats=shared)
        if nvfp4:
            hidden = prefill.grouped_mm4(*prefill.quantize_nvfp4(x.contiguous()), plan, True, *self.packed[:3])
            down = prefill.grouped_mm4(*prefill.swiglu_quantize(hidden, True, c.GATE_SOFTCAP, c.UP_SOFTCAP), plan, False, *self.packed[3:])
        else:
            hidden = prefill.grouped_mm8(*prefill.quantize_mxfp8(x.contiguous()), plan, True, *self.packed[:3])
            down = prefill.grouped_mm8(*prefill.swiglu_quantize(hidden, False, c.GATE_SOFTCAP, c.UP_SOFTCAP), plan, False, *self.packed[3:])
        return kernels.combine(down, plan.where.view_as(ids), weights.contiguous())
