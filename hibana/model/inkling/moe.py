# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 routed experts: 128 softcapped-SwiGLU experts of width 1,152, the top 8 by sigmoid score.

    scores = sigmoid(W_r x)                top 8 of (scores + bias)
    weight = 8 * score / sum(the 8 scores)
    expert(x) = W_2 (softcap_4(g) * sigmoid(g) * softcap_25(u)),   [g; u] = W_13 x,   softcap_c(z) = c tanh(z / c)

The checkpoint stores the experts quantized: NVFP4 (two E2M1 codes a byte, an E4M3 scale per 16 values and a factor
amax / denominator per expert) in layers 0-19, MXFP8 (E4M3 values, a power-of-two scale per 32) in layers 20-23.
Every call runs on the checkpoint's tensors, its inputs quantized as the checkpoint requires of them
(hf_quant_config.json): an NVFP4 expert's input and activation as NVFP4 with a scale per token, an MXFP8 expert's as
MXFP8. A projection is one tensor-core launch over every (row, expert) pair (`kernels/inkling_prefill.py`: both sides
packed, on the block-scaled product): wide calls (prefill) sort the pairs by expert, so an expert's rows share its
weights' tiles; calls of up to `kernels.MOE_ROWS` rows (decode, verification, in captured graphs) multiply each pair
where it is, or past `prefill.GROUP_PAIRS` pairs find each expert's rows on the device, without the host's sync.
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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x [M, H] BF16 -> the routed experts' weighted sum, BF16."""
        nvfp4 = self.packed[0].dtype == torch.uint8
        if x.shape[0] <= kernels.MOE_ROWS and self.pair_calls:
            x = x.contiguous()
            ids, weights = kernels.route(x, self.router16, self.bias, c.EXPERTS_PER_TOKEN, c.ROUTE_SCALE)
            quantize = prefill.quantize_nvfp4 if nvfp4 else prefill.quantize_mxfp8
            act = prefill.pair_mm(quantize(x), ids, c.EXPERTS_PER_TOKEN, *self.packed[:3], swiglu=(c.GATE_SOFTCAP, c.UP_SOFTCAP))
            down = prefill.pair_mm(quantize(act), ids, 1, *self.packed[3:])
            return kernels.combine(down, None, weights)
        ids, weights = self.route(x)
        # The (token, expert) pairs sorted by expert, placed on the GPU (no host sync): each expert multiplies one
        # contiguous run of rows, all of them in one launch a projection, its first reading each token's row where
        # the pair needs it; the activation with its quantization and the tokens' weighted sums are a kernel each.
        plan = prefill.place(ids, c.NUM_EXPERTS)
        if nvfp4:
            hidden = prefill.grouped_mm4(*prefill.quantize_nvfp4(x.contiguous()), plan, True, *self.packed[:3])
            down = prefill.grouped_mm4(*prefill.swiglu_quantize(hidden, True, c.GATE_SOFTCAP, c.UP_SOFTCAP), plan, False, *self.packed[3:])
        else:
            hidden = prefill.grouped_mm8(*prefill.quantize_mxfp8(x.contiguous()), plan, True, *self.packed[:3])
            down = prefill.grouped_mm8(*prefill.swiglu_quantize(hidden, False, c.GATE_SOFTCAP, c.UP_SOFTCAP), plan, False, *self.packed[3:])
        return kernels.combine(down, plan.where.view_as(ids), weights.contiguous())
