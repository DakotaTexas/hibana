"""The verification norms (`kernels/verify_norm.py`, and the native norm beside them) against their math in float64,
to FP32 rounding; their ownership of the inputs; graph replay."""

import pytest
import torch

U = 2.0 ** -24


def assert_rounds_from(out, exact, n):
    """`out` (BF16) is `exact` (float64) rounded, up to what FP32 arithmetic explains. Every element lies within half a
    BF16 step plus the error of an n-term FP32 sum of squares in any order (gamma_n), halved by the square root, and a
    few more roundings; and at most 1e-4 of them fall on the other side of a rounding midpoint. The kernels measure
    about 1e-5 on Triton 3.7.1 and 3.8.0; a variance taken from the BF16 sum gives about 5e-3, a numerator that skips
    the BF16 store 0.2."""
    out, exact = out.double(), exact.double()
    step = torch.pow(2.0, torch.floor(torch.log2(exact.abs().clamp_min(1e-30))) - 7)
    rel = n * U / (1 - n * U) / 2 + 8 * U
    assert ((out - exact).abs() <= step / 2 + rel * exact.abs()).all()
    assert (out != exact.to(torch.bfloat16).double()).double().mean() <= 1e-4



@pytest.mark.parametrize('rows', [1, 16, 64])
@torch.inference_mode()
def test_initial_verify_norm_graph_reads_new_embeddings(rows):
    from hibana.model.norm import GemmaRMSNorm
    from hibana.kernels.fp8_quant import quant_fp8

    torch.manual_seed(105)
    x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(5120, device='cuda', dtype=torch.bfloat16) * .1
    scale = torch.tensor(.0237, device='cuda')
    norm = GemmaRMSNorm(weight, 1e-6)
    for _ in range(2):
        norm.fused_add_quant(x, None, scale, verify=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, packed, retained = norm.fused_add_quant(x, None, scale, verify=True)
    for seed in [106, 107]:
        torch.manual_seed(seed)
        x.normal_()
        original = x.clone()
        xd = x.double()
        exact = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + 1e-6) * (1 + weight.double())
        graph.replay()
        assert torch.equal(x, original) and torch.equal(retained, original)
        assert_rounds_from(out, exact, 5120)
        assert torch.equal(packed.view(torch.uint8), quant_fp8(out, scale).view(torch.uint8))


@pytest.mark.parametrize('rows', [1, 16, 64])
@pytest.mark.parametrize('verify', [False, True])
@pytest.mark.parametrize('inplace', [False, True])
@torch.inference_mode()
def test_phase_specific_norm_uses_unrounded_variance(rows, verify, inplace):
    from hibana.model.norm import GemmaRMSNorm
    from hibana.kernels.fp8_quant import quant_fp8

    torch.manual_seed(841)
    x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x) * .01
    original, x_original = residual.clone(), x.clone()
    w = torch.randn(5120, device='cuda', dtype=torch.bfloat16) * .1
    scale = torch.tensor(.0237, device='cuda')
    summed = x.float() + residual.float()
    numerator = summed.bfloat16().double() if verify else summed.double()  # verify: the sum's BF16 store
    exact = numerator * torch.rsqrt(summed.double().square().mean(-1, keepdim=True) + 1e-6) * (1 + w.double())
    out, packed, retained = GemmaRMSNorm(w, 1e-6).fused_add_quant(
        x, residual, scale, inplace=inplace, verify=verify)
    assert torch.equal(retained, summed.bfloat16())
    assert (retained.data_ptr() == residual.data_ptr()) == inplace
    assert torch.equal(x, x_original)
    if not inplace:
        assert torch.equal(residual, original)
    assert_rounds_from(out, exact, 5120)
    assert torch.equal(packed.view(torch.uint8), quant_fp8(out, scale).view(torch.uint8))


@torch.inference_mode()
def test_verify_norm_graph_reads_new_inputs_and_preserves_ownership():
    from hibana.model.norm import GemmaRMSNorm
    torch.manual_seed(22)
    x = torch.randn(16, 5120, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(5120, device='cuda', dtype=torch.bfloat16) * .1
    norm = GemmaRMSNorm(weight, 1e-6)
    for _ in range(2):
        norm.fused_add(x, residual, verify=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, retained = norm.fused_add(x, residual, verify=True)
    for seed in [27, 28]:
        torch.manual_seed(seed)
        x.normal_(); residual.normal_()
        before = residual.clone()
        expected, summed = norm.fused_add(x, residual, verify=True)
        graph.replay()
        assert torch.equal(out, expected) and torch.equal(retained, summed)
        assert torch.equal(residual, before)


@pytest.mark.parametrize('tokens', [2, 16, 64])
@torch.inference_mode()
def test_compiled_gdn_norm_scalar_precision_and_graph(tokens):
    from hibana.kernels.verify_norm import verify_gated_norm
    torch.manual_seed(123)
    x = torch.randn(tokens, 48, 128, device='cuda', dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(128, device='cuda', dtype=torch.bfloat16)
    def exact():
        xd, zd = x.double(), z.double()
        return xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + 1e-6) * weight.double() * zd * torch.sigmoid(zd)
    for _ in range(2):
        verify_gated_norm(x, z, weight, 1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = verify_gated_norm(x, z, weight, 1e-6)
    for seed in [15, 16]:
        torch.manual_seed(seed); x.normal_(); z.normal_()
        want = exact()
        graph.replay()
        assert_rounds_from(out, want, 128)
