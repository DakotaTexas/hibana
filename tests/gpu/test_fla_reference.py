"""The vendored FLA kernels on the GDN layers' path (`kernels/fla/`) against PyTorch references in float64, the way
FLA's own tests check them (`fla.utils.assert_close`): the root-mean-square error relative to the reference's RMS.
Inputs are BF16, as served, in the 27B's and Flash-Next's geometry (16 key heads, 48 value heads of 128). A BF16 or E4M3
output is held to 1.5 times the ratio that rounding the reference itself to that type gives, so the kernel's own FP32
arithmetic may add little on top; an FP32 output (the recurrent state) to 1e-5."""
import pytest
import torch

H, HV, D = 16, 48, 128


def err_ratio(ref: torch.Tensor, out: torch.Tensor) -> float:
    """FLA's `get_err_ratio`: the RMS of the difference over the RMS of the reference."""
    ref, out = ref.double(), out.double()
    return ((ref - out).square().mean().sqrt() / (ref.square().mean().sqrt() + 1e-12)).item()


def assert_close(name: str, ref: torch.Tensor, out: torch.Tensor, ratio: float) -> None:
    assert not torch.isnan(out.double()).any(), f"{name}: NaN"
    got = err_ratio(ref, out)
    assert got <= ratio, f"{name}: error ratio {got:.3e} > {ratio:.3e}"


def gated_rms_norm(x, z, w, activation, eps=1e-6):
    """Grouped RMS norm (groups of D) times the gate, in float64."""
    rows = x.shape[0]
    xd, zd = x.double().view(rows, -1, D), z.double().view(rows, -1, D)
    gate = zd * torch.sigmoid(zd) if activation == "swish" else torch.sigmoid(zd)
    return (xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + eps) * w.double().view(-1, D) * gate).view(rows, -1)


@pytest.mark.parametrize("activation", ["swish", "sigmoid"])  # the 27B's gate, Flash-Next's
@pytest.mark.parametrize("rows", [1, 7, 2048])
def test_rms_norm_gated(rows, activation):
    from hibana.kernels.fla.layernorm_gated import rms_norm_gated
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, HV * D, generator=g, device="cuda") * 3).bfloat16()
    z = torch.randn(rows, 2 * HV * D, generator=g, device="cuda").bfloat16()[:, HV * D:]  # strided rows, as served
    w = torch.randn(HV * D, generator=g, device="cuda").bfloat16()
    ref = gated_rms_norm(x, z, w, activation)
    out = rms_norm_gated(x=x, weight=w, z=z, eps=1e-6, group_size=D, activation=activation)
    assert out.dtype == torch.bfloat16
    assert_close("gated norm", ref, out, 1.5 * err_ratio(ref, ref.bfloat16()))


@pytest.mark.parametrize("rows", [1, 7, 2048])
def test_rms_norm_gated_quant(rows):
    from hibana.kernels.fla.layernorm_gated import rms_norm_gated_quant
    g = torch.Generator(device="cuda").manual_seed(rows + 1)
    x = (torch.randn(rows, HV * D, generator=g, device="cuda") * 3).bfloat16()
    z = torch.randn(rows, HV * D, generator=g, device="cuda").bfloat16()
    w = torch.randn(HV * D, generator=g, device="cuda").bfloat16()
    scale = 0.1
    y = gated_rms_norm(x, z, w, "swish")
    ref = y / scale
    out = rms_norm_gated_quant(x=x, weight=w, z=z, scale=torch.tensor(scale, device="cuda"), eps=1e-6, group_size=D)
    assert out.dtype == torch.float8_e4m3fn and ref.abs().max() < 448  # no clamping in this range
    rounded = (y.to(torch.bfloat16).double() / scale).to(torch.float8_e4m3fn)  # the BF16 output divided, then E4M3
    assert_close("gated norm, E4M3", ref, out, 1.5 * err_ratio(ref, rounded))


@pytest.mark.parametrize("rows", [1, 33, 4096])
def test_l2norm(rows):
    from hibana.kernels.fla.l2norm import l2norm_fwd
    g = torch.Generator(device="cuda").manual_seed(rows + 2)
    x = (torch.randn(rows, H, D, generator=g, device="cuda") * 2).bfloat16()
    ref = x.double() / torch.sqrt(x.double().square().sum(-1, keepdim=True) + 1e-6)
    out = l2norm_fwd(x)
    assert out.dtype == torch.bfloat16
    assert_close("l2norm", ref, out, 1.5 * err_ratio(ref, ref.bfloat16()))


@pytest.mark.parametrize("batch", [1, 4, 16])
def test_gated_delta_rule_decode_step(batch):
    """One decode step of the gated delta rule on packed q|k|v rows and a slot pool, with a padding row (slot -1)."""
    from hibana.kernels.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
    g = torch.Generator(device="cuda").manual_seed(batch + 3)
    mixed = torch.randn(batch, 2 * H * D + HV * D, generator=g, device="cuda").bfloat16()
    a = torch.randn(batch, HV, generator=g, device="cuda").bfloat16()
    b = torch.randn(batch, HV, generator=g, device="cuda").bfloat16()
    A_log = torch.rand(HV, generator=g, device="cuda") * 2 - 1
    dt_bias = torch.randn(HV, generator=g, device="cuda").bfloat16()
    pool = torch.randn(batch + 2, HV, D, D, generator=g, device="cuda") * 0.05
    slots = torch.randperm(batch + 2, generator=g, device="cuda")[:batch].to(torch.int32)
    if batch > 1:
        slots[-1] = -1  # a padding row: zero output, no state touched
    before = pool.clone()
    out = torch.empty(batch, 1, HV, D, device="cuda", dtype=torch.bfloat16)
    scale = D ** -0.5
    fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv=mixed, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale,
                                                   initial_state=pool, out=out, ssm_state_indices=slots,
                                                   use_qk_l2norm_in_kernel=True)
    m = mixed.double()
    q, k = m[:, :H * D].view(batch, H, D), m[:, H * D:2 * H * D].view(batch, H, D)
    v = m[:, 2 * H * D:].view(batch, HV, D)
    q = (q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)).repeat_interleave(HV // H, 1) * scale
    k = (k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)).repeat_interleave(HV // H, 1)
    x = a.double() + dt_bias.double()
    decay = torch.exp(-torch.exp(A_log.double()) * torch.where(x <= 20, torch.log1p(torch.exp(x)), x))
    beta = torch.sigmoid(b.float()).bfloat16().double()  # the kernel rounds beta to b's type
    live = slots >= 0
    h = before[slots[live].long()].double() * decay[live, :, None, None]  # [n, HV, V, K]
    delta = (v[live] - (h * k[live, :, None, :]).sum(-1)) * beta[live, :, None]
    h = h + delta[..., None] * k[live, :, None, :]
    ref = (h * q[live, :, None, :]).sum(-1)
    assert_close("decode output", ref, out[live, 0], 1.5 * err_ratio(ref, ref.bfloat16()))
    assert_close("decode state", h, pool[slots[live].long()], 1e-5)
    assert (out[~live] == 0).all()
    untouched = torch.ones(batch + 2, dtype=torch.bool, device="cuda")
    untouched[slots[live].long()] = False
    assert torch.equal(pool[untouched], before[untouched])
