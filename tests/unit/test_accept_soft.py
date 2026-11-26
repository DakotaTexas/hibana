"""Relaxed acceptance for sampled requests (SamplingParams.accept_soft s): a draft keeps its uniform, scaled by e^-s,
and is accepted when u * q(y) < e^s * p(y). At one verify slot the first emitted token then has the law
    P(x) = min(q(x), e^s p(x)) + (1 - alpha) (p(x) - q(x))+ / Z,
with alpha = sum_y min(q(y), e^s p(y)), so P(x) <= e^s p(x); s = 0 is exact sampling. With a gate (accept_top_k K,
accept_margin M) e^s applies only on G, the tokens p ranks within its top K and within M nats of its top token, and 1
elsewhere, so a token outside G never comes out more often than p gives it. Checked by Monte Carlo through
verify_rows and verify_batch for MTP's point-mass drafts and DFlash2's draft laws (the fused kernel:
tests/gpu/test_sampled_kernels.py)."""
import math

import pytest
import torch

from hibana.api import validation
from hibana.api.validation import _sampling_from
from hibana.engine.rejection import verify_batch, verify_rows
from hibana.engine.sampler import InvalidRequest, SamplingParams

P = torch.tensor([0.60, 0.30, 0.07, 0.03])  # the target's law
Q = torch.tensor([0.90, 0.06, 0.02, 0.02])  # a drafter sharper than its target
SOFTS = (0.0, 0.25, 0.5)


def params(s, **kw):
    return SamplingParams(temperature=1.0, top_k=-1, top_p=1.0, accept_soft=s, **kw)


def gate_set(top_k, margin):
    """G: the tokens P ranks within its top top_k and within margin nats of its top token."""
    rank = (P.unsqueeze(0) > P.unsqueeze(1)).sum(1)
    return (rank < top_k) & (P.max().log() - P.log() <= margin)


def soft_law(s, y=None, gate=None):
    """The first token's law: DFlash2's draft law Q, or (y given) MTP's point mass at y; e^s only on the gate's G."""
    q = Q if y is None else torch.nn.functional.one_hot(torch.tensor(y), 4).float()
    c = torch.full((4,), math.exp(s)) if gate is None else torch.where(gate_set(*gate), math.exp(s), 1.0)
    a = torch.minimum(q, c * P)
    left = (P - q).clamp(min=0)
    return a + (1 - a.sum()).clamp(min=0) * left / left.sum()


def first_tokens(tokens, accepted, correction):
    return torch.where(accepted > 0, tokens[:, 0], correction)


def dflash2_rows(n, seed):
    tokens = torch.multinomial(Q, n, replacement=True, generator=torch.Generator().manual_seed(seed)).view(n, 1)
    return tokens, P.log().expand(n, 2, 4).contiguous(), Q.expand(n, 1, 4).contiguous(), torch.arange(4).expand(n, 1, 4).contiguous()


def generators(n):
    return [torch.Generator().manual_seed(i) for i in range(n)]


@pytest.mark.parametrize("draft", ["mtp-top", "mtp-tail", "dflash2"])
def test_verify_batch_law(draft):
    n = 40_000
    torch.manual_seed(1)
    logits = P.log().expand(n, 2, 4).contiguous()
    for s in SOFTS:
        if draft == "dflash2":
            tokens = torch.multinomial(Q, n, replacement=True).view(n, 1)
            q, support, law = Q.expand(n, 1, 4).contiguous(), torch.arange(4).expand(n, 1, 4).contiguous(), soft_law(s)
        else:
            y = 0 if draft == "mtp-top" else 2
            tokens, q, support, law = torch.full((n, 1), y), None, None, soft_law(s, y)
        gens = [torch.Generator().manual_seed(10 * n + i) for i in range(n)]
        accepted, correction, flags, _ = verify_batch(tokens, logits, [params(s)] * n, gens, q, support)
        got = torch.bincount(first_tokens(tokens, accepted, correction), minlength=4).float() / n
        assert (got - law).abs().max() < 0.012, (draft, s, got, law)
        assert (got <= math.exp(s) * P + 0.012).all()


@pytest.mark.parametrize("draft", ["mtp", "dflash2"])
def test_verify_rows_law(draft):
    n = 12_000
    torch.manual_seed(2)
    p = P.expand(2, 4).contiguous()
    for s in (0.0, 0.25, 1.0):
        counts = torch.zeros(4)
        for i in range(n):
            g = torch.Generator().manual_seed(i)
            if draft == "dflash2":
                tokens, q, support = torch.multinomial(Q, 1, generator=g), Q.view(1, 4), torch.arange(4).view(1, 4)
            else:
                tokens, q, support = torch.tensor([1]), None, None
            a, c, _ = verify_rows(tokens, p, q, support, g, soft_scale=params(s).soft_scale)
            counts[int(tokens[0]) if int(a) > 0 else int(c)] += 1
        law = soft_law(s) if draft == "dflash2" else soft_law(s, 1)
        assert (counts / n - law).abs().max() < 0.02, (draft, s, counts / n, law)


def test_soft_zero_is_exact():
    assert torch.allclose(soft_law(0.0), P) and torch.allclose(soft_law(0.0, 1), P)
    assert params(0.0).soft_scale == 1.0 and SamplingParams().accept_soft == 0.0
    assert not params(0.0).relaxed and params(0.5).relaxed


def test_exact_rows_are_unchanged_beside_relaxed_ones():
    n = 2_000
    tokens, logits, q, support = dflash2_rows(n, 4)
    alone = verify_batch(tokens, logits, [params(0.0)] * n, generators(n), q, support)
    mixed = verify_batch(tokens, logits, [params(0.0 if i % 2 else 0.5) for i in range(n)], generators(n), q, support)
    for a, b in zip(alone[:3], mixed[:3]):
        assert torch.equal(a[1::2], b[1::2])


def test_a_draft_its_law_never_gives_is_rejected():
    """A drafted token with q(y) = 0 (a rounding edge of the draw) is rejected whatever s, as by the exact test."""
    n, s = 2000, 5.0
    q = torch.tensor([0.5, 0.5, 0.0, 0.0]).expand(n, 1, 4).contiguous()
    tokens, support = torch.full((n, 1), 2), torch.arange(4).expand(n, 1, 4).contiguous()
    gens = [torch.Generator().manual_seed(i) for i in range(n)]
    accepted, _, flags, _ = verify_batch(tokens, P.log().expand(n, 2, 4).contiguous(), [params(s)] * n, gens, q, support)
    assert int(accepted.sum()) == 0 and not flags.any()
    p = P.unsqueeze(0).expand(2, 4).contiguous()
    for i in range(200):
        a, _, f = verify_rows(torch.tensor([2]), p, q[0], support[0], torch.Generator().manual_seed(i), math.exp(-s))
        assert int(a) == 0 and not f.any()


def test_flags_mark_only_the_boosted_accepts():
    n = 20_000
    tokens, logits, q, support = dflash2_rows(n, 3)
    exact = verify_batch(tokens, logits, [params(0.0)] * n, generators(n), q, support)
    soft = verify_batch(tokens, logits, [params(0.25)] * n, generators(n), q, support)
    assert not exact[2].any()  # s = 0 boosts nothing
    # the same uniforms: an accept at s = 0 stays an accept, and the flags are the accepts s = 0.25 adds
    assert torch.equal(soft[2][:, 0], (soft[0] > 0) & (exact[0] == 0))
    assert soft[2].any()


GATES = ((2, 1.0), (3, 2.5))  # G = {0, 1}: token 2 is 2.15 nats below the top; G = {0, 1, 2}


@pytest.mark.parametrize("draft", ["mtp-top", "mtp-mid", "mtp-tail", "dflash2"])
@pytest.mark.parametrize("gate", GATES)
def test_verify_batch_gated_law(draft, gate):
    n, s = 40_000, 0.5
    logits = P.log().expand(n, 2, 4).contiguous()
    if draft == "dflash2":
        tokens = torch.multinomial(Q, n, replacement=True, generator=torch.Generator().manual_seed(7)).view(n, 1)
        q, support, law = Q.expand(n, 1, 4).contiguous(), torch.arange(4).expand(n, 1, 4).contiguous(), soft_law(s, gate=gate)
    else:
        y = {"mtp-top": 0, "mtp-mid": 1, "mtp-tail": 2}[draft]
        tokens, q, support, law = torch.full((n, 1), y), None, None, soft_law(s, y, gate)
    gens = [torch.Generator().manual_seed(20 * n + i) for i in range(n)]
    accepted, correction, flags, _ = verify_batch(tokens, logits, [params(s, accept_top_k=gate[0], accept_margin=gate[1])] * n, gens, q, support)
    got = torch.bincount(first_tokens(tokens, accepted, correction), minlength=4).float() / n
    assert (got - law).abs().max() < 0.012, (draft, gate, got, law)
    outside = ~gate_set(*gate)
    assert (got[outside] <= P[outside] + 0.012).all()  # outside the gate: never more often than the target
    if draft != "dflash2" and not bool(gate_set(*gate)[tokens[0, 0]]):
        assert not flags.any()  # a draft outside the gate takes the exact test only


@pytest.mark.parametrize("draft", ["mtp", "dflash2"])
def test_verify_rows_gated_law(draft):
    n, s, gate = 12_000, 0.5, (2, 1.0)
    p = P.expand(2, 4).contiguous()
    counts = torch.zeros(4)
    sp = params(s, accept_top_k=gate[0], accept_margin=gate[1])
    assert sp.soft_gate == gate
    for i in range(n):
        g = torch.Generator().manual_seed(i)
        if draft == "dflash2":
            tokens, q, support = torch.multinomial(Q, 1, generator=g), Q.view(1, 4), torch.arange(4).view(1, 4)
        else:
            tokens, q, support = torch.tensor([1]), None, None
        a, c, _ = verify_rows(tokens, p, q, support, g, soft_scale=sp.soft_scale, gate=sp.soft_gate)
        counts[int(tokens[0]) if int(a) > 0 else int(c)] += 1
    law = soft_law(s, gate=gate) if draft == "dflash2" else soft_law(s, 1, gate)
    assert (counts / n - law).abs().max() < 0.02, (draft, counts / n, law)


def test_a_gate_holding_every_token_is_the_ungated_rule():
    n = 4_000
    tokens, logits, q, support = dflash2_rows(n, 5)
    for s in (0.0, 0.5):
        plain = verify_batch(tokens, logits, [params(s)] * n, generators(n), q, support)
        wide = verify_batch(tokens, logits, [params(s, accept_top_k=4, accept_margin=10.0)] * n, generators(n), q, support)
        assert all(torch.equal(a, b) for a, b in zip(plain[:3], wide[:3]))
    tail = torch.full((n, 1), 2)  # MTP's point mass on a token outside the gate 2:1.0
    gated = verify_batch(tail, logits, [params(0.5, accept_top_k=2, accept_margin=1.0)] * n, generators(n))
    plain = verify_batch(tail, logits, [params(0.5)] * n, generators(n))
    assert not gated[2].any() and plain[2].any()  # outside the gate: no boosted accept
    assert torch.equal(gated[0] > 0, (plain[0] > 0) & ~plain[2][:, 0])  # the same uniforms: exactly the boosts go


def test_soft_gate_needs_a_sampled_relaxed_request():
    assert params(0.5, accept_top_k=3, accept_margin=2.0).soft_gate == (3, 2.0)
    assert params(0.5).soft_gate is None  # the defaults (1, 0.0): no gate
    assert params(0.5, accept_top_k=3).soft_gate == (3, 0.0) and params(0.5, accept_margin=1.0).soft_gate == (1, 1.0)
    assert params(0.0, accept_top_k=3, accept_margin=2.0).soft_gate is None
    assert not params(0.0, accept_top_k=3, accept_margin=2.0).relaxed
    assert SamplingParams(temperature=0.0, accept_soft=0.5, accept_top_k=3, accept_margin=2.0).soft_gate is None


def test_greedy_rows_ignore_accept_soft():
    tokens = torch.tensor([[1]])
    logits = P.log().view(1, 1, 4).expand(1, 2, 4).contiguous()
    greedy = SamplingParams(temperature=0.0, accept_soft=0.5)
    assert not greedy.relaxed
    a, c, flags, _ = verify_batch(tokens, logits, [greedy], [torch.Generator().manual_seed(0)])
    assert int(a[0]) == 0 and int(c[0]) == 0 and not flags.any()


def test_validation_and_server_default():
    SamplingParams(accept_soft=0.0).validate()
    SamplingParams(accept_soft=2).validate()
    for bad in (math.inf, -0.1, float("nan"), "1", True):
        with pytest.raises(InvalidRequest, match="finite number >= 0"):
            SamplingParams(accept_soft=bad).validate()
    with pytest.raises(InvalidRequest):
        _sampling_from({"temperature": 1.0, "accept_soft": math.inf})
    with pytest.raises(InvalidRequest):
        validation.set_accept_default(1, 0.0, soft=math.inf)
    assert _sampling_from({"temperature": 1.0}).accept_soft == 0.0
    assert _sampling_from({"temperature": 1.0, "accept_soft": 0.25}).accept_soft == 0.25
    assert _sampling_from({"sampling_params": {"temperature": 1.0, "accept_soft": 0.5}}).accept_soft == 0.5
    try:
        validation.set_accept_default(1, 0.0, soft=0.3)
        assert _sampling_from({"temperature": 1.0}).relaxed and _sampling_from({"temperature": 1.0}).accept_soft == 0.3
        assert not _sampling_from({"temperature": 1.0, "accept_soft": 0}).relaxed
    finally:
        validation.set_accept_default(1, 0.0)
    assert validation.ACCEPT_SOFT_DEFAULT == 0.0


def test_server_soft_gate_default():
    try:
        validation.set_accept_default(3, 2.0, soft=0.5, soft_gate=(2, 1.0))
        assert _sampling_from({"temperature": 1.0}).soft_gate == (2, 1.0)
        assert _sampling_from({"temperature": 1.0, "accept_top_k": 4, "accept_margin": 3.0}).soft_gate == (4, 3.0)
        assert _sampling_from({"temperature": 1.0, "accept_top_k": 1}).soft_gate == (2, 1.0)  # not both: the server's
        assert _sampling_from({"temperature": 1.0, "accept_top_k": 1, "accept_margin": 0}).soft_gate is None  # both: none
        assert _sampling_from({"temperature": 1.0, "accept_top_k": 1, "accept_margin": 0.5}).soft_gate == (1, 0.5)
        greedy = _sampling_from({"temperature": 0.0})
        assert (greedy.accept_top_k, greedy.accept_margin) == (3, 2.0) and greedy.soft_gate is None
        for bad in ((2, -1.0), (0, 1.0), (2, math.inf)):
            with pytest.raises(InvalidRequest, match="accept_soft gate"):
                validation.set_accept_default(1, 0.0, soft=0.5, soft_gate=bad)
        assert validation.parse_soft_gate("off") == validation.parse_soft_gate("1:0") == (1, 0.0)
        assert validation.parse_soft_gate("3:2.0") == (3, 2.0) and validation.parse_soft_gate("1:0.5") == (1, 0.5)
        for bad in ("3", "0:1", "3:-1", "3:inf", "a:b"):
            with pytest.raises(ValueError, match="accept_soft gate"):
                validation.parse_soft_gate(bad)
    finally:
        validation.set_accept_default(1, 0.0)
    assert validation.ACCEPT_SOFT_GATE == (1, 0.0) and _sampling_from({"temperature": 1.0, "accept_soft": 0.5}).soft_gate is None
    # no server gate: one field alone gates nothing (the rule stays ungated, as before sampled requests read the fields)
    for one in ({"accept_top_k": 3}, {"accept_margin": 1.0}, {"sampling_params": {"accept_top_k": 3}}):
        assert _sampling_from({"temperature": 1.0, "accept_soft": 0.5, **one}).soft_gate is None, one
    assert _sampling_from({"temperature": 1.0, "accept_soft": 0.5, "accept_top_k": 3, "accept_margin": 2.0}).soft_gate == (3, 2.0)
    assert _sampling_from({"temperature": 0.0, "accept_top_k": 3}).accept_top_k == 3  # greedy requests keep their fields
