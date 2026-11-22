"""The `original` QSA selection's two implementations, the Triton scorer (`kernels/qsa/selection.py`) and the eager
SM121 kernels (`kernels/qsa/score_exact.py`), against its math in float64: scores within FP32 rounding, the same pages
up to near-ties at the top-k cutoff, through long contexts and several sequences; the top-k's fallbacks against
Torch's; and the attention output through the indexer."""
import pytest
import torch

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (12, 1), reason="SM121 kernels")
SCALE = 128 ** -0.5


def gamma(dim, heads=4):
    """Any order of a score's FP32 additions (dim - 1 in a head's dot product, the heads, then the scale) rounds within
    gamma_n = n u / (1 - n u) of the sum of the magnitudes (BF16 products are exact in FP32); relu does not widen it."""
    n = dim + heads + 1
    return n * 2.0 ** -24 / (1 - n * 2.0 ** -24)


def _inputs(contexts, rows, seed=0, key_scale=1.0, dim=128):
    """Rows ending each sequence of `contexts`, on scrambled pages of one pool."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(contexts)
    aux = (torch.randn(total + 8, 2, dim, generator=g, device="cuda") * key_scale).to(torch.bfloat16)
    table = (torch.randperm(total + 8, generator=g, device="cuda")[:total]).to(torch.int32)
    offsets = torch.tensor([0] + list(contexts), device="cuda").cumsum(0)[:-1]
    positions = torch.cat([torch.arange(c - r, c, device="cuda") for c, r in zip(contexts, rows)])
    base = torch.repeat_interleave(offsets, torch.tensor(rows, device="cuda")).to(torch.int32)
    q = torch.randn(sum(rows), 4, dim, generator=g, device="cuda").to(torch.bfloat16)
    return q, aux, table, base, positions


def _reference(q, aux, table, base, positions, gcap, ratio=4, scale=SCALE):
    """The scores in float64 (scale x the heads' summed relu(q . k), each group's key the compressed key on its last
    page, -inf for groups a row cannot see yet) and each score's FP32 rounding bound."""
    want = torch.full((q.shape[0], gcap), float("-inf"), dtype=torch.float64, device="cuda")
    bound = torch.zeros_like(want)
    visible = (positions.long() + 1) // ratio
    for b in base.unique().tolist():
        rows = (base == b).nonzero().flatten()
        n = int(visible[rows].max())
        keys = aux[table[b + ratio - 1: b + n * ratio: ratio].long(), 1].double()
        qs = q[rows].double()
        scores = torch.einsum("rhd,gd->rhg", qs, keys).clamp_min(0).sum(1) * scale
        seen = torch.arange(n, device="cuda")[None, :] < visible[rows, None]
        want[rows, :n] = torch.where(seen, scores, float("-inf"))
        bound[rows, :n] = torch.einsum("rhd,gd->rg", qs.abs(), keys.abs()) * scale * gamma(q.shape[-1], q.shape[1])
    return want, bound


CASES = [([32768 + 4099], [300]), ([65536, 40001], [200, 129]), ([131072], [256]), ([3000, 9001, 5], [64, 70, 5])]


@pytest.mark.parametrize("contexts,rows", CASES)
def test_scores_are_the_float64_scores_to_fp32_rounding(contexts, rows):
    from hibana.kernels.qsa.score_exact import _tiles, load_exact_scorer
    from hibana.kernels.qsa.selection import score_groups
    for key_scale in (1.0, 30.0):
        q, aux, table, base, positions = _inputs(contexts, rows, key_scale=key_scale)
        gcap = max(contexts) // 4
        want, bound = _reference(q, aux, table, base, positions, gcap)
        got = {"triton": score_groups(q, aux, table, base, positions, 4, gcap, SCALE)}
        tiles = torch.tensor(_tiles(rows, 0, sum(rows)), dtype=torch.int32, device="cuda").view(-1, 2)
        for per_block in (1, 8):
            got[f"cuda, {per_block} tiles a block"] = out = torch.full_like(got["triton"], float("nan"))
            load_exact_scorer().score(q, aux, table, base, positions, tiles, out, SCALE, per_block)
        seen = torch.isfinite(want)
        for name, scores in got.items():
            assert torch.equal(torch.isfinite(scores), seen) and (scores[~seen] == float("-inf")).all(), name
            assert ((scores.double() - want).abs() <= bound)[seen].all(), name


@pytest.mark.parametrize("dim", [64, 32])
def test_triton_scorer_at_other_widths(dim):
    """Other key widths than score_exact.cu's 128 take the Triton scorer with tl.sum: the same float64 check."""
    from hibana.kernels.qsa.selection import score_groups
    contexts, rows = [3000, 9001], [64, 70]
    q, aux, table, base, positions = _inputs(contexts, rows, dim=dim)
    gcap, scale = max(contexts) // 4, dim ** -0.5
    want, bound = _reference(q, aux, table, base, positions, gcap, scale=scale)
    got = score_groups(q, aux, table, base, positions, 4, gcap, scale)
    seen = torch.isfinite(want)
    assert torch.equal(torch.isfinite(got), seen) and (got[~seen] == float("-inf")).all()
    assert ((got.double() - want).abs() <= bound)[seen].all()


def _groups(pages, count, inverse, base, position, ratio=4):
    """The groups a row's pages select (its tail, the incomplete group's tokens, left out), after checking that each
    page appears once, each selected group brings all `ratio` of its pages, and the tail is the incomplete group's
    tokens up to the row, in order, last."""
    tokens = (inverse[pages[:count].long()] - base).tolist()
    assert len(set(tokens)) == count, "a page selected twice"
    visible = (position + 1) // ratio
    tail = list(range(visible * ratio, position + 1))
    assert tokens[len(tokens) - len(tail):] == tail
    chosen = [t // ratio for t in tokens[:len(tokens) - len(tail)]]
    assert all(chosen.count(g) == ratio for g in set(chosen)), "a group without all its pages"
    return set(chosen)


@pytest.mark.parametrize("contexts,rows", CASES)
def test_selection_is_the_float64_selection_up_to_near_ties(contexts, rows):
    from hibana.kernels.qsa.score_exact import select_exact
    from hibana.kernels.qsa.selection import select_pages
    q, aux, table, base, positions = _inputs(contexts, rows, seed=1)
    gcap, top = max(contexts) // 4, 2048 // 4
    want, bound = _reference(q, aux, table, base, positions, gcap)
    inverse = torch.full((int(table.max()) + 1,), -1, dtype=torch.long, device="cuda")
    inverse[table.long()] = torch.arange(table.numel(), device="cuda")
    selections = [select_pages(q, aux, table, base, positions, 4, 2048, gcap, SCALE, chunk=256),
                  *(select_exact(q, aux, table, base, positions, gcap, SCALE, rows, chunk) for chunk in (256, 100))]
    for r in range(q.shape[0]):
        position, visible = int(positions[r]), (int(positions[r]) + 1) // 4
        kept = min(visible, top)
        values, order = want[r].topk(kept)
        reference, near = set(order.tolist()), set()
        if kept:  # the groups within rounding of the cutoff may land on either side of it
            cut, cut_bound = values[-1], bound[r, order[-1]]
            near = set(((want[r] - cut).abs() <= bound[r] + cut_bound).nonzero().flatten().tolist())
        tail = position + 1 - visible * 4
        for pages, counts in selections:
            assert int(counts[r]) == kept * 4 + tail and (pages[r, counts[r]:] == -1).all()
            chosen = _groups(pages[r], int(counts[r]), inverse, int(base[r]), position)
            assert len(chosen) == kept and chosen ^ reference <= near, f"row {r}"


def _scores(kind, rows, cols, g):
    if kind == "random":
        return torch.randn(rows, cols, generator=g, device="cuda").relu()
    if kind == "ties":  # few distinct values: the 512th is tied, so only the first ties by group fit
        return torch.randint(0, 40, (rows, cols), generator=g, device="cuda").float() / 8
    if kind == "zeros":  # fewer than 512 positive scores: ties at 0
        s = torch.zeros(rows, cols, device="cuda")
        s[:, torch.randperm(cols, generator=g, device="cuda")[:300]] = 1.0 + torch.rand(rows, 300, generator=g, device="cuda")
        return s
    if kind == "crowded":  # every key in one 11-bit bin: resolved over the row instead of shared memory
        return 1.0 + torch.rand(rows, cols, generator=g, device="cuda") * 0.2
    if kind == "crowded_ties":
        return 1.0 + torch.randint(0, 3, (rows, cols), generator=g, device="cuda").float() / 64
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["random", "ties", "zeros", "crowded", "crowded_ties"])
@pytest.mark.parametrize("cols", [700, 8192, 32768])
def test_topk_matches_torch(kind, cols, monkeypatch):
    """The page assembly from the long-row top-512 against Torch's top-k and `_assemble_kernel`, on given scores (some
    rows partly invisible: -inf)."""
    from hibana.kernels.qsa import selection
    from hibana.kernels.qsa.score_exact import load_exact_scorer
    g = torch.Generator(device="cuda").manual_seed(cols)
    rows = 64
    scores = _scores(kind, rows, cols, g)
    visible = torch.randint(cols // 2, cols + 1, (rows,), generator=g, device="cuda")
    visible[:4] = torch.tensor([cols, 1, 511, 513], device="cuda")
    scores[torch.arange(cols, device="cuda")[None, :] >= visible[:, None]] = float("-inf")
    positions = visible * 4 + torch.randint(0, 4, (rows,), generator=g, device="cuda") - 1
    positions = positions.clamp_min(0)
    table = torch.randperm(cols * 4 + 8, generator=g, device="cuda").to(torch.int32)
    base = torch.zeros(rows, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(selection, "score_groups", lambda *args: scores)
    q = torch.empty(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
    want = selection.select_pages(q, None, table, base, positions, 4, 2048, cols, SCALE)
    pages = torch.empty(rows, 2051, dtype=torch.int32, device="cuda")
    counts = torch.empty(rows, dtype=torch.int32, device="cuda")
    load_exact_scorer().topk_pages_long(scores, table, base.long(), positions.long(), pages, counts)
    assert torch.equal(counts, want[1]) and torch.equal(pages, want[0])


@pytest.mark.parametrize("prefix,length", [(40960, 1024), (2048, 2048)])
@torch.inference_mode()
def test_indexer_attention_is_the_same(prefix, length, monkeypatch):
    """`FlashNextIndexer.run` (the `original` backend) with and without the exact kernels: the same attention output
    to BF16 rounding."""
    from hibana.kernels.qsa import score_exact
    from tests.qsa_fixture import QsaFixture
    f = QsaFixture(1, length, prefix, hidden=256)
    aux = f.aux.clone()
    calls = []
    select_exact = score_exact.select_exact
    monkeypatch.setattr(score_exact, "select_exact", lambda *a: calls.append(1) or select_exact(*a))
    got = f.native().clone()
    assert calls
    f.aux.copy_(aux)
    monkeypatch.setattr(score_exact, "supported", lambda *a: False)
    want = f.native()
    assert len(calls) == 1
    torch.testing.assert_close(got, want)  # the same pages up to near-ties, so the same output to BF16 rounding
