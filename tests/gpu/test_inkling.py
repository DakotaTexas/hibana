"""Inkling's decode and prefill kernels against the layer in PyTorch operators, on random weights under the
checkpoint's names, and the engine's decode against a prefill of the same tokens on the pinned checkpoint."""

import gc

import pytest
import torch

from hibana.cache.kv_pool import KvPool
from hibana.model.inkling import attention, config as c
from hibana.model.inkling.attention import Layout, Sequence, rms
from hibana.model.inkling.model import Layer, attn_res, decode_visit, prefill_visit
from hibana.model.inkling.moe import Moe
from hibana.target import INKLING

DEVICE = torch.device("cuda")
H, KV = c.HIDDEN_SIZE, c.KV_SIZE


class Weights(dict):
    def get(self, name):
        return self[name]


def experts(gen, rows: int, columns: int, fp4: bool) -> dict:
    """A layer's packed expert matrix under the checkpoint's names' suffixes: NVFP4 or MXFP8 as the checkpoint stores them."""
    e = c.NUM_EXPERTS
    if fp4:
        scale = (torch.rand((e, rows, columns // 16), generator=gen) * 0.006 + 0.002).to(torch.float8_e4m3fn)
        return {"": torch.randint(0, 256, (e, rows, columns // 2), generator=gen, dtype=torch.uint8), "_scale": scale,
                "_amax": torch.rand(e, generator=gen) + 0.5, "_outer_scale_denominator": torch.full((e,), 2.0)}
    return {"": (torch.randn((e, rows, columns), generator=gen) * 4).to(torch.float8_e4m3fn),
            "_scale": torch.randint(118, 121, (e, rows, columns // 32), generator=gen, dtype=torch.uint8)}


def layer_weights(index: int, fp4: bool = True) -> Weights:
    gen = torch.Generator().manual_seed(index)
    bf16 = lambda *shape, scale=0.02: (torch.randn(shape, generator=gen) * scale).to(torch.bfloat16)  # noqa: E731
    gain = lambda *shape: (1 + 0.1 * torch.randn(shape, generator=gen)).to(torch.bfloat16)  # noqa: E731
    p = f"{c.PREFIX}.layers.{index}"
    heads, local = c.NUM_ATTENTION_HEADS, index in c.LOCAL_LAYERS
    # Biases two apart rank the experts by themselves: scores (in 0..1) a rounding apart must not pick different
    # experts in the two paths under comparison. The weights still come from each row's scores.
    bias = torch.randperm(c.NUM_EXPERTS, generator=gen).float() * 2
    w = Weights({f"{p}.mlp_norm.weight": gain(H), f"{p}.attn_sconv.weight": bf16(H, 1, 4, scale=0.3), f"{p}.mlp_sconv.weight": bf16(H, 1, 4, scale=0.3),
                 f"{p}.attn.wq_du.weight": bf16(heads * 128, H, scale=0.2), f"{p}.attn.wr_du.weight": bf16(heads * c.D_REL, H, scale=0.2),
                 f"{p}.attn.gate_attn.weight": bf16(heads * 128, H), f"{p}.attn.attn_sink_proj.weight": bf16(heads, H),
                 f"{p}.attn.wk_dv.weight": bf16(KV, H, scale=0.2), f"{p}.attn.wv_dv.weight": bf16(KV, H), f"{p}.attn.wo_ud.weight": bf16(H, heads * 128),
                 f"{p}.attn.q_l2_clip.scale": gain(heads, 128), f"{p}.attn.rel_l2_clip.scale": gain(heads, c.D_REL),
                 f"{p}.attn.rel_logits_proj.proj": bf16(c.D_REL, c.REL_EXTENT, scale=0.3),
                 f"{p}.mlp.gate.weight": bf16(c.NUM_EXPERTS, H), f"{p}.mlp.gate.bias": bias,
                 f"{c.PREFIX}.attn_res.attn_queries.{index - 1}": torch.randn((1 if local else 3, H), generator=gen),
                 f"{c.PREFIX}.attn_res.mlp_queries.{index}": torch.randn(H, generator=gen)})
    for name in ("attn_norm",) if local else ("q_input_norm", "k_input_norm", "v_input_norm"):
        w[f"{p}.{name}.weight"] = gain(H)
    for stream in "kv":
        if local:
            w[f"{p}.attn.{stream}_sconv.weight"] = bf16(KV, 1, 4, scale=0.3)
        else:
            w[f"{p}.attn.{stream}_sconv.static_w"] = bf16(4, KV, scale=0.3)
            w[f"{p}.attn.{stream}_sconv.weight_proj"] = bf16(c.NUM_KV_HEADS * 4, H)
    for name in ("attn", "k", "v", "mlp"):
        w[f"{p}.controllers.{name}.down_weight"] = bf16(c.GATE_RANK, 2 * H)
        w[f"{p}.controllers.{name}.forget_weight"] = bf16(H, c.GATE_RANK, scale=0.1)
        w[f"{p}.controllers.{name}.gain_weight"] = bf16(H, c.GATE_RANK, scale=0.1)
    for name, (rows, columns) in (("w13", (2 * c.MOE_INTERMEDIATE_SIZE, H)), ("w2", (H, c.MOE_INTERMEDIATE_SIZE))):
        w.update({f"{p}.mlp.experts.{name}_weight{suffix}": t for suffix, t in experts(gen, rows, columns, fp4).items()})
    return w


def checkpoint():
    """The pinned checkpoint's snapshot; the test skips without it, as the suite's other checkpoint modules do."""
    try:
        return INKLING.snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def load():
    """The pinned checkpoint's model, loaded once the memory an earlier module cached is free, as the suite's other
    checkpoint modules load theirs."""
    from hibana.target import load_model
    snap = checkpoint()
    gc.collect()
    torch.cuda.empty_cache()
    return load_model(INKLING, snap, log=None)


def cosine(a, b) -> float:
    return torch.nn.functional.cosine_similarity(a.double().flatten(1), b.double().flatten(1), dim=-1).min().item()


@pytest.mark.parametrize("index, fp4, message, kv_dtype, steps", [
    (7, True, False, torch.float8_e4m3fn, 1),   # sliding layer, first pass
    (7, True, True, torch.bfloat16, 1),         # sliding layer with messages
    (11, True, True, torch.float8_e4m3fn, 1),   # full attention with messages
    (12, False, False, torch.float8_e4m3fn, 1),  # a layer that opens a block, MXFP8 experts
    (7, True, True, torch.float8_e4m3fn, 4),    # verification: four rows a sequence, sliding layer
    (11, True, False, torch.bfloat16, 3),       # verification, full attention (per-token convolution taps)
    (11, True, True, torch.float8_e4m3fn, 8),   # DFlash2's depth 7: eight rows a sequence on FP8 full attention, 24 gated rows
    (7, True, False, torch.float8_e4m3fn, 2),   # two rows a sequence, sliding layer
])
def test_kernel_visit_matches_the_reference_layer(index, fp4, message, kv_dtype, steps):
    """One layer visit of three sequences with different histories through the decode kernels (`decode_visit`: mix,
    gate, prep, attention, conv, route, experts), through prefill's (`prefill_visit`: mix, gate, convolution and
    attention kernels) and through `Layer.forward` in PyTorch operators: the same hidden, K/V rows and windows.
    With `steps` rows a sequence the decode kernels verify: a row's convolution reads its sequence's earlier rows, the
    stored windows stay and the window after each row is recorded. A wrong tap order, window, bias index, gate or
    state slot would still produce plausible numbers."""
    gen = torch.Generator().manual_seed(100 + index)
    layer = Layer(layer_weights(index, fp4), index, DEVICE, "bf16")
    visit, lens = 20, [steps, 700, 1500]  # keys each sequence's last row sees, its own included: past the 1,024-key window for the last
    seqs, rows, pages_total = len(lens), len(lens) * steps, 2400
    pool = KvPool(pages_total, DEVICE, target=INKLING, layer_ids=[visit], dtype=kv_dtype)
    for cache, scale in zip(pool.cache(visit), (1.5, 1.0)):
        cache.copy_((torch.randn(cache.shape, generator=gen) * scale).to(kv_dtype))
    pages = [torch.randperm(pages_total - 1, generator=gen)[:n].add(1).to(torch.int32).to(DEVICE) for n in lens]
    positions = torch.cat([torch.arange(n - steps, n) for n in lens]).to(DEVICE)
    slots = torch.tensor([3, 1, 2], dtype=torch.int32, device=DEVICE)
    state = (torch.randn((4, c.NUM_VISITS, c.CONV_KERNEL - 1, c.CONV_WIDTH), generator=gen) * 0.5).to(torch.bfloat16).to(DEVICE)
    hidden = torch.randn((rows, H), generator=gen).to(torch.bfloat16).to(DEVICE)
    count = 2
    states = torch.zeros((rows, 5, H), device=DEVICE)
    states[:, :count] = rms(torch.randn((rows, count, H), generator=gen).to(DEVICE))
    sources = queries = messages = None
    if message:
        previous = torch.randn((rows, H), generator=gen).to(DEVICE)
        sources = torch.stack([states[:, 1], rms(previous)], dim=1).contiguous()
        queries = torch.randn((4, H), generator=gen).to(DEVICE)
        messages = attn_res([states[:, 1]], previous, queries)

    ref_state = state.clone()
    blocks = [states[:, n] for n in range(count)]
    layout = Layout([Sequence(i * steps, steps, n - steps, pages[i], int(slots[i])) for i, n in enumerate(lens)], ref_state, [])
    attention.FUSED = False
    try:
        expected = layer.forward(hidden, blocks, messages, positions, layout, visit, pool)
    finally:
        attention.FUSED = True
    own = torch.cat([p[-steps:] for p in pages]).long()
    expected_kv = [cache[own].float() for cache in pool.cache(visit)]
    for cache in pool.cache(visit):  # the kernels must write the rows' own pages themselves
        cache[own] = 0

    filled, fill_state = states.clone(), state.clone()
    got = prefill_visit(layer, hidden, filled, count, messages.permute(1, 0, 2).contiguous() if message else None, positions,
                        Layout(layout.seqs, fill_state, []), visit, pool)
    assert cosine(got, expected) > 0.999  # the experts quantize their inputs: a rounding apart upstream is a code apart there
    for cache, want in zip(pool.cache(visit), expected_kv):
        assert cosine(cache[own].float(), want) > 0.9995
        cache[own] = 0
    assert torch.equal(fill_state[0], state[0]) and cosine(fill_state[slots.long(), visit].float(), ref_state[slots.long(), visit].float()) > 0.9999
    if layer.boundary:
        assert cosine(filled[:, count], rms(hidden)) > 0.999999

    indptr = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE)
    windows = None if steps == 1 else torch.zeros((seqs, steps + 1, *state.shape[1:]), dtype=state.dtype, device=DEVICE)
    before = state.clone()
    layout = Layout([], state, [], (indptr, torch.cat(pages)), own.to(torch.int32), slots,
                    torch.arange(rows, dtype=torch.int32, device=DEVICE) // steps, (positions + 1).to(torch.int32), steps, windows)
    got = decode_visit(layer, hidden, states, count, (sources, queries) if message else None, positions, layout, visit, pool)

    assert cosine(got, expected) > 0.999  # the experts quantize their inputs: a rounding apart upstream is a code apart there
    for cache, want in zip(pool.cache(visit), expected_kv):
        assert cosine(cache[own].float(), want) > 0.9995  # FP8 rows may round a last bit apart
    used = slots.long()
    if steps == 1:  # decode advances the stored windows
        after = state[used, visit]
        assert torch.equal(state[0], before[0]) and torch.equal(state[used, visit + 1], before[used, visit + 1])
    else:  # verification leaves them and records the window after each row: the last is the reference's
        after = windows[:, steps - 1, visit]
        assert torch.equal(state, before)
        assert torch.equal(windows[:, 0, visit, :2], before[used, visit, 1:])  # after the first row: the stored window shifted by one
    assert cosine(after.float(), ref_state[used, visit].float()) > 0.9999
    if layer.boundary:  # the block the layer opened: rms of its input
        assert cosine(states[:, count], rms(hidden)) > 0.999999


@pytest.mark.parametrize("rows", [5, 40])
@pytest.mark.parametrize("fp4", [True, False])
def test_decode_and_prefill_experts_agree(fp4, rows):
    """Decode's experts (the route kernel, a product a (row, expert) pair, or at 40 rows the pairs sorted by expert)
    against prefill's (the wide route kernel, the pairs sorted by expert) on the same packed NVFP4 / MXFP8 tensors."""
    w = layer_weights(3, fp4)
    moe = Moe(w, f"{c.PREFIX}.layers.3.mlp", DEVICE)
    x = torch.randn((rows, H), generator=torch.Generator().manual_seed(3)).to(torch.bfloat16).to(DEVICE)
    packed = moe(x)
    Moe.pair_calls = False
    try:
        dense = moe(x)
    finally:
        Moe.pair_calls = True
    assert cosine(packed, dense) > 0.9999 and dense.float().abs().mean() > 0.01


@pytest.mark.parametrize("fp4", [True, False])
def test_shared_experts_agree_on_both_paths(fp4):
    """Expert sharing at 24 rows (6 requests x 4), the rows alike and a budget under a row's 8 experts, so each row
    fills its unused slots with its top expert at weight 0: that expert holds 5 x 24 pairs. Decode's pair path, which
    leaves the filler pairs unwritten, against the sorted prefill path on the same budgeted routing."""
    w = layer_weights(3, fp4)
    moe = Moe(w, f"{c.PREFIX}.layers.3.mlp", DEVICE)
    moe.keep, moe.per_request, moe.budget = 2, 0, 4
    gen = torch.Generator().manual_seed(5)
    x = (torch.randn((1, H), generator=gen) + 0.05 * torch.randn((24, H), generator=gen)).to(torch.bfloat16).to(DEVICE)
    slots = torch.arange(1, 7, dtype=torch.int32, device=DEVICE)
    ids, weights = moe.share(x, slots, 4)
    assert int(torch.bincount(ids.flatten()).max()) > 24 and int((weights == 0).sum()) > 0
    packed = moe(x, slots, 4)
    Moe.pair_calls = False
    try:
        dense = moe(x, slots, 4)
    finally:
        Moe.pair_calls = True
    assert torch.isfinite(packed).all() and cosine(packed, dense) > 0.9999


def test_decode_continues_what_prefill_computes():
    """On the pinned checkpoint: tokens decoded through the engine (chunked prefill, conv windows and K/V carried in
    the pools, decode graphs) against one stand-alone prefill of prompt plus output. The pool keeps BF16 K/V as the
    stand-alone pass does. The two paths round differently and the experts' 4-bit inputs amplify that, most on the
    last rows (their minimum cosine moves from 0.973 to 0.997 with the prefill chunk alone), so the mean holds the
    paths to each other and the minimum only bounds the drift. It is blind to the hidden streams' convolution windows
    (a wrong state slot there scored 0.997): the visit test holds those exactly."""
    from hibana.benchmark_workloads import CODE_QUESTION, frozen_source
    from hibana.engine import config as engine_config
    from hibana.engine.executor import Executor
    from hibana.engine.sampler import SamplingParams
    before = engine_config.current()
    engine_config.install(before.replace(kv_format="sim-bf16"))
    try:
        model = load()
        ex = Executor(model, num_pages=8192, num_slots=8, max_batch=2, log=None)
        ex.capture_graphs(log=None)
        ids = chat_ids(f"Here is inference-engine source:\n{frozen_source()[:6000]}\n{CODE_QUESTION}")  # past the 1,024-key window
        kept, sample = [], ex.sampler.sample
        ex.sampler.sample = lambda logits, *a, **k: (kept.append(logits[:1].double().cpu()), sample(logits, *a, **k))[1]
        req = ex.admit(ids, SamplingParams(max_new_tokens=12, temperature=0.0, ignore_eos=True))
        while req.prefill_pos < req.prefill_len:
            ex.prefill(req, max_tokens=512)
        while not req.finished:
            ex.decode_step([req])
        out = list(req.output_ids)
        ex.release(req)
        full = model.logits(model.prefill(torch.tensor(ids + out, device=DEVICE))["norm"]).double().cpu()
        decoded = torch.cat(kept)  # the prefill's last row, then a row per decode step
        rows = torch.nn.functional.cosine_similarity(decoded, full[len(ids) - 1:len(ids) - 1 + len(decoded)], dim=-1)
        assert rows.mean() > 0.99 and rows.min() > 0.95, rows
    finally:
        engine_config.install(before)


@pytest.mark.parametrize("proposer, depth", [("dspark", 3), ("dflash2", 7)])
def test_draft_rounds_emit_what_a_prefill_predicts(proposer, depth):
    """On the pinned checkpoint with its bundled DSpark drafter or the pinned DFlash2 draft: every token a speculative
    round emits (accepted drafts and the correction) is the argmax of a stand-alone prefill at its position, up to the
    model's rounding sensitivity, and the drafter earns its rounds. A verification that misread a draft row's
    convolution window or K/V, or committed the wrong window, or a draft fed the wrong taps, would emit other tokens."""
    from hibana.engine import config as engine_config
    from hibana.engine.executor import Executor
    from hibana.engine.sampler import SamplingParams
    before = engine_config.current()
    engine_config.install(before.replace(spec="7", proposer=proposer, spec_depths=f"1:{depth},2:{depth}"))
    try:
        model = load()
        ex = Executor(model, num_pages=8192, num_slots=8, max_batch=2, log=None)
        ex.enable_spec(7, proposer=proposer, state_strategy="snapshots")
        ex.capture_graphs(log=None)
        prompts = [chat_ids("Write a short story about a lighthouse keeper who finds a message in a bottle."),
                   chat_ids("Write a Python function that merges overlapping intervals, then explain how it handles intervals that only touch.")]
        reqs = [ex.admit(ids, SamplingParams(max_new_tokens=48, temperature=0.0, ignore_eos=True)) for ids in prompts]
        for r in reqs:
            ex.prefill(r)
        while not all(r.finished for r in reqs):
            ex.decode_step([r for r in reqs if not r.finished])
        for ids, r in zip(prompts, reqs):
            out = list(r.output_ids)
            full = model.logits(model.prefill(torch.tensor(ids + out, device=DEVICE))["norm"]).argmax(-1).tolist()
            agree = sum(a == b for a, b in zip(out, full[len(ids) - 1:len(ids) - 1 + len(out)])) / len(out)
            assert agree > 0.8, agree
        for r in reqs:  # the drafter earns its rounds (DFlash2 at depth 7: 3.6 tokens a round on six greedy prompts)
            assert r.spec_emitted / r.spec_rounds > (2.5 if proposer == "dflash2" else 1.3), (r.spec_emitted, r.spec_rounds)
        for r in reqs:
            ex.release(r)
    finally:
        engine_config.install(before)


@pytest.mark.parametrize("proposer", ["dspark", "dflash2"])
def test_ssd_resume_matches_a_resume_from_memory(tmp_path, proposer):
    """On the pinned checkpoint with a drafter: a prefix the SSD tier stored and restored gives back what memory held
    (each visit's K/V, its convolution windows, DSpark's context rows or DFlash2's feature window) and resumes at the
    same point to the same tokens, bit for bit, as a resume from the GPU prefix cache; DFlash2 resumes drafting
    from the window it kept with the prefix in both."""
    from hibana.engine import config as engine_config
    from hibana.engine.executor import Executor
    from hibana.engine.sampler import SamplingParams
    before = engine_config.current()
    engine_config.install(before.replace(spec="7", proposer=proposer, spec_depths="1:3", ssd=True, ssd_dir=str(tmp_path), ssd_gib=2.0,
                                         ssd_block_tokens=64))
    try:
        model = load()
        ex = Executor(model, num_pages=8192, num_slots=8, max_batch=1, log=None)
        ex.enable_spec(7, proposer=proposer, state_strategy="snapshots")
        ex.capture_graphs(log=None)
        ex.enable_ssd_tier()
        tier = ex.cache.ssd
        assert tier is not None
        tier.min_tokens = tier.min_hit = 1
        # a follow-up: the prefix up to the last message's start is what a later request resumes from (the server's
        # `message_boundary`), long enough for a few 64-token blocks and a tail
        from hibana.tokenizer import SnapshotTokenizer
        tok = SnapshotTokenizer(checkpoint())
        head = ("<|message_system|><|content_text|>Thinking effort level: 0\nYou answer questions about lighthouses. A "
                "lighthouse lens is a Fresnel lens: concentric rings of glass prisms that bend the light of one lamp into a "
                "narrow horizontal beam, so the light reaches the horizon instead of the sky or the sea at the foot of the "
                "tower. The largest lenses stand taller than a person and weigh several tonnes. They turn on a bed of "
                "mercury or on rollers, driven by a clockwork weight that a keeper wound every few hours; the turning "
                "lens sweeps the beam around, and each station's pattern of flashes tells ships which light they see. "
                "Keepers trimmed the wicks, polished the brass and glass, and logged the weather through the night."
                "<|end_message|>")
        boundary = len(tok.encode(head))
        ids = tok.encode(head + "<|message_user|><|content_text|>Why does the lens turn, and what kept it turning?<|end_message|>")
        assert 64 < boundary < len(ids)

        def run():
            req = ex.admit(ids, SamplingParams(max_new_tokens=24, temperature=0.0, ignore_eos=True, message_boundary=boundary))
            ex.prefill(req)
            while not req.finished:
                ex.decode_step([req])
            ex.release(req)
            return req.prefix_len, list(req.output_ids)

        def held(n):
            m = ex.cache.match(ids[:n], touch=False)
            assert m.length == n
            rows, kvp = m.pages[:n].long(), ex.kv_pool
            tensors = {f"pool.{k}": t[m.snapshot.slot].clone() for k, t in ex.state_pool.aux.items()}
            tensors.update({f"paux.{k}": t.index_select(0, rows).clone() for k, t in kvp.page_aux.items()})
            tensors.update({f"kv.{lid}.{s}": pool.index_select(0, rows).clone() for lid in kvp.layer_ids
                            for s, pool in (("k", kvp.k[lid]), ("v", kvp.v[lid]))})
            return tensors

        run()
        memory = run()
        assert 64 <= memory[0] <= boundary, memory[0]
        stored = held(memory[0])
        assert any(k.startswith("pool.") for k in stored)
        assert proposer != "dspark" or any(k.startswith("paux.dspark") for k in stored)
        draft = ex.spec.proposer
        if proposer == "dflash2":  # the second run resumed from memory with the window kept at the prefix
            assert (draft.cache_hits, draft.cache_fallbacks) == (1, 0)
        tier.drain(ex.cache)
        ex.cache.evict(need_pages=ex.kv_pool.num_pages)
        assert ex.cache.match(ids, touch=False).length == 0
        assert ex.cache.restore(ids) == memory[0]
        restored = held(memory[0])
        assert [k for k in stored if not torch.equal(stored[k], restored[k])] == []
        assert run() == memory
        if proposer == "dflash2":  # and from the SSD tier with the window it stored
            assert (draft.cache_hits, draft.cache_fallbacks) == (2, 0)
        ex.state.check()
        tier.drain(ex.cache)
        ex.flush_cache()
    finally:
        engine_config.install(before)


def test_a_row_alone_matches_it_beside_another(monkeypatch):
    """On the pinned checkpoint: a one-token prefill against the same token as the first row of a two-token one, which
    sees only itself. A core pass's message sources must be copies: a single row's slice of the block states is a view
    of the slot the pass rewrites, so a one-row call (a one-token chunk, decode at one request) would read the new."""
    from hibana.model.inkling import model as inkling_model
    model = load()
    ids = torch.tensor(chat_ids("Why does a lighthouse lens turn?")[:2], device=DEVICE)  # the template's first two
    mixed = []  # prefill.mix's first inputs: the block states or their views, and the passes' message sources (one slot)
    mix = inkling_model.prefill.mix
    monkeypatch.setattr(inkling_model.prefill, "mix", lambda states, *a, **k: mixed.append(states) or mix(states, *a, **k))
    alone = model.logits(model.prefill(ids[:1])["norm"])
    states = mixed[-1].untyped_storage().data_ptr()  # the output's mix reads the block states
    sources = [s for s in mixed if s.shape[1] == 1]
    assert len(sources) == 2 * 12 and all(s.untyped_storage().data_ptr() != states for s in sources)
    beside = model.logits(model.prefill(ids)["norm"][:1])
    # The two calls take other GEMM paths (one row against two), which leaves ~1e-3: 0.99901 with copies, 0.99787 reading
    # the view. The bug's size depends on the token (other pairs and visits do not separate it as well), so the storage
    # check above is the regression test and this bounds the rest.
    assert cosine(alone, beside) > 0.9984


def test_dspark_covers_the_served_context():
    """The drafter is built for the context the engine serves (HIBANA_MAX_CONTEXT), and an executor that serves more
    than it covers is refused at startup: a request past the drafter's positions would stop the engine."""
    from hibana.engine import config as engine_config
    from hibana.engine.executor import Executor
    before = engine_config.current()
    engine_config.install(before.replace(spec="7", spec_depths="1:3", max_context=40960))
    try:
        model = load()
        assert model.draft.cos.shape[0] >= 40960 + model.draft.block
        ex = Executor(model, num_pages=4096, num_slots=4, max_batch=1, max_context=40960, log=None)
        ex.enable_spec(7, proposer="dspark", state_strategy="snapshots")
        del ex
        gc.collect()
        with pytest.raises(ValueError, match="DSpark drafter covers"):
            Executor(model, num_pages=4096, num_slots=4, max_batch=1, max_context=81920, log=None).enable_spec(
                7, proposer="dspark", state_strategy="snapshots")
    finally:
        engine_config.install(before)


def chat_ids(text: str) -> list[int]:
    from hibana.tokenizer import SnapshotTokenizer
    tok = SnapshotTokenizer(checkpoint())
    return tok.encode("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>" + text + "<|end_message|>")
