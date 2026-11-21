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
    """Decode's experts (the route kernel, a product a (row, expert) pair, or at 40 rows the pairs grouped by expert on
    the device) against prefill's (the wide route kernel, the pairs sorted by expert) on the same packed NVFP4 / MXFP8
    tensors."""
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


def test_dspark_rounds_emit_what_a_prefill_predicts():
    """On the pinned checkpoint with its bundled drafter: every token a speculative round emits (accepted drafts and
    the correction) is the argmax of a stand-alone prefill at its position, up to the model's rounding sensitivity,
    and the drafter earns its rounds. A verification that misread a draft row's convolution window or K/V, or
    committed the wrong window, would emit other tokens."""
    from hibana.engine import config as engine_config
    from hibana.engine.executor import Executor
    from hibana.engine.sampler import SamplingParams
    before = engine_config.current()
    engine_config.install(before.replace(spec="7", spec_depths="1:3,2:3"))
    try:
        model = load()
        ex = Executor(model, num_pages=8192, num_slots=8, max_batch=2, log=None)
        ex.enable_spec(7, proposer="dspark", state_strategy="snapshots")
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
        for r in reqs:  # the drafter is right more often than not
            assert r.spec_emitted / r.spec_rounds > 1.3, (r.spec_emitted, r.spec_rounds)
        for r in reqs:
            ex.release(r)
    finally:
        engine_config.install(before)


def chat_ids(text: str) -> list[int]:
    from hibana.tokenizer import SnapshotTokenizer
    tok = SnapshotTokenizer(checkpoint())
    return tok.encode("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>" + text + "<|end_message|>")
