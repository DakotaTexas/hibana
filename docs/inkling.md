# Inkling 2 Mini

`hibana serve inkling` serves the text model of `thinkingmachines/Inkling-2-Mini` on one Spark, on port 30007.
`inkling-2-mini` is an alias. The checkpoint is pinned in `hibana/model/inkling/config.py` and downloaded by the
normal launcher.

```bash
hibana serve inkling
hibana chat inkling
```

It serves text only (the checkpoint's vision tower is not loaded) and drafts with incoai's DFlash2 draft
([DFlash2](#dflash2)), or with the checkpoint's bundled DSpark drafter (`HIBANA_PROPOSER=dspark`, [DSpark](#dspark)).
`exact` serves the checkpoint's own weights and verifies every draft exactly; `fast` (the default) and `throughput`
quantize the attention projections and the vocabulary head to FP8 at load, share experts within a budget ([expert
sharing](#expert-sharing)) and accept sampled requests' drafts softly within a gate ([relaxed
verification](#relaxed-verification)), still verifying greedy requests exactly; `throughput` decodes up to 16 requests
together.
Every profile keeps the SSD tier on, as the other targets do.

## The model

A looped decoder: 24 layers of weights, of which the core (layers 6-17) runs three passes a token, so a token makes
48 layer visits. `Target` numbers the visits, in execution order, as the target's layers (`config.VISITS` maps a
visit to its layer and pass): a visit is what holds per-request state.

- **Block attention residuals.** There is no residual stream. Layers come in blocks of six; a block's sum of
  attention and MLP outputs is its state, and every sublayer input is a softmax mix, by a learned query, over the
  RMS-normalized states of the earlier blocks (the embeddings first) and the running block.
- **The loop.** Each pass restarts from the prelude's state. From the second pass on, every core sublayer input
  (query, key, value and MLP) merges a message, mixed from the previous pass's two block states, through a rank-64
  gate: `(1 - tanh(forget)) * input + tanh(gain) * message`.
- **Attention.** 24 query and 8 KV heads of 128. Queries and keys are L2-clipped, values RMS-normalized; the score
  adds a learned relative-position bias over the last 1,024 distances and the softmax includes a per-query sink.
  Twenty layers attend a sliding window of 1,024 keys, four attend everything and scale queries with the
  position's logarithm past 8,192 tokens. Keys, values, the attention output and the MLP output each pass a causal
  convolution of four taps (per-token taps on the full-attention keys and values).
- **Experts.** 128 softcapped-SwiGLU experts of width 1,152, the top 8 by sigmoid score.

## Package

| File | Contents |
|---|---|
| `config.py` | The pinned checkpoint, the geometry, the visit schedule and the convolution streams |
| `model.py` | Loading, the layer (`Layer.forward`), the message gate, `decode_visit`, `prefill_visit`, the engine contract |
| `attention.py` | Projections, convolutions, prefill's relative-position attention, decode's on the kernels |
| `moe.py` | The router and the experts: a pair at a time for decode, grouped by expert for prefill |
| `dspark.py` | The bundled DSpark drafter: its block forward, context K/V and recurrent head |
| `kernels/inkling.py`, `kernels/csrc/inkling.cu` | The decode and verification kernels |
| `kernels/inkling_prefill.py` | Prefill's kernels: attention, grouped experts, convolutions, message gates; every call's expert products |
| `kernels/inkling_dspark.py` | The drafter's kernels: low-row projections, norms and rotation, the recurrent head's gate and vocabulary step |
| `proposers/dspark.py` | The drafter's adapter to the speculative driver |

## State

- **K/V.** Every visit writes its own K/V, so the KV pool has 48 layers of one geometry: 96 KiB a token in FP8
  (the accuracy screen's `HIBANA_KV_FORMAT=sim-bf16` keeps it in BF16, for parity checks). Sliding-window visits keep their K/V for
  the whole prompt too, though they read the last 1,024 tokens: a cached prefix then resumes at any position.
- **Convolution windows.** A visit's four streams keep their last three inputs: 1.83 MiB a slot
  (`StatePool` aux `inkling_conv`, `Target.slot_state_bytes`), snapshotted like a recurrent state.
- **The SSD tier** stores a prefix's K/V with its pages' other rows (DSpark's context) and its snapshot's convolution
  windows, as it stores a recurrent state; a resume from the SSD matches one from memory bit for bit.

## Arithmetic

Two paths compute a layer visit, and `tests/gpu/test_inkling.py` holds both to `Layer.forward`, the same layer in
PyTorch operators:

- **Prefill** runs `prefill_visit` over each sequence's pages, tensor-core products with one kernel for each pass
  between them (`kernels/inkling_prefill.py`). Attention's query side (scales, clips, the position's logarithm and
  the relative bias) is one launch and each of the key and value streams one more (convolution, then the clip or
  the norm per head); the attention itself is a flash-attention kernel over the sequence's pages: tiles of 42
  tokens stream their key blocks through the two products with an online softmax, the relative bias, the sink and
  the output gate, and no score is written out. The router is one launch (scores, top 8, weights) and a second
  sorts the (token, expert) pairs by expert on the GPU, without a host sync; each projection is one launch over all
  experts on the tensor cores' block-scaled product, the quantized rows against the checkpoint's packed weights
  (the first reads each token's quantized row where its pairs need it), with one kernel for the activation and its
  quantization and one for the tokens' weighted sums. A sublayer's inputs are one launch (`mix`), which for the MLP
  also convolves the attention stream and joins its residual; the message gates are one more (`gate`: the
  normalized pair, its rank-64 read, the merge and the norm). Cold real text prefills at about 4,000 tokens/s
  (2,000 tokens) and 3,800 (9,000) in 2,048-token chunks, over half of it in the experts: every chunk streams all
  128 experts of all 48 visits, whatever its length. Prefill runs eagerly.
- **Decode and verification** (one row a request, or the pending token and its drafts; in CUDA graphs) run
  `decode_visit` on the kernels of `csrc/inkling.cu`, about 20 launches a visit: `mix` (the block attention
  residual, the message gate, the RMS norm), the projections, `prep` (queries, relative logits, convolved keys and
  values written to their page), `attention` over the pages the engine planned, `conv`, `route` and the experts on
  the checkpoint's packed NVFP4 (layers 0-19) and MXFP8 (layers 20-23) tensors: the tensor cores' block-scaled
  product a (row, expert) pair at a time (`pair_mm`), the activation in the up projection's epilogue, and above 24
  rows the pairs sorted by expert on the GPU as prefill's are. A verified row's convolutions read
  its request's earlier rows, the stored windows stay as they are, and the window after each row is kept for the
  commit (`HIBANA_VERIFY_STATE=snapshots`: there is no recurrent state to reconstruct).

The decode kernels are bound by memory traffic: one block draws about 5 GB/s of cold weights on a GB10, so every
kernel that reads more than a few tens of KB spreads over many blocks; a plain streaming read reaches about 240 GB/s,
not the 258 of the specification. The experts multiply whole packed rows a step (one row's eight NVFP4 up projections:
104 us, a pure read of their bytes 102-104). The projections of up to four rows run `gemv`, each row split over warps that load
1,536 contiguous bytes of it before multiplying (BF16 at one row: 232-245 GB/s, about a plain streaming read's rate,
against 160 through cuBLAS's GEMV). A call's rows that route to one expert each multiply it, the later ones from
cache: with real verification routing (four rows of a request read about 19 distinct experts a visit) a call costs
about its distinct experts' bytes, and grouping the rows by expert pays only from about 28 rows.

The attention projections and the vocabulary head are the checkpoint's BF16 in `exact`. `HIBANA_INKLING_TRUNK=fp8`,
which `fast` and `throughput` set, quantizes them at load (`TrunkLinear`, one scale a matrix): a decode token then
streams 3.6 GB of weights instead of 5.4. Calls of up to four rows multiply the E4M3 matrices with the BF16
activations in `gemv` (a visit's query, key and value in one launch); wider calls quantize the activations per call.
On GSM8K (200 items, four at a time) `fast` scores 95.5% and `exact` 96.0%; quantizing the activations of every call
instead scored 94.0% (paired against BF16 activations: 3 items better, 6 worse, not significant). The lossy-change
gate judges it (`tools/lossy_gate.py --target inkling`, its first layer scored on the decode step, where the FP8 trunk
and expert sharing act).

The experts read their inputs quantized as the checkpoint requires of them (`hf_quant_config.json`): an NVFP4
expert's input and activation as NVFP4 with a scale per token (of the two E4M3 block scales that put a block's
largest value at 6 and at 4, the one with the smaller absolute error), an MXFP8 expert's as MXFP8. vLLM's
implementation does the same; the transformers one multiplies unquantized activations, so its logits sit a little
further from Hibana's (mean row cosine 0.984 to 0.994 on the bring-up prompts) than they did with 16-bit
activations (0.988 to 0.999). Decode and prefill multiply the same packed rows on the same block-scaled product, a
pair at a time and grouped by expert.
A 4-bit input turns a rounding apart upstream into a code apart, so two paths over the same tokens (decode after a
prefill against one prefill) agree less closely than their kernels do: 0.999 in the logits' mean cosine.

The model is numerically sensitive: implementations that differ only in rounding (BF16 against FP32 attention
scores) agree on about 90% of greedy tokens over a 1,400-token prompt, so compare logits and task scores, not text.

## DSpark

The checkpoint's `mtp.safetensors` is a DSpark drafter: three dense layers over a block of the pending token and seven
mask rows, and a recurrent head that corrects each drafted position's logits from the previous token
(`model/inkling/dspark.py`). Its context is the target itself: every committed position's K/V for each draft layer,
projected from that layer's mix of the target's hidden states at four visits (the inputs of visits 5, 23 and 41 and
the final norm). Those K/V are per-page rows of the KV pool (12 KiB a token), written as positions commit, so a
request that resumes from a cached prefix finds its draft context in the pages it matched and the adapter
(`proposers/dspark.py`) keeps one row a request.

A round drafts the block and verifies the first 3 drafts (`HIBANA_SPEC_DEPTHS`): each verified row reads its own
experts, and the drafts past the third are rarely reached. Past four requests `exact` decodes plainly and `throughput`
keeps drafting (at 16 streams 243.3 tokens/s against 196.0); the drafter's context for each step's rows is then one batched commit
(16 requests: 79.7 ms a step, against 83.4 a request at a time and 79.6 without the drafter). Drafts are the draft's argmax; a sampled request verifies them by
rejection sampling. The draft runs in a CUDA graph per batch bucket and verify depth, about 7.6 ms a round with the
BF16 head, nearly half of it the vocabulary head, which a round reads twice: once for the draft's logits, once for
the verification's.

Served on one GB10 over the 18 h2h prompts (chat, code, math, prose, structured; thinking off, up to 1,024 tokens;
one request: the median of the prompts' decode tokens/s; streams: closed loop, all streams' tokens/s):

| | 1 request, greedy | 1 request, T 0.7 | 2 streams | 4 streams |
|---|---|---|---|---|
| `fast` | 85.2 | 77.2 | 107.9 | 144.1 |
| FP8 trunk without expert sharing | 68.5 | 65.7 | 84.5 | 109.9 |
| `exact` | 56.6 | 51.7 | 72.7 | 99.9 |
| BF16 trunk, no drafter (`--exact --spec 0`) | 39.6 | 38.9 | 61.0 | 85.7 |

The drafter earns most on math and structured output (`fast`, greedy: 81 and 75 tokens/s) and least on prose (52).
Cold real-text prompts of 2,000, 9,300 and 17,000 tokens prefill at 3,800-3,900 tokens/s in `exact` and about 4,000
in `fast`.

## DFlash2

Inkling drafts with `incoai/Inkling-2-Mini-DFlash2` by default (pinned in `model/inkling/config.py`, downloaded by
the launcher; `HIBANA_PROPOSER=dspark` serves the bundled DSpark instead): the DFlash2 drafter the 27B and Flash-Next serve (`model/parallel_draft.py`,
`proposers/parallel.py`), five layers over a block of the pending token and seven mask rows, and a selector that
walks the top 16 candidates of each position. Three things are Inkling's, as in the publisher's SGLang patch: its
fourth tap (layer id 48) is the target's final norm, which the target computes anyway (`FeatureRequest.final_norm`);
its block embeds through the target's embedding norm (`Inkling.embed_ids`); and its selector softcaps the scores after
the predecessor correction (`selector_softcap_mode: after_correction`), so the candidates come from the head's logits
before the softcap (`Inkling.head_logits`). Its context is a 2,048-position window of projected features a request,
kept with each cached prefix and stored with it by the SSD tier, as for the 27B. With it the bundled DSpark drafter is
not loaded.

A round verifies all 7 drafts at one to four requests (`HIBANA_SPEC_DEPTHS=1:7,2:7,4:7,8:0,16:0`; `throughput` at
every batch). On six chat, code and math prompts (greedy, 128 tokens) it emits 3.63 tokens a round against DSpark's
2.39 at the same depth (2.57 and 2.20 at depth 3), and every emitted token agrees with a stand-alone prefill as often
as DSpark's. Served on one GB10 over the 18 h2h prompts (as [DSpark](#dspark)'s table; one run each), `exact`:

| | 1 request, greedy | 1 request, T 0.7 | 2 streams | 4 streams |
|---|---|---|---|---|
| DSpark | 54.8 | 53.8 | 72.3 | 100.6 |
| DFlash2 | 58.6 | 60.6 | 82.6 | 111.8 |
| DFlash2 verifying 5 / 4 | 55.5 / 55.6 | 51.1 / 58.6 | 80.4 / 81.9 | 109.4 / 107.3 |

and with the FP8 trunk, exact acceptance, up to 16 requests and drafts at every batch (DSpark verifying 3, DFlash2 7),
before `fast` settled on its sharing budget ([expert sharing](#expert-sharing)):

| | 1 request, greedy | 1 request, T 0.7 | 2 | 4 | 8 | 16 streams |
|---|---|---|---|---|---|---|
| no drafter, keep 6 with 16:64 | 53.4 | 52.1 | 74.2 | 98.8 | 125.8 | 196.0 |
| DSpark, keep 6 with 16:64 | 72.1 | 67.0 | 86.7 | 112.3 | 173.3 | 243.3 |
| DFlash2, no sharing | 75.2 | 69.7 | 91.9 | 124.3 | 163.8 | 300.0 |
| DFlash2, keep 6 with 20:80 | 73.4 | 72.5 | 96.7 | 128.5 | 176.9 | 304.8 |
| DFlash2, keep 6 with 16:64 | 80.2 | 76.1 | 99.9 | 133.4 | 178.8 | 303.3 |

DFlash2 earns most on math (`exact`, greedy: 87.4 against DSpark's 64.8 tokens/s) and least on prose (44.9 against
41.4). Both drafters share experts within the same budget ([expert sharing](#expert-sharing)) and accept softly within
the same gate ([relaxed verification](#relaxed-verification)); DSpark's drafts are point masses, which the sampled rule
accepts with probability min(1, p e^s), and DFlash2's are drawn from a law near the target's.

## Expert sharing

`HIBANA_MOE_KEEP` / `HIBANA_MOE_BUDGET` ([operations](operations.md#serving-profiles)) apply to decode and verification
calls with Nemotron's selection (`route_budget`): every row keeps its top F experts, its other top-8 experts ride
along when the call's set holds them, and the set grows past the rows' floors by summed routing weight up to the cap.
The cap is per call, min(B, P x requests), and each row's top F stays in past it. `fast` and `throughput` keep 4 with
12 a request up to 48 a call with either drafter; `exact` shares none.

Accuracy, sampled at temperature 1.0 with thinking on, 16 requests a server, 3 seeds an arm, paired problem by problem
with the same drafter's `exact` run (95% bootstrap intervals over problems). LiveCodeBench v6 (454 problems) at a 256K
output cap, every arm on one build: keep 4 with 12:48 DSpark 82.6% against 83.3% (-0.66, -2.42 to +0.95), DFlash2
83.0% against 82.9% (+0.07, -1.69 to +1.84), with 2.4% and 1.8% of answers at the cap against 1.8% and 1.7%. At a
192K cap, against each drafter's `exact` run (83.5% / 83.0%), and on GPQA Diamond (198, 120K cap) against its own:

| Sharing | Experts a call reads at 16 requests | LiveCodeBench, DSpark / DFlash2 | GPQA change, DSpark / DFlash2 |
|---|---|---|---|
| none | 89.5 / 90.2 | 83.5% / 83.0% | - |
| keep 2, 6 a request, 24 a call | 50.2 / 50.7 | 80.5% / 80.4% (-2.9 / -2.6) | +2.2 / +0.7 |
| keep 2, 12 a request, 48 a call | 53.3 / 53.6 | 81.1% / 81.8% (-2.4 / -1.2) | +1.2 / 0.0 |
| keep 4, 12 a request, 48 a call (`fast`) | 68.4 / 68.9 | 82.3% / 82.3% (-1.2 / -0.7) | +2.2 / +0.3 |

GPQA, at 16 settings from keep 2 to keep 6 and 6:24 to 64:128, stayed within 1.7 points of exact either way for
DFlash2 and above it for DSpark; LiveCodeBench separates them. Experts a call reads come from replaying recorded router
traffic. At temperature 1.0 over the 18 h2h prompts (thinking off, 1,024 tokens, one Spark), keep 4 with 12:48 decodes
DSpark +20 / +9 / +26 / +17% and DFlash2 +22 / +25 / +16 / +17% over no sharing at 1 / 4 / 8 / 16 requests (`exact`
acceptance).

## Relaxed verification

`fast` accepts sampled requests' drafts with soft acceptance s = 0.25, gated to drafts the target ranks within its
top 3 and within 2.0 nats of its top token (`HIBANA_ACCEPT_SOFT_GATE=3:2.0`; a draft outside takes the exact test), and
verifies greedy requests exactly. At temperature 1.0 (18 h2h prompts, one Spark, no sharing, 1 / 4 / 8 / 16 requests)
the gated rule decodes DSpark +7 / +5 / +4 / +6% and DFlash2 +4 / +10 / +6 / +7% over exact acceptance, within noise
of the ungated rule (+5 / +5 / +6 / +7% and +10 / +6 / +8 / +12%). The GPQA numbers below are the ungated rule's. DSpark's
drafts are single tokens, so a drafted token y is accepted with probability min(1, p(y) e^s) instead of p(y): s scales
every draft's acceptance directly. GPQA Diamond (198, sampled, thinking on, `exact` with 16 requests, drafts verified at
every batch size, paired with exact acceptance at 68.7%): s = 0.25 68.7% (53 answers at the cap against 51); s = 0.5
61.6% (p 0.016, 67 at the cap, 1.09x the tokens). Sampled h2h (T 0.7) on `fast`, two passes each: one request
66.5 -> 73.8 tok/s at s = 0.25 (75.1 at 0.5), 2 streams 85.0 -> 88.2, 4 streams 110.9 -> 114.8. Greedy relaxed
acceptance cost MMLU-Pro greedy (200, thinking on): top-3 within 2.0 77.0% against 83.5% (p 0.004), top-2 within 1.0
79.5% against 82.0% (p 0.27), so greedy requests stay exact.

`fast` as served, the gated rule at s = 0.25 with keep 4 with 12:48 sharing, paired with the same build's `exact` (16
requests a server, LiveCodeBench v6 at 256K with 4 seeds, GPQA Diamond at 120K with 3): DSpark -0.66 points (-2.07 to
+0.73) and -0.51 (-2.36 to +1.35), DFlash2 -0.07 (-1.89 to +1.69) and -0.51 (-2.86 to +2.02). At s = 0.5 the same
stack lost LiveCodeBench 2.1 points with DSpark (-3.89 to -0.44) and 2.8 with DFlash2 (-4.77 to -0.73), its answers
20-24% longer and more of them at the cap. At temperature 1.0 over the 18 h2h prompts (thinking off, 1,024 tokens, one
Spark, drafts at every batch; 8 and 16 requests are `throughput`) it decodes, against `exact`:

| tokens/s | 1 | 4 | 8 | 16 requests |
|---|---|---|---|---|
| DSpark, `exact` | 62.2 | 107.5 | 156.4 | 246.2 |
| DSpark, `fast` | 78.2 | 126.2 | 207.4 | 303.0 |
| DFlash2, `exact` | 69.2 | 126.7 | 191.8 | 303.4 |
| DFlash2, `fast` | 93.0 | 164.7 | 241.4 | 369.4 |

## Memory

Loaded weights take 18 GiB: the packed experts (12.4 GiB), which every path reads as stored, the trunk, embeddings
and head, the routers and gates, and the drafter (0.7 GiB). The pools are planned from what remains.

## Chat format

Inkling frames a turn with tokens, not text tags (`Target.chat_format = "inkling"`). The chat template renders in
transformers' dialect (`api/chat_template.py`), `enable_thinking: false` becomes the template's
`reasoning_effort: "none"` (a request's `chat_template_kwargs` can set `reasoning_effort` itself), and
`api/parsers/inkling.py` splits a reply's `<|content_thinking|>`, `<|content_text|>` and
`<|content_invoke_tool_json|>` messages into reasoning, content and tool calls.
