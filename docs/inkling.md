# Inkling 2 Mini

`hibana serve inkling` serves the text model of `thinkingmachines/Inkling-2-Mini` on one Spark, on port 30007.
`inkling-2-mini` is an alias. The checkpoint is pinned in `hibana/model/inkling/config.py` and downloaded by the
normal launcher.

```bash
hibana serve inkling
hibana chat inkling
```

It serves text only (the checkpoint's vision tower is not loaded) and drafts with the checkpoint's bundled DSpark
drafter, verifying drafts exactly in every profile. `exact` serves the checkpoint's own weights; `fast` (the
default) and `throughput` quantize the attention projections and the vocabulary head to FP8 at load and share
experts within a budget ([expert sharing](#expert-sharing)), and `throughput` decodes up to 16 requests together.
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
  product a (row, expert) pair at a time (`pair_mm`), the activation in the up projection's epilogue, and above 192
  pairs grouped by expert on the GPU. A verified row's convolutions read
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
experts, and the drafts past the third are rarely reached. Past four requests the rows stop sharing a round's
fixed cost and the profiles decode plainly; the drafter's context for each step's rows is then one batched commit
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

## Expert sharing

`HIBANA_MOE_KEEP` / `HIBANA_MOE_BUDGET` ([operations](operations.md#serving-profiles)) apply to decode and verification
calls with Nemotron's selection (`route_budget`): every row keeps its top F experts, its other top-8 experts ride
along when the call's set holds them, and the set grows past the rows' floors by summed routing weight up to the cap.
`fast` keeps 6 with 16 a request up to 64 a call; `throughput` keeps 4 with a cap of 20, which does better at 8-16
streams. `exact` shares none. Closed-loop streams over the 18 h2h prompts (thinking off, greedy, `HIBANA_MAX_BATCH=16`),
all streams' tokens/s, off against keep 4 with a cap of 20:

| | 1 | 2 | 4 | 8 | 16 streams |
|---|---|---|---|---|---|
| BF16 trunk, no drafter | 38.1 -> 37.7 | 58.9 -> 58.6 | 81.4 -> 97.0 | 114.4 -> 152.2 | 160.2 -> 211.3 |
| `exact` (verify 3 at 1-4 requests) | 45.4 -> 44.7 | 68.4 -> 81.5 | 95.6 -> 117.8 | 111.6 -> 153.1 | 153.1 -> 199.2 |

On the replies' teacher-forced routes, keep 4 with a cap of 20 reads 19.7 experts a visit instead of 26.3 at four
streams (88% of the routing weight kept), 25.4 instead of 44.2 at eight (77%) and 41.3 instead of 66.2 at sixteen
(83%). One stream reads its own 8 either way. GSM8K-200 four at a time (`exact`, to 1,024 tokens), paired against
sharing off (95.0%): keep 4 with a cap of 20 93.0% (1 better / 5 worse, p 0.22); keep 4 with 6 a request up to 48,
which also cuts a request alone to 6 experts a row (1 request: 45.4 -> 53.7 tokens/s), 94.5% (3 / 4).

GSM8K did not show what long reasoning does. On GPQA Diamond (198 items, sampled at temperature 1.0 with thinking on,
16,384-token cap), with `fast`'s FP8 trunk and DSpark, against `exact` without drafts (71.7%):

| Sharing | Experts a 4-row verify call reads (of 18.1) | Verify rows' routing weight kept | GPQA | Paired |
|---|---|---|---|---|
| none | 18.1 | 100% | 69.7% | p 0.57 |
| keep 6, 16 a request (`fast`) | 15.9 | 95.8% | 69.7% | p 0.60 |
| keep 6, 12 a request | 14.4 | 92.3% | 66.2% | p 0.052 |
| keep 5, 10 a request | 12.2 | 86.5% | 67.2% | p 0.12 |
| keep 4, 6 a request | 9.8 | 77.7% | 62.1% | p 0.001 |

The routing columns come from replaying `route_budget` over Inkling's own GPQA reasoning, one call per 4-row window.
Too little kept routing shows as longer reasoning: keep 4 at 6 a request wrote 1.11x the tokens and ran 66 answers into
the cap against `exact`'s 47; the BF16 trunk with that sharing scored 63.6%, so the sharing, not the FP8 trunk, cost it.

## Memory

Loaded weights take 18 GiB: the packed experts (12.4 GiB), which every path reads as stored, the trunk, embeddings
and head, the routers and gates, and the drafter (0.7 GiB). The pools are planned from what remains.

## Chat format

Inkling frames a turn with tokens, not text tags (`Target.chat_format = "inkling"`). The chat template renders in
transformers' dialect (`api/chat_template.py`), `enable_thinking: false` becomes the template's
`reasoning_effort: "none"` (a request's `chat_template_kwargs` can set `reasoning_effort` itself), and
`api/parsers/inkling.py` splits a reply's `<|content_thinking|>`, `<|content_text|>` and
`<|content_invoke_tool_json|>` messages into reasoning, content and tool calls.
