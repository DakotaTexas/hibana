# SPDX-License-Identifier: Apache-2.0
"""Pinned Inkling-2-Mini checkpoint and geometry (Thinking Machines' release: NVFP4 and MXFP8 routed experts, BF16 trunk).

24 decoder layers of weights; the 12 core layers run three passes a token, so a token makes 48 layer visits. A visit
is what holds per-request state (its own K/V and its own four convolution windows), so `hibana.target` numbers the
visits, in execution order, as the target's layers: pools and features address a visit, `VISITS` maps it back to
the layer whose weights it runs and the pass it belongs to.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
MODEL_REVISION = "6e7c8248f75265bdd4fbab5273304243b5d8c2cb"
MODEL_CACHE_KEY = "models--thinkingmachines--Inkling-2-Mini"
MODEL_ARCHITECTURE = "Inkling2LoopForConditionalGeneration"
PREFIX = "model.llm"

# Block drafters trained on this checkpoint's features besides its bundled DSpark (`mtp.safetensors`)
DRAFT_PINS = {"dflash2": ("incoai/Inkling-2-Mini-DFlash2", "4f9088ea66c21f2b99ed657d1941c306b03e84d2")}

STOP_TOKEN_IDS = (200006,)  # <|content_model_end_sampling|> (generation_config.json)

HIDDEN_SIZE = 2304
NUM_LAYERS = 24
VOCAB_SIZE = 200_058  # the unembedding's rows; the embedding table is padded to 201,024
RMS_NORM_EPS = 1e-6
MAX_POSITION_EMBEDDINGS = 1_048_576

NUM_ATTENTION_HEADS = 24
NUM_KV_HEADS = 8
HEAD_DIM = 128
D_REL = 16  # per-head width of the relative-position states
REL_EXTENT = 1024  # backward distances that carry a relative-position bias
SLIDING_WINDOW = 1024  # keys a sliding-window layer attends, the query's own included
LOCAL_LAYERS = tuple(i for i in range(NUM_LAYERS) if (i + 1) % 6)  # sliding-window layers; every sixth attends everything
QK_CLIP = 22.627416997969522
REL_CLIP = 8.0
LOG_SCALING_FLOOR = 8192
LOG_SCALING_ALPHA = 0.1

CONV_KERNEL = 4
BLOCK_LAYERS = 6  # layers summed into one block of the attention residuals

NUM_EXPERTS = 128
EXPERTS_PER_TOKEN = 8
MOE_INTERMEDIATE_SIZE = 1152
ROUTE_SCALE = 8.0
GATE_SOFTCAP = 4.0
UP_SOFTCAP = 25.0
LOGIT_SOFTCAP = 30.0
NVFP4_GROUP_SIZE = 16

LOOP_START, LOOP_END, LOOP_PASSES, GATE_RANK = 6, 18, 3, 64
# (layer, pass) of every visit, in execution order: the prelude, the core once per pass, the coda.
VISITS = (tuple((i, 0) for i in range(LOOP_START))
          + tuple((i, p) for p in range(LOOP_PASSES) for i in range(LOOP_START, LOOP_END))
          + tuple((i, 0) for i in range(LOOP_END, NUM_LAYERS)))
NUM_VISITS = len(VISITS)

# Streams of a visit's convolution window, concatenated: key and value projections, attention and MLP outputs.
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
CONV_STREAMS = {"k": (0, KV_SIZE), "v": (KV_SIZE, KV_SIZE), "attn": (2 * KV_SIZE, HIDDEN_SIZE), "mlp": (2 * KV_SIZE + HIDDEN_SIZE, HIDDEN_SIZE)}
CONV_WIDTH = 2 * KV_SIZE + 2 * HIDDEN_SIZE

# Resident weights: the packed experts (12.4 GiB), the BF16 trunk, embeddings and head (3.6 GiB), the routers and
# gates, the drafter; the load stages one layer's experts at a time.
WEIGHT_LOAD_PEAK_BYTES = 22 * 2**30
PREFILL_CHUNK = 2048

# config.json `text_config` -> the value this target was built for.
EXPECTED = {
    "hidden_size": HIDDEN_SIZE, "num_hidden_layers": NUM_LAYERS, "vocab_size": 201_024, "unpadded_vocab_size": VOCAB_SIZE,
    "num_attention_heads": NUM_ATTENTION_HEADS, "num_key_value_heads": NUM_KV_HEADS, "head_dim": HEAD_DIM,
    "swa_num_attention_heads": NUM_ATTENTION_HEADS, "swa_num_key_value_heads": NUM_KV_HEADS, "swa_head_dim": HEAD_DIM,
    "d_rel": D_REL, "rel_extent": REL_EXTENT, "sliding_window_size": SLIDING_WINDOW, "local_layer_ids": list(LOCAL_LAYERS),
    "log_scaling_n_floor": LOG_SCALING_FLOOR, "log_scaling_alpha": LOG_SCALING_ALPHA, "rms_norm_eps": RMS_NORM_EPS,
    "qk_l2_clip_threshold": QK_CLIP, "rel_l2_clip_threshold": REL_CLIP, "sconv_kernel_size": CONV_KERNEL,
    "n_layers_per_attn_res_block": BLOCK_LAYERS, "attn_res_normalize_values": True, "use_value_norm_no_gamma": True,
    "n_routed_experts": NUM_EXPERTS, "num_experts_per_tok": EXPERTS_PER_TOKEN, "n_shared_experts": 0,
    "intermediate_size": MOE_INTERMEDIATE_SIZE, "mlp_gate_softcap": GATE_SOFTCAP, "mlp_up_softcap": UP_SOFTCAP,
    "route_scale": ROUTE_SCALE, "use_gate_bias": True, "final_logit_softcapping": LOGIT_SOFTCAP, "q_bias": False, "o_bias": False,
    "loop_config": {"core_start": LOOP_START, "core_end": LOOP_END, "num_passes": LOOP_PASSES, "gate_rank": GATE_RANK, "is_varloop": False},
}


def check_config(target, snap: Path) -> dict:
    """config.json of the snapshot, after checking its text model against the pinned geometry; raises on a mismatch."""
    cfg = json.loads((snap / "config.json").read_text())
    text = cfg["text_config"]
    wrong = {k: (text.get(k), v) for k, v in EXPECTED.items() if text.get(k) != v}
    if cfg["architectures"] != [target.architecture] or wrong:
        raise ValueError(f"unexpected model config: arch={cfg['architectures']} mismatches={wrong}")
    return cfg
