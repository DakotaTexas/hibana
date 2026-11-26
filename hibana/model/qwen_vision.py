# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 vision tower, shared by Flash-Next and the 27B: a 27-block ViT (width 1152, 16-pixel patches, a learned
48 x 48 position table, 2-D rotary attention) and a merger turning each 2 x 2 block of patches into one embedding of the
language model's width. Preprocessing follows the checkpoints' `Qwen2VLImageProcessor`; the arithmetic follows
transformers' `modeling_qwen3_5.py` in BF16 with FP32 rotary and position interpolation. Image tokens are rotated by 2-D
positions in the language model (M-RoPE, engine/mrope.py), so the encoder also reports each image's merged `grid`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

PREFIX = "model.visual"
PATCH, MERGE, TEMPORAL = 16, 2, 2
FACTOR = PATCH * MERGE
MIN_PIXELS, MAX_PIXELS = 65_536, 16_777_216  # preprocessor_config.json size.shortest_edge / longest_edge
WIDTH, HEADS, LAYERS, MLP = 1152, 16, 27, 4304
HEAD_DIM = WIDTH // HEADS
POS_SIDE = 48
ROPE_THETA = 10_000.0
LN_EPS = 1e-6
VISION_BYTES = 1_000_000_000  # the BF16 tower and merger as loaded (~0.92 GB): the load's headroom
WORKSPACE_PER_PATCH = 38_000  # one encode's device memory beyond its output: 34 KB measured on GB10, with headroom
IMAGE_TOKEN, IMAGE_START, IMAGE_END = 248056, 248053, 248054
MARKER = "<|vision_start|><|image_pad|><|vision_end|>"  # what chat_template.jinja writes for an image part


def smart_resize(height: int, width: int, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """(height, width), multiples of 32 within [MIN_PIXELS, max_pixels], as the reference processor rounds them."""
    if min(height, width) < 1 or max(height, width) / min(height, width) > 200:
        raise ValueError(f"a {width} x {height} image is outside the aspect ratio limit of 200")
    h = round(height / FACTOR) * FACTOR
    w = round(width / FACTOR) * FACTOR
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h * w < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h = math.ceil(height * beta / FACTOR) * FACTOR
        w = math.ceil(width * beta / FACTOR) * FACTOR
    return h, w


def preprocess(rgb: np.ndarray, max_pixels: int = MAX_PIXELS) -> tuple[torch.Tensor, int, int]:
    """An RGB [H, W, 3] uint8 image -> (patches [n, 3 * 16 * 16] BF16 in 2 x 2-block order, patch rows, patch cols), on
    the CPU, values in [-1, 1]; one temporal slot (`encode` duplicates it)."""
    height, width = smart_resize(rgb.shape[0], rgb.shape[1], max_pixels)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
    if t.shape[-2:] != (height, width):
        # torch's native uint8 bicubic, as torchvision resizes uint8 on the CPU: an FP32 resize differs by up to ~20 levels
        t = F.interpolate(t, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    x = (t.float() - 127.5) / 127.5
    gh, gw = height // PATCH, width // PATCH
    x = x.reshape(3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH).permute(1, 4, 2, 5, 0, 3, 6)
    return x.reshape(gh * gw, 3 * PATCH * PATCH).to(torch.bfloat16), gh, gw


class QwenVision:
    """The tower and merger on the device (engine/contracts.VisionEncoder)."""

    image_token, image_start, image_end, marker, mrope = IMAGE_TOKEN, IMAGE_START, IMAGE_END, MARKER, True

    def __init__(self, w, device: torch.device, max_pixels: int = MAX_PIXELS):
        def get(name: str) -> torch.Tensor:
            return w.get(f"{PREFIX}.{name}").to(device)

        if not MIN_PIXELS <= max_pixels <= MAX_PIXELS:
            raise ValueError(f"the pixel cap must be in [{MIN_PIXELS:,}, {MAX_PIXELS:,}]")
        self.device, self.max_pixels = device, max_pixels
        # part of every image's digest: a prefix encoded under other preprocessing is never resumed
        self.identity = (f"qwen3.8 vit: smart_resize factor {FACTOR} pixels {MIN_PIXELS}-{max_pixels}, torchvision bicubic "
                         "antialias on uint8, x/127.5-1, 2x2-block patch order\n").encode()
        conv = get("patch_embed.proj.weight")  # Conv3d with kernel = stride: a linear on the patch's 1536 values
        if conv.shape != (WIDTH, 3, TEMPORAL, PATCH, PATCH):
            raise ValueError(f"unexpected patch embedding {tuple(conv.shape)}")
        self.patch_w, self.patch_b = conv.reshape(WIDTH, -1), get("patch_embed.proj.bias")
        self.pos = get("pos_embed.weight")
        names = ("norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
                 "norm2.weight", "norm2.bias", "mlp.linear_fc1.weight", "mlp.linear_fc1.bias", "mlp.linear_fc2.weight",
                 "mlp.linear_fc2.bias")
        self.blocks = [[get(f"blocks.{i}.{n}") for n in names] for i in range(LAYERS)]
        self.merger = [get(f"merger.{n}") for n in ("norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias",
                                                     "linear_fc2.weight", "linear_fc2.bias")]
        self.hidden = self.merger[4].shape[0]
        self.inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM // 2, 2, dtype=torch.float32, device=device)
                                              / (HEAD_DIM // 2)))
        self.workspace_bytes = (max_pixels // PATCH**2) * WORKSPACE_PER_PATCH

    @property
    def bytes(self) -> int:
        tensors = [self.patch_w, self.patch_b, self.pos, *self.merger, *(t for b in self.blocks for t in b)]
        return sum(t.numel() * t.element_size() for t in tensors)

    def grid(self, width: int, height: int) -> tuple[int, int]:
        """(rows, cols) of embeddings a `width` x `height` image becomes."""
        h, w = smart_resize(height, width, self.max_pixels)
        return h // FACTOR, w // FACTOR

    def tokens(self, width: int, height: int) -> int:
        rows, cols = self.grid(width, height)
        return rows * cols

    def preprocess(self, rgb: np.ndarray) -> tuple[torch.Tensor, int, int]:
        return preprocess(rgb, self.max_pixels)

    def _block_coords(self, gh: int, gw: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Patch (row, col) of each patch in 2 x 2-block order."""
        i = torch.arange(gh * gw, device=self.device)
        block = i // (MERGE * MERGE)
        return (block // (gw // MERGE)) * MERGE + (i // MERGE) % MERGE, (block % (gw // MERGE)) * MERGE + i % MERGE

    def _positions(self, rows: torch.Tensor, cols: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        """The position table resampled to the grid (bilinear, align_corners, edges clamped; FP32 as the reference)."""
        def axis(index, size):
            src = index.float() * (POS_SIDE - 1) / max(size - 1, 1)
            floor = torch.floor(src)
            offsets = torch.arange(2, device=self.device)
            taps = (floor.long()[:, None] + offsets).clamp(0, POS_SIDE - 1)
            return taps, (1 - (src[:, None] - floor[:, None] - offsets).abs()).clamp(min=0)

        h_taps, h_w = axis(rows, gh)
        w_taps, w_w = axis(cols, gw)
        idx = (h_taps[:, :, None] * POS_SIDE + w_taps[:, None, :]).reshape(-1, 4)
        weights = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
        out = self.pos[idx[:, 0]] * weights[:, :1]  # tap by tap: a [n, 4, 1152] FP32 product took ~1.8 GB at the cap
        for k in range(1, 4):
            out += self.pos[idx[:, k]] * weights[:, k:k + 1]
        return out.to(torch.bfloat16)

    @torch.inference_mode()
    def encode(self, pixels) -> torch.Tensor:
        """`preprocess`'s (patches, rows, cols) -> [rows * cols / 4, hidden] BF16 embeddings in block raster order."""
        from hibana.kernels.vision import add_layer_norm, rope_qk
        patches, gh, gw = pixels
        x = patches.to(self.device, torch.bfloat16)
        n = x.shape[0]
        x = x.view(n, 3, 1, PATCH * PATCH).expand(n, 3, TEMPORAL, PATCH * PATCH).reshape(n, -1)
        rows, cols = self._block_coords(gh, gw)
        h = F.linear(x, self.patch_w, self.patch_b) + self._positions(rows, cols, gh, gw)
        f = torch.cat([rows.float()[:, None] * self.inv_freq, cols.float()[:, None] * self.inv_freq], dim=-1)
        f = torch.cat([f, f], dim=-1)  # frequencies laid out [rows, cols, rows, cols] over the 72 dims
        cos, sin = f.cos(), f.sin()
        x = F.layer_norm(h, (WIDTH,), self.blocks[0][0], self.blocks[0][1], LN_EPS)
        for i, (_, _, qkv_w, qkv_b, proj_w, proj_b, n2w, n2b, fc1_w, fc1_b, fc2_w, fc2_b) in enumerate(self.blocks):
            qkv = F.linear(x, qkv_w, qkv_b).view(n, 3, HEADS, HEAD_DIM)
            q, k = rope_qk(qkv, cos, sin)  # patch-major: q, k and v reach flash attention without copies
            a = F.scaled_dot_product_attention(q[None].transpose(1, 2), k[None].transpose(1, 2),
                                               qkv[:, 2][None].transpose(1, 2))
            h, x = add_layer_norm(h, F.linear(a[0].transpose(0, 1).reshape(n, WIDTH), proj_w, proj_b), n2w, n2b, LN_EPS)
            m = torch._addmm_activation(fc1_b, x, fc1_w.t(), use_gelu=True)  # tanh-GELU in the GEMM epilogue
            nw, nb = self.blocks[i + 1][:2] if i + 1 < LAYERS else self.merger[:2]
            h, x = add_layer_norm(h, F.linear(m, fc2_w, fc2_b), nw, nb, LN_EPS)
        _, _, f1w, f1b, f2w, f2b = self.merger
        return F.linear(F.gelu(F.linear(x.view(-1, WIDTH * MERGE * MERGE), f1w, f1b)), f2w, f2b)


def mrope_section(cfg: dict) -> tuple[int, int, int]:
    """The text model's interleaved M-RoPE sections (`rope_parameters`)."""
    rope = cfg["text_config"].get("rope_parameters") or {}
    section = tuple(rope.get("mrope_section") or ())
    if not rope.get("mrope_interleaved") or len(section) != 3:
        raise ValueError(f"expected interleaved M-RoPE in the text config, got {rope}")
    return section


def load(w, device: torch.device, cfg: dict, max_pixels: int, log=print) -> QwenVision:
    """The tower and merger of a Qwen3.8 checkpoint, its config checked against this module's constants."""
    v = cfg.get("vision_config") or {}
    expected = {"depth": (v.get("depth"), LAYERS), "hidden_size": (v.get("hidden_size"), WIDTH),
                "num_heads": (v.get("num_heads"), HEADS), "intermediate_size": (v.get("intermediate_size"), MLP),
                "patch_size": (v.get("patch_size"), PATCH), "spatial_merge_size": (v.get("spatial_merge_size"), MERGE),
                "temporal_patch_size": (v.get("temporal_patch_size"), TEMPORAL),
                "num_position_embeddings": (v.get("num_position_embeddings"), POS_SIDE ** 2),
                "deepstack_visual_indexes": (v.get("deepstack_visual_indexes"), []),
                "image_token_id": (cfg.get("image_token_id"), IMAGE_TOKEN),
                "vision_start_token_id": (cfg.get("vision_start_token_id"), IMAGE_START),
                "vision_end_token_id": (cfg.get("vision_end_token_id"), IMAGE_END)}
    wrong = {k: got for k, (got, want) in expected.items() if got != want}
    if wrong:
        raise ValueError(f"unexpected vision config: {wrong}")
    vision = QwenVision(w, device, max_pixels)
    if vision.hidden != cfg["text_config"]["hidden_size"]:
        raise ValueError(f"the merger makes {vision.hidden}-wide embeddings for a {cfg['text_config']['hidden_size']}-wide model")
    if log:
        log(f"  vision: Qwen ViT and merger ({vision.bytes / 2**30:.2f} GiB), images up to {max_pixels:,} pixels")
    return vision
