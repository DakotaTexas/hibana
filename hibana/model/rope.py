"""Partial rotary embedding (NeoX style) over the first rotary_dim of each head.

For text the multimodal rope sections collapse to one 1-D position, so this is plain RoPE with rotary_dim = head_dim *
partial_rotary_factor. With `mrope_section`, rows past the 1-D table hold one forward's image-token rotations
(`write_scratch`, engine/mrope.py), so every rope kernel keeps reading `cos_sin[row]`.
"""

from __future__ import annotations

import torch

from hibana.engine.upload import upload

SCRATCH_ROWS = 65_536  # image rows one forward can rotate (16 MiB of FP32 cos/sin)


class PartialRotary:
    def __init__(self, head_dim: int, rotary_dim: int, base: float, max_positions: int, device: torch.device,
                 mrope_section: tuple[int, int, int] | None = None):
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.max_positions = max_positions
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
        t = torch.arange(max_positions, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        # Keep the target's cache and rotary products in FP32. Rounding either
        # to BF16 changes attention, including the first prefill chunk.
        self.cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.inv_freq = inv_freq
        if mrope_section is not None:
            # interleaved sections, as the reference's `apply_interleaved_rope`: frequency j reads t, h or w by j % 3
            j = torch.arange(rotary_dim // 2, device=device)
            self.axis = torch.where((j % 3 == 1) & (j < 3 * mrope_section[1]), 1,
                                    torch.where((j % 3 == 2) & (j < 3 * mrope_section[2]), 2, 0))
            self.cos_sin = torch.cat((self.cos_sin, self.cos_sin[:1].expand(SCRATCH_ROWS, -1)))

    def write_scratch(self, positions: list[tuple[int, int, int]]) -> None:
        """The cos/sin of each (t, h, w) into the rows from `max_positions` on, for the next forward."""
        if not positions:
            return
        if len(positions) > SCRATCH_ROWS:
            raise ValueError(f"{len(positions)} image rows in one forward; the rotary table has {SCRATCH_ROWS} scratch rows")
        p = upload(positions, torch.int64, self.inv_freq.device)
        angles = p.gather(1, self.axis.expand(len(positions), -1)).float() * self.inv_freq
        half = self.rotary_dim // 2
        rows = self.cos_sin[self.max_positions:self.max_positions + len(positions)]
        rows[:, :half], rows[:, half:] = angles.cos(), angles.sin()

    def __call__(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """x: [T, heads, head_dim] bf16. Returns a new tensor."""
        return rotate(x, self.cos_sin[positions], self.rotary_dim)


def rotate(x: torch.Tensor, cos_sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """x [T, heads, D] with its first rotary_dim dims rotated (NeoX halves, in FP32) by the table rows cos_sin."""
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rot = x[..., :rotary_dim]
    rest = x[..., rotary_dim:]
    x1, x2 = rot.float().chunk(2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1.to(x.dtype), o2.to(x.dtype), rest), dim=-1)
