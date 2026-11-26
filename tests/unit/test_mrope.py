"""Image tokens on the Qwen targets: M-RoPE rows (engine/mrope.py), the rotary table's scratch rows (model/rope.py), the
Qwen preprocessing's sizes, the API's marker and grids, and `without_specials` keeping the other added tokens' ids."""
import json
from types import SimpleNamespace

import pytest
import torch

from hibana.api import media as api_media
from hibana.engine import media
from hibana.engine.mrope import RopeLayout
from hibana.model import qwen_vision as qv
from hibana.model.rope import PartialRotary


def image(start: int, rows: int, cols: int):
    return media.Image(start=start, ids=media.placeholder_ids(bytes([start]), rows * cols), data=b"", grid=(rows, cols))


def position(lay: RopeLayout, i: int) -> tuple[int, int, int]:
    scratch: list = []
    row = lay.rows(i, i + 1, -1, scratch)[0]
    return scratch[0] if scratch else (row, row, row)


def test_positions_follow_the_reference_rule():
    """Text is (p, p, p); the block at (r, c) of an image whose first placeholder has rope start s is (s, s + r, s + c);
    the text after it resumes at s + max(rows, cols); a second image starts from the shifted text."""
    a, b = image(3, 2, 3), image(11, 4, 1)  # 6 placeholders at 3..8, then text 9, 10, then 4 placeholders at 11..14
    lay = RopeLayout((b, a))
    assert [position(lay, i) for i in range(3)] == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    assert [position(lay, i) for i in range(3, 9)] == [(3, 3, 3), (3, 3, 4), (3, 3, 5), (3, 4, 3), (3, 4, 4), (3, 4, 5)]
    assert position(lay, 9) == (6, 6, 6) and position(lay, 10) == (7, 7, 7)  # 3 + max(2, 3)
    assert [position(lay, i) for i in range(11, 15)] == [(8, 8, 8), (8, 9, 8), (8, 10, 8), (8, 11, 8)]
    assert position(lay, 15) == (12, 12, 12)
    assert lay.delta == (3 - 6) + (4 - 4) and position(lay, 40) == (40 + lay.delta,) * 3


def test_chunked_rows_are_the_whole_prompts_rows():
    lay = RopeLayout((image(2, 3, 4), image(20, 5, 2), image(31, 1, 1)))
    base, whole = 1000, []
    s_whole: list = []
    whole = lay.rows(0, 40, base, s_whole)
    for cut in range(1, 40):
        for a, b in ((0, cut), (cut, 40)):
            scratch: list = []
            rows = lay.rows(a, b, base, scratch)
            assert [scratch[r - base] if r >= base else r for r in rows] == \
                   [s_whole[r - base] if r >= base else r for r in whole[a:b]]


def test_scratch_rows_rotate_each_frequency_by_its_axis():
    rope = PartialRotary(256, 64, 1e7, 64, torch.device("cpu"), mrope_section=(11, 11, 10))
    rope.write_scratch([(5, 5, 5), (7, 20, 30)])
    assert torch.equal(rope.cos_sin[64], rope.cos_sin[5])  # a text position on every axis is the 1-D row
    axis = [0 if j % 3 == 0 or j >= 33 else 1 if j % 3 == 1 else 2 for j in range(32)]  # T, H, W interleaved
    pos = torch.tensor([(7, 20, 30)[a] for a in axis], dtype=torch.float32)
    angles = pos * rope.inv_freq
    assert torch.equal(rope.cos_sin[65], torch.cat((angles.cos(), angles.sin())))
    with pytest.raises(ValueError):
        rope.write_scratch([(0, 0, 0)] * (65_536 + 1))


@pytest.mark.parametrize("size,grid", [((200, 200), (8, 8)), ((16, 16), (8, 8)), ((48, 80), (7, 11)),
                                       ((720, 1280), (22, 40)), ((1080, 1920), (34, 60)), ((4000, 6000), (104, 156)),
                                       ((8192, 8192), (128, 128))])
def test_qwen_sizes_match_the_reference_processor(size, grid):
    """`smart_resize` cases from the reference notes (height, width) -> merged (rows, cols)."""
    h, w = qv.smart_resize(*size)
    assert (h // qv.FACTOR, w // qv.FACTOR) == grid


def test_qwen_preprocess_lays_patches_out_in_2x2_blocks():
    """Patch (row 1, col 0) is the third of block (0, 0); patch (row 0, col 2) the first of block (0, 1): sequence
    indices 2 and 4 on a 16 x 16-patch image (256 x 256 pixels, not resized)."""
    import numpy as np
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[16:32, 0:16] = 255
    rgb[0:16, 32:48] = 255
    patches, gh, gw = qv.preprocess(rgb)
    assert patches.shape == (256, 3 * 16 * 16) and (gh, gw) == (16, 16)
    assert (patches.float().mean(1) > 0).nonzero().flatten().tolist() == [2, 4]
    with pytest.raises(ValueError):
        qv.smart_resize(1, 500)


class QwenLike:
    image_token, image_start, image_end, marker = qv.IMAGE_TOKEN, qv.IMAGE_START, qv.IMAGE_END, qv.MARKER
    identity = b"test"

    def grid(self, width, height):
        return (height // 32, width // 32)


def test_the_qwen_marker_expands_with_grids():
    ids = [1, qv.IMAGE_TOKEN, 2]
    images = [api_media.Prepared(b"a" * 32, 6, b"A", (2, 3))]
    out, moved, engine_images = api_media.expand(ids, 3, images, QwenLike())
    assert out[:2] == [1, qv.IMAGE_START] and out[-2:] == [qv.IMAGE_END, 2] and len(out) == 3 + 6 + 1
    assert engine_images[0].grid == (2, 3) and engine_images[0].start == 2 and moved == 3 + 7
    assert api_media.escape("say <|vision_start|><|image_pad|><|vision_end|>", qv.MARKER) != \
        "say <|vision_start|><|image_pad|><|vision_end|>"


def test_without_specials_keeps_every_other_id():
    """Image tokens that sit past the model vocabulary (as the Qwen checkpoints' do) read as text, and the added
    tokens after them keep their ids."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from hibana.tokenizer import SnapshotTokenizer
    vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz<>|_ ")}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="_"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok.add_special_tokens(["<img>", "<think>"])
    t = SnapshotTokenizer.__new__(SnapshotTokenizer)
    t._tok = Tokenizer.from_str(tok.to_str())
    img, think = t.encode("<img>")[0], t.encode("<think>")[0]
    out = t.without_specials({img})
    assert out.encode("<think>") == [think]
    assert img not in out.encode("<img>") and len(out.encode("<img>")) == 5
    assert json.loads(out._tok.to_str())["added_tokens"][-1]["id"] == think


def test_prefill_rows_and_group_firsts_from_one_pass():
    """A batch of a text-only span and an image span starting mid-sequence: each row's rotary row, each row's
    group-first row (index - 3, for Flash-Next's indexer), and every image row written to the scratch rows once."""
    from hibana.engine.context import ForwardContext
    from hibana.engine.executor import Executor
    ex = Executor.__new__(Executor)
    ex.mrope, ex.device = True, torch.device("cpu")
    ex.model = SimpleNamespace(rope=PartialRotary(256, 64, 1e7, 64, torch.device("cpu"), mrope_section=(11, 11, 10)))
    ex.target = SimpleNamespace(indexer_budget=2048, indexer_compress_ratio=4)
    with_image = SimpleNamespace(media=(), rope=RopeLayout((image(2, 2, 3),)))  # placeholders at 2..7, delta -3 after
    ctx = ForwardContext(mode="prefill", kv_pool=None, state_pool=None, positions=torch.zeros(1))
    ex._image_rows(ctx, [(SimpleNamespace(media=(), rope=None), 5, 7, 0), (with_image, 1, 10, 2)])
    scratch = list(range(64, 70))
    assert ctx.rope_positions.tolist() == [5, 6, 1, *scratch, 5, 6]
    assert ctx.rope_first.tolist() == [2, 3, 0, 0, 0, 1, *scratch[:5]]
    assert torch.equal(ex.model.rope.cos_sin[64], ex.model.rope.cos_sin[2])  # (2, 2, 2): the image's first block
