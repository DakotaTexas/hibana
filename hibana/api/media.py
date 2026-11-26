"""Image parts of a chat request, for a server whose target serves a vision encoder (`model.vision`:
engine/contracts.VisionEncoder, model/nemotron/vision.py).

An image arrives as an OpenAI `image_url` (or `input_image`) part holding a `data:` URL, or as an Anthropic `image`
block with a base64 source (or a `data:` URL); the server fetches nothing. `prepare` reads each image's header, which
sets its placeholder count, and keeps the encoded bytes; the engine decodes an image only when its span is encoded
(engine/media.py), so an image in a cached prefix is never decoded again. A request's new image is decoded once here
as well, so that one that does not decode is a 400 (or, in a tool result, a note) rather than a failed engine step.

The template writes the encoder's `marker` for each image part. `escape` keeps the same text in a message from counting
as one, and `Markers` encodes the render with each marker as the image token and the rest as text. `expand` replaces each
such token with the image start, its placeholder ids and the image end, as the checkpoints' processors do, and returns
the images (with their grids) for the engine.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass

from hibana.api.messages import MEDIA_NOTE
from hibana.engine import media
from hibana.engine.sampler import InvalidRequest

IMAGE_PARTS = ("image_url", "input_image", "image")
MAX_IMAGE_BYTES = 32 * 2**20  # an encoded image
MAX_IMAGES = 100  # in one request (Anthropic's limit); their tokens are bounded by the context
_DECODED: dict[bytes, None] = {}  # digests of images that decoded here: a repeat is not decoded again to check it


@dataclass
class Prepared:
    digest: bytes  # SHA-256 of the encoder's identity and the encoded bytes
    tokens: int  # the embeddings it becomes
    data: bytes  # the encoded image
    grid: tuple[int, int] | None = None  # (rows, cols) of embeddings


def image_bytes(part: dict) -> bytes:
    """The encoded image of an OpenAI or Anthropic image part; ValueError saying why there is none."""
    if part.get("type") == "image":  # Anthropic: {"source": {"type": "base64", "media_type": ..., "data": ...}} or a URL source
        source = part.get("source")
        if not isinstance(source, dict):
            raise ValueError("an image block needs a source")
        if source.get("type") == "base64":
            return _b64(source.get("data"))
        url = source.get("url")
    else:  # OpenAI: {"image_url": {"url": ...}} (chat) or {"image_url": "..."} (Responses-style input_image)
        url = part.get("image_url")
        url = url.get("url") if isinstance(url, dict) else url
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("send images inline, as a data: URL or base64; this server does not fetch URLs")
    header, _, data = url.partition(",")
    if not header.endswith(";base64"):
        raise ValueError("a data: URL image must be base64")
    return _b64(data)


def _b64(data) -> bytes:
    if not isinstance(data, str):
        raise ValueError("image data must be a base64 string")
    if len(data) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise ValueError(f"the image is larger than {MAX_IMAGE_BYTES >> 20} MiB")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("the image data is not valid base64") from None


def _read(part: dict, vision, seen: dict[bytes, Prepared], decode: bool) -> Prepared:
    raw = image_bytes(part)
    digest = hashlib.sha256(vision.identity + raw).digest()
    if digest not in seen:
        width, height = media.size(raw)
        grid, tokens = vision.grid(width, height), vision.tokens(width, height)
        if decode and digest not in _DECODED:
            media.decode(raw)
            if len(_DECODED) >= 4096:
                _DECODED.clear()
            _DECODED[digest] = None
        seen[digest] = Prepared(digest, tokens, raw, grid)
    return seen[digest]


def prepare(messages: list[dict], vision, decode: bool = True) -> tuple[list[dict], list[Prepared]]:
    """`messages` with each image of a tool message that cannot be read replaced by a note saying why (an agent's
    screenshot that failed is not the user's mistake), and every image, in the order the template renders them (an
    image sent twice is read once). `decode`: also decode each image not decoded before (count_tokens reads headers
    only). An image elsewhere that cannot be read is a 400."""
    out: list[dict] = []
    images: list[Prepared] = []
    seen: dict[bytes, Prepared] = {}
    for i, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list) or not any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in content):
            out.append(message)
            continue
        parts = []
        for j, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") in IMAGE_PARTS:
                try:
                    images.append(_read(part, vision, seen, decode))
                except ValueError as exc:
                    if message.get("role") != "tool":
                        raise InvalidRequest(f"messages[{i}].content[{j}]: {exc}") from None
                    part = {"type": "text", "text": MEDIA_NOTE.format(kind="image", why=exc)}
            parts.append(part)
        out.append({**message, "content": parts})
    if len(images) > MAX_IMAGES:
        raise InvalidRequest(f"the request has {len(images)} images; at most {MAX_IMAGES} are read")
    return out, images


def _escaped(marker: str) -> str:
    """The marker in the request's own text while rendered: a private-use character for its "<"."""
    return "\ue000" + marker[1:]


def escape(value, marker: str):
    """`value` (messages or tools, at any nesting) with the image marker in its strings escaped for the render."""
    if isinstance(value, str):
        return value.replace(marker, _escaped(marker))
    if isinstance(value, list):
        return [escape(v, marker) for v in value]
    if isinstance(value, dict):
        return {k: escape(v, marker) for k, v in value.items()}
    return value


class Markers:
    """`tok` for a render of escaped messages: each marker the template wrote is the image token, and the rest, the
    escapes undone, is encoded with the image tokens read as text. Offsets are the render's."""

    def __init__(self, tok, vision):
        self.image_token = vision.image_token
        self.marker = vision.marker
        self.text = tok.without_specials({vision.image_token, vision.image_start, vision.image_end})

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids, offsets, at = [], [], 0
        marker = self.marker
        for k, piece in enumerate(text.split(marker)):
            if k:
                ids.append(self.image_token)
                offsets.append((at, at + len(marker)))
                at += len(marker)
            piece_ids, piece_offsets = self.text.encode_with_offsets(piece.replace(_escaped(marker), marker))
            ids += piece_ids
            offsets += [(a + at, b + at) for a, b in piece_offsets]
            at += len(piece)
        return ids, offsets

    def encode(self, text: str) -> list[int]:
        return self.encode_with_offsets(text)[0]


def expand(ids: list[int], boundary: int, images: list[Prepared], vision) -> tuple[list[int], int, tuple[media.Image, ...]]:
    """`ids` with the n-th `<image>` token replaced by `<img>`, the n-th image's placeholders and `</img>`; the message
    boundary moved past what was inserted before it (0 stays unknown); and the engine's images."""
    marks = [i for i, t in enumerate(ids) if t == vision.image_token]
    if len(marks) != len(images):
        raise InvalidRequest(f"the template placed {len(marks)} of the request's {len(images)} images")
    out: list[int] = []
    engine_images: list[media.Image] = []
    moved, last = boundary, 0
    for mark, image in zip(marks, images):
        out += ids[last:mark]
        out.append(vision.image_start)
        engine_images.append(media.Image(start=len(out), ids=media.placeholder_ids(image.digest, image.tokens), data=image.data,
                                         grid=image.grid))
        out += engine_images[-1].ids
        out.append(vision.image_end)
        last = mark + 1
        if boundary and mark < boundary:
            moved += image.tokens + 1
    out += ids[last:]
    return out, moved, tuple(engine_images)
