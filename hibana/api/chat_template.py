"""Render the snapshot's chat template exactly as the reference server does."""

from __future__ import annotations

import json
from pathlib import Path

import jinja2
from jinja2.ext import Extension


def _raise_exception(message: str):
    raise jinja2.exceptions.TemplateError(message)


class Generation(Extension):
    """transformers' `{% generation %}...{% endgeneration %}`, which marks the assistant's text for training masks:
    rendered as its body."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.Scope(body, lineno=lineno)


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    """transformers' `tojson`: plain `json.dumps` (no HTML escapes), with its keyword arguments."""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class ChatTemplate:
    def __init__(self, snap: Path, defaults: tuple[tuple[str, object], ...] = (), chat_format: str = "qwen"):
        """`chat_format` (`Target.chat_format`): "inkling" renders with transformers' template dialect (its `tojson`
        arguments and `generation` blocks) and turns `enable_thinking` into the template's `reasoning_effort`."""
        self.defaults = dict(defaults)  # the target's template variables; a render's own kwargs win
        self.chat_format = chat_format
        inkling = chat_format == "inkling"
        env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            extensions=["jinja2.ext.loopcontrols", *([Generation] if inkling else [])],
            undefined=jinja2.Undefined,
        )
        env.globals["raise_exception"] = _raise_exception
        env.filters["tojson"] = _tojson if inkling else jinja2.filters.do_tojson if hasattr(jinja2.filters, "do_tojson") else None
        self._template = env.from_string((snap / "chat_template.jinja").read_text())

    def render(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        tools: list | None = None,
        **kwargs,
    ) -> str:
        if self.chat_format == "inkling" and not enable_thinking:
            kwargs.setdefault("reasoning_effort", "none")  # thinking off: effort 0
        return self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=tools,
            **{**self.defaults, **kwargs},
        )
