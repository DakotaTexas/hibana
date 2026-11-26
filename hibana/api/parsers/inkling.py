"""Split Inkling 2 output into reasoning, content and tool calls, incrementally.

The model frames its own turn with tokens, which reach this parser as text (the chat endpoints decode Inkling output
with its special tokens kept):

    <|message_model|><|content_thinking|>reasoning<|end_message|>
    <|message_model|><|content_text|>content<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>

A turn holds any number of such messages. Thinking goes out as reasoning and text as content; a tool call goes into
the content as a `<tool_call>` block holding {"name", "arguments"} JSON, where `parsers/tools.parse_tool_calls` finds
it. The interface is `parsers/reasoning.ReasoningParser`'s.
"""

from __future__ import annotations

import json

MESSAGE = "<|message_model|>"
END = "<|end_message|>"
KINDS = {"<|content_thinking|>": "thinking", "<|content_text|>": "text", "<|content_tool_error|>": "text",
         "<|content_invoke_tool_text|>": "text", "<|content_invoke_tool_json|>": "tool"}
TOKENS = (MESSAGE, END, "<|content_model_end_sampling|>", *KINDS)


def tool_block(body: str) -> str:
    """The model's {"name", "args"} call as the `<tool_call>` block the tool parser reads; its text when it is not JSON."""
    try:
        call = json.loads(body)
        text = json.dumps({"name": call["name"], "arguments": call.get("args", {})}, ensure_ascii=False)
    except (ValueError, KeyError, TypeError):
        text = body
    return f"<tool_call>\n{text}\n</tool_call>"


class InklingParser:
    def __init__(self, in_reasoning: bool = False, tools: bool = False):
        self.reasoning = ""
        self.content = ""
        self._kind = "header"  # header (between <|message_model|> and its content token) | thinking | text | tool
        self._pending = ""  # a possible token start
        self._tool = ""

    def _emit(self, text: str) -> tuple[str, str]:
        if self._kind == "thinking":
            return text, ""
        if self._kind == "text":
            return "", text
        if self._kind == "tool":
            self._tool += text
        return "", ""  # a header (the tool's name, repeated in its call) is not output

    def _token(self, token: str) -> tuple[str, str]:
        out = ("", "")
        if self._kind == "tool":
            if self._tool.strip():
                out = ("", tool_block(self._tool))
            self._tool = ""
        self._kind = KINDS.get(token, "header")
        return out

    def feed(self, delta: str) -> tuple[str, str]:
        """Returns (reasoning_delta, content_delta) for this piece of text."""
        text = self._pending + delta
        self._pending = ""
        r_out, c_out = "", ""
        while text:
            at = text.find("<|")
            if at < 0:
                at = len(text) - 1 if text.endswith("<") else len(text)  # a lone "<" may start a token
            r, c = self._emit(text[:at])
            r_out, c_out, text = r_out + r, c_out + c, text[at:]
            if not text:
                break
            token = next((t for t in TOKENS if text.startswith(t)), None)
            if token is None:
                if any(t.startswith(text) for t in TOKENS):
                    self._pending = text  # an incomplete token: wait for the rest
                    break
                r, c = self._emit(text[:2])  # "<|" that is no token: output as it is
                r_out, c_out, text = r_out + r, c_out + c, text[2:]
                continue
            r, c = self._token(token)
            r_out, c_out, text = r_out + r, c_out + c, text[len(token):]
        self.reasoning += r_out
        self.content += c_out
        return r_out, c_out

    def finish(self) -> tuple[str, str]:
        """Flush at the end of the output: held text, and a tool call the output ended inside."""
        pending, self._pending = self._pending, ""
        r, c = self._emit(pending)
        if self._kind == "tool" and self._tool.strip():
            c += tool_block(self._tool)
            self._tool = ""
        self.reasoning += r
        self.content += c
        return r, c
