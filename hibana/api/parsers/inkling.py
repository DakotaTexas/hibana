"""Split Inkling 2 output into reasoning, content and tool calls, incrementally.

The model frames its own turn with tokens, which reach this parser as text (the chat endpoints decode Inkling output
with its special tokens kept):

    <|message_model|><|content_thinking|>reasoning<|end_message|>
    <|message_model|><|content_text|>content<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>

A turn holds any number of such messages. Thinking goes out as reasoning and text as content; a tool call goes into
the content as a `<tool_call>` block holding {"name", "arguments"} JSON, where `parsers/tools.parse_tool_calls` finds
it, and a call that is not one (cut off, or not JSON) stays in the content as the model wrote it. Another role's
message token ends the turn: what follows it is not the model's. The interface is
`parsers/reasoning.ReasoningParser`'s.
"""

from __future__ import annotations

import json

MESSAGE = "<|message_model|>"
END = "<|end_message|>"
KINDS = {"<|content_thinking|>": "thinking", "<|content_text|>": "text", "<|content_tool_error|>": "text",
         "<|content_invoke_tool_text|>": "text", "<|content_invoke_tool_json|>": "tool"}
OTHER_ROLES = ("<|message_user|>", "<|message_system|>", "<|message_tool|>", "<|endoftext|>")
TOKENS = (MESSAGE, END, "<|content_model_end_sampling|>", *KINDS, *OTHER_ROLES)


def tool_block(body: str, header: str = "") -> str:
    """The model's {"name", "args"} call as the `<tool_call>` block the tool parser reads (the message header names the
    tool when the body does not); the body as it is when it is not such a call."""
    try:
        call = json.loads(body)
        args = call.get("args", call.get("arguments", {}))
        if isinstance(args, str):
            args = json.loads(args)
        name = call.get("name") or header.strip()
        if not name or not isinstance(args, dict):
            return body
    except (ValueError, TypeError, AttributeError):
        return body
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n</tool_call>"


class InklingParser:
    def __init__(self, in_reasoning: bool = False, tools: bool = False):
        self.reasoning = ""
        self.content = ""
        self._kind = "header"  # header (between <|message_model|> and its content token) | thinking | text | tool | done
        self._pending = ""  # a possible token start
        self._header = ""  # the current message's header: a tool call's name
        self._tool = ""

    def _emit(self, text: str) -> tuple[str, str]:
        if self._kind == "thinking":
            return text, ""
        if self._kind == "text":
            return "", text
        if self._kind == "tool":
            self._tool += text
        elif self._kind == "header":
            self._header += text
        return "", ""  # a header (the tool's name, repeated in its call) and what follows the turn are not output

    def _flush_tool(self) -> str:
        block = tool_block(self._tool, self._header) if self._kind == "tool" and self._tool.strip() else ""
        self._tool = ""
        return block

    def _token(self, token: str) -> tuple[str, str]:
        if self._kind == "done":
            return "", ""
        out = ("", self._flush_tool())
        if token in OTHER_ROLES:
            self._kind = "done"
            return out
        if token == MESSAGE:
            self._header = ""
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
        c += self._flush_tool()
        self.reasoning += r
        self.content += c
        return r, c
