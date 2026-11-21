import json

import pytest

from hibana.api.parsers.reasoning import ReasoningParser
from hibana.api.parsers.tools import parse_tool_calls


def test_reasoning_split_streaming_with_split_tag():
    p = ReasoningParser(in_reasoning=True)
    out = [p.feed(x) for x in ["I think", " so.</thi", "nk>\n\nThe answer", " is 4"]]
    assert p.reasoning == "I think so." and p.content == "\n\nThe answer is 4"
    assert out[0] == ("I think", "") and out[3] == ("", " is 4")


def test_reasoning_disabled_is_plain_content():
    p = ReasoningParser(in_reasoning=False)
    p.feed("Hello <think>not a tag here")  # tag not at start: content
    assert p.content.startswith("Hello") and p.reasoning == ""


def test_reasoning_explicit_start_tag():
    p = ReasoningParser(in_reasoning=False)
    p.feed("<think>hmm</think>ok")
    assert p.reasoning == "hmm" and p.content == "ok"


def test_tool_call_parse_with_schema_casts():
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]
    text = 'Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>'
    content, calls = parse_tool_calls(text, tools)
    assert content == "Let me check."
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
    import json
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris", "days": 3}


def test_no_tool_call_passthrough():
    content, calls = parse_tool_calls("plain answer", None)
    assert content == "plain answer" and calls == []


CALL = "<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"


def feed_all(p, text, size):
    out = [p.feed(text[i:i + size]) for i in range(0, len(text), size)] + [p.finish()]
    return "".join(r for r, _ in out), "".join(c for _, c in out)


@pytest.mark.parametrize("size", [1, 3, 1000])
def test_tool_call_in_unclosed_reasoning_is_content(size):
    """The turn ends inside reasoning with a complete call; the call is content, the rest stays reasoning."""
    p = ReasoningParser(in_reasoning=True, tools=True)
    assert feed_all(p, "Check the files.\n\n" + CALL, size) == ("Check the files.\n\n", CALL)
    assert (p.reasoning, p.content) == ("Check the files.\n\n", CALL)


@pytest.mark.parametrize("size", [1, 3, 1000])
def test_tool_call_drafted_in_closed_reasoning_stays_reasoning(size):
    p = ReasoningParser(in_reasoning=True, tools=True)
    text = "Maybe " + CALL + " no.</think>\n\nDone."
    assert feed_all(p, text, size) == ("Maybe " + CALL + " no.", "\n\nDone.")


@pytest.mark.parametrize("tools,text", [
    (False, "Thinking " + CALL),                        # no tools offered: stays reasoning
    (True, "Thinking <tool_call>\n<function=Bash>"),    # cut off mid-call (max_tokens): still reasoning
])
def test_unclosed_reasoning_otherwise_stays_reasoning(tools, text):
    p = ReasoningParser(in_reasoning=True, tools=tools)
    assert feed_all(p, text, 2) == (text, "")


INKLING_TURN = ('<|message_model|><|content_thinking|>Need the tool.<|end_message|><|message_model|><|content_text|>Let me check.<|end_message|>'
                '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris","days":3}}<|end_message|>')


@pytest.mark.parametrize("step", [1, 5, 1000])
def test_inkling_turn_splits_at_any_chunking(step):
    """Inkling frames its turn with tokens: thinking is reasoning, text is content, a tool's name header is dropped
    and its {"name", "args"} call becomes a block the tool parser reads, however the text arrives."""
    from hibana.api.parsers.inkling import InklingParser
    p = InklingParser()
    deltas = [p.feed(INKLING_TURN[i:i + step]) for i in range(0, len(INKLING_TURN), step)] + [p.finish()]
    assert "".join(r for r, _ in deltas) == p.reasoning == "Need the tool."
    assert "".join(c for _, c in deltas) == p.content and p.content.startswith("Let me check.<tool_call>")
    content, calls = parse_tool_calls(p.content, None)
    assert content == "Let me check." and json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris", "days": 3}
    assert calls[0]["function"]["name"] == "get_weather"


def test_inkling_text_keeps_what_is_not_a_token():
    from hibana.api.parsers.inkling import InklingParser
    p = InklingParser()
    for piece in ("<|message_model|><|content_text|>a <", "| b <|x|> c<", "|end_message|>"):
        p.feed(piece)
    p.finish()
    assert p.content == "a <| b <|x|> c" and p.reasoning == ""
