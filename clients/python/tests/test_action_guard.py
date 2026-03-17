import pytest

from text_generation import Client, AsyncClient
from text_generation.errors import ValidationError
from text_generation.types import Message, ActionGuardDecision, ToolCall


# Shared payload and helpers to avoid repetition across tests
_COMMON_PAYLOAD = {
    "id": "1",
    "object": "chat.completion",
    "created": 1,
    "model": "tgi",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "here",
                "tool_calls": [
                    {"id": 1, "type": "http", "function": {"name": "fetch"}}
                ],
            },
            "logprobs": None,
            "finish_reason": None,
        }
    ],
    "usage": {},
}


class _FakeResp:
    status_code = 200

    def __init__(self, payload=None):
        self._payload = payload or _COMMON_PAYLOAD

    def json(self):
        return self._payload


def _fake_post(url, json, headers=None, cookies=None, timeout=None, **kwargs):
    return _FakeResp()


# -------- Tests for blocking guards --------


def _blocking_guard(tc: ToolCall):
    return ActionGuardDecision.BLOCK


def test_action_guard_blocks_tool_call_sync(monkeypatch, fake_url, hf_headers):
    client = Client(fake_url, hf_headers)
    monkeypatch.setattr("requests.post", _fake_post)

    result = client.chat(
        messages=[Message(role="user", content="hello")],
        action_guard=_blocking_guard,
    )
    # blocked actions should be annotated with is_safe=False instead of raising
    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls[0].get("is_safe") is False or getattr(tool_calls[0], "is_safe", None) is False


def test_action_guard_blocks_tool_call_in_stream_sync(
    monkeypatch, fake_url, hf_headers
):
    client = Client(fake_url, hf_headers)

    # Monkeypatch the client's stream response generator to simulate a pending ToolCall
    def _fake_stream(self, request, action_guard=None):
        # yield a payload that contains a pending tool call
        yield _COMMON_PAYLOAD

    monkeypatch.setattr(Client, "_chat_stream_response", _fake_stream)

    gen = client.chat(
        messages=[Message(role="user", content="hello")],
        stream=True,
        action_guard=_blocking_guard,
    )
    first = next(gen)
    # ensure the tool call was annotated as unsafe
    tool_calls = first["choices"][0]["message"]["tool_calls"]
    assert tool_calls[0].get("is_safe") is False or getattr(tool_calls[0], "is_safe", None) is False


@pytest.mark.asyncio
async def test_action_guard_blocks_tool_call_async(monkeypatch, fake_url, hf_headers):
    client = AsyncClient(fake_url, hf_headers)
    payload = _COMMON_PAYLOAD
    async def _fake_single(self, request, action_guard=None):
        if action_guard is not None:
            for choice in payload["choices"]:
                msg = choice.get("message", {})
                tool_calls = msg.get("tool_calls", None)
                if tool_calls:
                    for i, raw in enumerate(tool_calls):
                        tc = ToolCall(**raw)
                        if action_guard(tc) == ActionGuardDecision.BLOCK:
                            # annotate instead of raising
                            payload["choices"][0]["message"]["tool_calls"][i]["is_safe"] = False
        from text_generation.types import ChatComplete

        return ChatComplete(**payload)

    monkeypatch.setattr(AsyncClient, "_chat_single_response", _fake_single)

    result = await client.chat(
        messages=[Message(role="user", content="hello")],
        action_guard=_blocking_guard,
    )
    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls[0].get("is_safe") is False or getattr(tool_calls[0], "is_safe", None) is False


@pytest.mark.asyncio
async def test_action_guard_blocks_tool_call_in_stream_async(
    monkeypatch, fake_url, hf_headers
):
    client = AsyncClient(fake_url, hf_headers)

    async def _fake_stream(self, request, action_guard=None):
        # yield a payload that contains a pending tool call
        yield _COMMON_PAYLOAD

    monkeypatch.setattr(AsyncClient, "_chat_stream_response", _fake_stream)

    gen = await client.chat(
        messages=[Message(role="user", content="hello")],
        stream=True,
        action_guard=_blocking_guard,
    )
    first = await gen.__anext__()
    tool_calls = first["choices"][0]["message"]["tool_calls"]
    assert tool_calls[0].get("is_safe") is False or getattr(tool_calls[0], "is_safe", None) is False


# -------- Tests for allowing guards --------


def _allowing_guard(tc: ToolCall):
    return ActionGuardDecision.ALLOW


def test_action_guard_allows_tool_call_sync(monkeypatch, fake_url, hf_headers):
    client = Client(fake_url, hf_headers)
    monkeypatch.setattr("requests.post", _fake_post)

    result = client.chat(
        messages=[Message(role="user", content="hello")],
        action_guard=_allowing_guard,
    )
    assert result.choices[0].message.content == "here"


def test_action_guard_allows_tool_call_in_stream_sync(
    monkeypatch, fake_url, hf_headers
):
    client = Client(fake_url, hf_headers)
    # Monkeypatch the client's stream response generator to simulate a pending ToolCall
    def _fake_stream(self, request, action_guard=None):
        tc = ToolCall(id=1, type="http", function={"name": "fetch"})
        if action_guard is not None and action_guard(tc) == ActionGuardDecision.BLOCK:
            raise ValidationError("Tool call blocked by action_guard in stream")
        yield _COMMON_PAYLOAD

    monkeypatch.setattr(Client, "_chat_stream_response", _fake_stream)

    gen = client.chat(
        messages=[Message(role="user", content="hello")],
        stream=True,
        action_guard=_allowing_guard,
    )
    assert next(gen)


@pytest.mark.asyncio
async def test_action_guard_allows_tool_call_async(monkeypatch, fake_url, hf_headers):
    client = AsyncClient(fake_url, hf_headers)
    payload = _COMMON_PAYLOAD
    async def _fake_single(self, request, action_guard=None):
        if action_guard is not None:
            for choice in payload["choices"]:
                msg = choice.get("message", {})
                tool_calls = msg.get("tool_calls", None)
                if tool_calls:
                    for raw in tool_calls:
                        tc = ToolCall(**raw)
                        if action_guard(tc) == ActionGuardDecision.BLOCK:
                            raise ValidationError(
                                "Tool call blocked by action_guard in response"
                            )
        from text_generation.types import ChatComplete

        return ChatComplete(**payload)

    monkeypatch.setattr(AsyncClient, "_chat_single_response", _fake_single)

    result = await client.chat(
        messages=[Message(role="user", content="hello")],
        action_guard=_allowing_guard,
    )
    assert result.choices[0].message.content == "here"


@pytest.mark.asyncio
async def test_action_guard_allows_tool_call_in_stream_async(
    monkeypatch, fake_url, hf_headers
):
    client = AsyncClient(fake_url, hf_headers)

    async def _fake_stream(self, request, action_guard=None):
        tc = ToolCall(id=1, type="http", function={"name": "fetch"})
        if action_guard is not None and action_guard(tc) == ActionGuardDecision.BLOCK:
            raise ValidationError("Tool call blocked by action_guard in stream")
        yield _COMMON_PAYLOAD

    monkeypatch.setattr(AsyncClient, "_chat_stream_response", _fake_stream)

    gen = await client.chat(
        messages=[Message(role="user", content="hello")],
        stream=True,
        action_guard=_allowing_guard,
    )
    first = await gen.__anext__()
    assert first
