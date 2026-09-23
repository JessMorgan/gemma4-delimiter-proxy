"""Tests for the streaming transpile generator using a fake async response."""

import json

import pytest

from gemma4_delimiter_proxy import proxy
from gemma4_delimiter_proxy.proxy import transpile_stream_generator
from tests.fakes import FakeAsyncClient, FakeUpstreamResponse, SimpleRequest


class FakeAsyncResponse:
    """Mimics an httpx streaming response: yields SSE byte chunks via aiter_bytes."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


def _sse(content: str) -> bytes:
    payload = {"choices": [{"delta": {"content": content}}]}
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _collect(chunks: list[bytes]) -> list[bytes]:
    return [chunk async for chunk in transpile_stream_generator(FakeAsyncResponse(chunks))]


async def test_content_passes_through():
    chunks = [_sse("Hello "), _sse("world")]
    out = await _collect(chunks)
    assert len(out) == 2
    for chunk, expected in zip(out, ["Hello ", "world"]):
        payload = json.loads(chunk.decode().removeprefix("data: ").strip())
        delta = payload["choices"][0]["delta"]
        assert delta["content"] == expected


async def test_thinking_tags_routed_to_reasoning():
    chunks = [_sse("<|channel>thought"), _sse("Let me think..."), _sse("<|channel>")]
    out = await _collect(chunks)
    assert len(out) == 3
    for chunk in out:
        payload = json.loads(chunk.decode().removeprefix("data: ").strip())
        delta = payload["choices"][0]["delta"]
        assert "content" not in delta
    # First chunk: tag stripped, no reasoning text yet
    first = json.loads(out[0].decode().removeprefix("data: ").strip())
    assert "reasoning_content" not in first["choices"][0]["delta"]
    # Middle chunk: reasoning routed
    middle = json.loads(out[1].decode().removeprefix("data: ").strip())
    delta = middle["choices"][0]["delta"]
    assert delta["reasoning_content"] == "Let me think..."
    assert delta["reasoning"] == "Let me think..."


async def test_tool_call_tags_produce_tool_calls():
    chunks = [
        _sse("<|tool_call>"),
        _sse('call:get_weather{location: <|"|>Boston<|"|>}'),
        _sse("<tool_call|>"),
    ]
    out = await _collect(chunks)
    assert len(out) == 3
    # First chunk: tool start tag stripped; empty accumulation falls back to
    # unknown_function/{} (matches original proxy behavior)
    first = json.loads(out[0].decode().removeprefix("data: ").strip())
    first_tc = first["choices"][0]["delta"]["tool_calls"][0]
    assert first_tc["function"]["name"] == "unknown_function"
    assert first_tc["function"]["arguments"] == "{}"
    # Second chunk: parsed function name + arguments
    second = json.loads(out[1].decode().removeprefix("data: ").strip())
    delta = second["choices"][0]["delta"]
    assert "content" not in delta
    assert len(delta["tool_calls"]) == 1
    tc = delta["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"location": "Boston"}
    assert tc["id"].startswith("call_")
    # Third chunk: end tag stripped; accumulated text is retained, so the
    # final tool_calls frame carries the complete parsed arguments
    third = json.loads(out[2].decode().removeprefix("data: ").strip())
    third_tc = third["choices"][0]["delta"]["tool_calls"][0]
    assert third_tc["function"]["name"] == "get_weather"
    assert json.loads(third_tc["function"]["arguments"]) == {"location": "Boston"}


async def test_two_tool_calls_get_distinct_ids():
    chunks = [
        _sse("<|tool_call>"),
        _sse('call:get_weather{location: <|"|>Boston<|"|>}'),
        _sse("<tool_call|>"),
        _sse("<|tool_call>"),
        _sse('call:search{query: <|"|>hi<|"|>}'),
        _sse("<tool_call|>"),
    ]
    out = await _collect(chunks)
    ids = set()
    for chunk in out:
        payload = json.loads(chunk.decode().removeprefix("data: ").strip())
        delta = payload["choices"][0]["delta"]
        if "tool_calls" in delta:
            ids.add(delta["tool_calls"][0]["id"])
    assert len(ids) == 2
    for tc_id in ids:
        assert tc_id.startswith("call_")


async def test_done_is_forwarded():
    chunks = [_sse("hi"), b"data: [DONE]\n\n"]
    out = await _collect(chunks)
    assert out[-1] == b"data: [DONE]\n\n"


async def test_stream_closes_response_and_client(monkeypatch):
    upstream = FakeUpstreamResponse(b"data: {\"choices\": []}\n\n")
    fake_client = FakeAsyncClient(upstream)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: fake_client)

    body = json.dumps({"stream": True}).encode()
    response = await proxy.chat_completions_proxy(SimpleRequest(body=body))
    chunks = [chunk async for chunk in response.body_iterator]
    assert len(chunks) == 1
    assert upstream.aclose_called == 1
    assert fake_client.aclose_called == 1
    # Forwarded to the upstream base URL + verbatim path
    method, url, content, _headers = fake_client.sent[0]
    assert method == "POST"
    assert url == proxy.UPSTREAM_URL + "/v1/chat/completions"
    assert content == body


async def test_non_stream_closes_client(monkeypatch):
    upstream = FakeUpstreamResponse(b'{"choices": []}')
    fake_client = FakeAsyncClient(upstream)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: fake_client)

    body = json.dumps({"stream": False}).encode()
    response = await proxy.chat_completions_proxy(SimpleRequest(body=body))
    assert response.status_code == 200
    assert json.loads(response.body) == {"choices": []}
    assert upstream.aclose_called == 1
    assert fake_client.aclose_called == 1


async def test_non_stream_upstream_status_forwarded(monkeypatch):
    upstream = FakeUpstreamResponse(
        b'{"error": "model not found"}', status_code=404,
        headers={"content-type": "application/json"},
    )
    fake_client = FakeAsyncClient(upstream)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: fake_client)

    body = json.dumps({"stream": False}).encode()
    response = await proxy.chat_completions_proxy(SimpleRequest(body=body))
    assert response.status_code == 404
    assert json.loads(response.body) == {"error": "model not found"}
    assert upstream.aclose_called == 1
    assert fake_client.aclose_called == 1


async def test_non_stream_post_failure_closes_client_and_propagates(monkeypatch):
    class FailingClient(FakeAsyncClient):
        async def post(self, url, content=None, headers=None, json=None):
            raise ConnectionError("upstream down")

    upstream = FakeUpstreamResponse(b"")
    fake_client = FailingClient(upstream)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: fake_client)

    body = json.dumps({"stream": False}).encode()
    with pytest.raises(ConnectionError, match="upstream down"):
        await proxy.chat_completions_proxy(SimpleRequest(body=body))
    assert fake_client.aclose_called == 1


async def test_chunks_split_across_buffers():
    # A single SSE frame split across two network chunks
    full = _sse("split")
    chunks = [full[:10], full[10:]]
    out = await _collect(chunks)
    assert len(out) == 1
    payload = json.loads(out[0].decode().removeprefix("data: ").strip())
    assert payload["choices"][0]["delta"]["content"] == "split"
