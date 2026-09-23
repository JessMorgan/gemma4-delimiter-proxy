"""Tests for the full-proxy passthrough behavior (no real network)."""

import json

from gemma4_delimiter_proxy import proxy
from tests.fakes import FakeAsyncClient, FakeUpstreamResponse, SimpleRequest

HOP_BY_HOP = {
    "host",
    "connection",
    "content-length",
    "transfer-encoding",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "upgrade",
}


def _assert_no_hop_by_hop(headers: dict[str, str]) -> None:
    assert not (set(headers) & HOP_BY_HOP)


async def test_healthz_proxied(monkeypatch):
    upstream = FakeUpstreamResponse(b"OK", status_code=200, headers={"content-type": "text/plain"})
    fake_client = FakeAsyncClient(upstream)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: fake_client)

    response = await proxy.proxy_passthrough(SimpleRequest(method="GET", path="/healthz"))
    assert response.status_code == 200
    assert response.body == b"OK"
    method, url, content, headers = fake_client.sent[0]
    assert method == "GET"
    assert url == proxy.UPSTREAM_URL + "/healthz"
    assert content in (None, b"")
    _assert_no_hop_by_hop(headers)
    assert upstream.aclose_called == 1
    assert fake_client.aclose_called == 1


async def _get_proxied(path: str, body: bytes, monkeypatch):
    upstream = FakeUpstreamResponse(body, status_code=200, headers={"content-type": "text/plain"})
    fake_client = FakeAsyncClient(upstream)

    def make_client(**kwargs):
        return fake_client

    monkeypatch.setattr(proxy.httpx, "AsyncClient", make_client)

    response = await proxy.proxy_passthrough(SimpleRequest(method="GET", path=path))
    assert response.status_code == 200
    assert response.body == body
    method, url, _, _ = fake_client.sent[0]
    assert method == "GET"
    assert url == proxy.UPSTREAM_URL + path


async def test_ready_and_metrics_proxied(monkeypatch):
    await _get_proxied("/ready", b"", monkeypatch)
    await _get_proxied("/metrics", b"vllm:num_requests 0\n", monkeypatch)


async def test_models_get_passthrough(monkeypatch):
    payload = b'{"object": "list", "data": [{"id": "gemma-4"}]}'
    upstream = FakeUpstreamResponse(
        payload, status_code=200, headers={"content-type": "application/json"}
    )
    fake_client = FakeAsyncClient(upstream)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: fake_client)

    response = await proxy.proxy_passthrough(
        SimpleRequest(method="GET", path="/v1/models", query="limit=5")
    )
    assert response.status_code == 200
    assert json.loads(response.body) == {"object": "list", "data": [{"id": "gemma-4"}]}
    method, url, content, headers = fake_client.sent[0]
    assert method == "GET"
    assert url == proxy.UPSTREAM_URL + "/v1/models?limit=5"
    assert content in (None, b"")
    _assert_no_hop_by_hop(headers)


async def test_non_json_body_forwarded_unchanged(monkeypatch):
    raw = b"--boundary\r\nContent-Disposition: form-data; name=\"file\"\r\n\r\nbinary\xff\xfe\r\n--boundary--"
    upstream = FakeUpstreamResponse(
        b'{"id": "file-123"}', status_code=200, headers={"content-type": "application/json"}
    )
    fake_client = FakeAsyncClient(upstream)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: fake_client)

    response = await proxy.proxy_passthrough(
        SimpleRequest(
            method="POST",
            path="/v1/files",
            body=raw,
            headers={"content-type": "multipart/form-data; boundary=boundary"},
        )
    )
    assert response.status_code == 200
    method, url, content, headers = fake_client.sent[0]
    assert method == "POST"
    assert url == proxy.UPSTREAM_URL + "/v1/files"
    assert content == raw  # Raw bytes forwarded unchanged
    assert headers["content-type"] == "multipart/form-data; boundary=boundary"


async def test_hop_by_hop_headers_stripped_both_ways(monkeypatch):
    upstream = FakeUpstreamResponse(
        b"{}",
        status_code=200,
        headers={
            "content-type": "application/json",
            "connection": "keep-alive",
            "transfer-encoding": "chunked",
            "content-length": "999",  # Deliberately wrong: must not leak through
            "x-request-id": "abc123",
        },
    )
    fake_client = FakeAsyncClient(upstream)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: fake_client)

    response = await proxy.proxy_passthrough(
        SimpleRequest(
            method="POST",
            path="/v1/moderations",
            body=b'{"input": "hi"}',
            headers={
                "authorization": "Bearer token",
                "connection": "keep-alive",
                "host": "example.com",
                "x-custom": "yes",
            },
        )
    )
    # Request side: hop-by-hop stripped, others kept
    _, _, _, req_headers = fake_client.sent[0]
    _assert_no_hop_by_hop(req_headers)
    assert req_headers["authorization"] == "Bearer token"
    assert req_headers["x-custom"] == "yes"
    # Response side: upstream hop-by-hop stripped, others kept
    assert "connection" not in response.headers
    assert "transfer-encoding" not in response.headers
    # The framework recomputes content-length for the body it serves
    assert response.headers["content-length"] == str(len(response.body))
    assert response.headers["x-request-id"] == "abc123"
    assert response.headers["content-type"] == "application/json"


async def test_sse_response_streamed_with_cleanup(monkeypatch):
    sse = b"data: {\"choices\": []}\n\ndata: [DONE]\n\n"
    upstream = FakeUpstreamResponse(sse, headers={"content-type": "text/event-stream"})
    fake_client = FakeAsyncClient(upstream)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: fake_client)

    response = await proxy.proxy_passthrough(
        SimpleRequest(
            method="POST",
            path="/v1/completions",
            body=json.dumps({"stream": True}).encode(),
        )
    )
    assert response.headers["content-type"] == "text/event-stream"
    chunks = [chunk async for chunk in response.body_iterator]
    assert b"".join(chunks) == sse  # Raw byte passthrough
    assert upstream.aclose_called == 1
    assert fake_client.aclose_called == 1


async def test_upstream_status_code_forwarded(monkeypatch):
    upstream = FakeUpstreamResponse(b'{"error": "not found"}', status_code=404)
    fake_client = FakeAsyncClient(upstream)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: fake_client)

    response = await proxy.proxy_passthrough(SimpleRequest(method="GET", path="/v1/models/missing"))
    assert response.status_code == 404
    assert json.loads(response.body) == {"error": "not found"}
