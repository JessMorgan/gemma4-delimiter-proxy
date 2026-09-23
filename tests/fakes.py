"""Shared fakes for proxy tests: no real network involved."""

import httpx


class FakeUpstreamResponse:
    """httpx.Response stand-in: streaming chunks, aread(), records aclose() calls."""

    def __init__(self, chunks: bytes = b"", status_code: int = 200, headers=None):
        self._chunks = [chunks] if chunks else []
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self.aclose_called = 0

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self):
        return b"".join(self._chunks)

    def json(self):
        import json

        return json.loads(b"".join(self._chunks))

    async def aclose(self):
        self.aclose_called += 1


class FakeAsyncClient:
    """httpx.AsyncClient stand-in: records sent requests and aclose() calls."""

    def __init__(self, upstream_response: FakeUpstreamResponse, timeout=None):
        self.upstream_response = upstream_response
        self.timeout = timeout
        self.aclose_called = 0
        self.sent: list[tuple[str, str, bytes | None, dict[str, str]]] = []

    def build_request(self, method, url, content=None, headers=None, json=None):
        self.sent.append((method, url, content, dict(headers or {})))
        return object()

    async def send(self, request, stream=False):
        return self.upstream_response

    async def post(self, url, content=None, headers=None, json=None):
        self.sent.append(("POST", url, content, dict(headers or {})))
        return self.upstream_response

    async def aclose(self):
        self.aclose_called += 1


class SimpleURL:
    def __init__(self, path: str, query: str):
        self.path = path
        self.query = query


class SimpleRequest:
    """Minimal FastAPI Request stand-in: method, url, headers, body()."""

    def __init__(
        self,
        method: str = "POST",
        path: str = "/v1/chat/completions",
        query: str = "",
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ):
        self.method = method
        self.url = SimpleURL(path, query)
        self.headers = headers or {}
        self._body = body

    async def body(self):
        return self._body

    async def json(self):
        import json

        return json.loads(self._body)
