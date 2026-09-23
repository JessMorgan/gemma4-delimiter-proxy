# gemma4-delimiter-proxy

A full OpenAI-compatible API proxy for Gemma 4 upstreams (vLLM / SGLang).

It forwards **every** request path (any method) verbatim to the upstream
server — chat/completions, models, embeddings, files, fine-tuning,
moderations, responses, beta endpoints, and health/metrics — while applying
Gemma 4 channel/tool-call delimiter transpilation **only** to streaming
`POST /v1/chat/completions`.

## What the transpilation does

Gemma 4 streams thinking and tool calls as raw text delimiters
(`<|channel>thought`, `<|tool_call>`, `<tool_call|>`, etc.). For streaming
chat completions, the proxy intercepts the upstream SSE stream and rewrites
it so:

- Thinking text is routed to `reasoning_content` / `reasoning` delta fields.
- Tool-call text is accumulated, parsed (`call:func_name{...}` syntax,
  including `<|"|>` string escapes and unquoted JSON keys), and emitted as a
  proper `tool_calls` array with function name and JSON arguments.
- Plain content passes through unchanged, and `data: [DONE]` is forwarded.

All other routes are pure byte-level passthrough (method, path, query string,
raw body, and headers forwarded; hop-by-hop headers stripped both ways;
upstream status code and response headers preserved).

## Running

```bash
uv sync
uv run gemma4-delimiter-proxy
```

The proxy listens on `0.0.0.0:4001` by default; bind address, port, and log
level are configurable via environment variables (see [Configuration](#configuration)).

## Configuration

| Env var | Default | Description |
|---|---|---|
| `GEMMA_UPSTREAM_URL` | `http://localhost:8000` | Upstream **base URL**; the request path + query string are appended verbatim (e.g. `/v1/models?limit=5` → `http://localhost:8000/v1/models?limit=5`) |
| `GEMMA_PROXY_HOST` | `0.0.0.0` | Address the proxy binds to |
| `GEMMA_PROXY_PORT` | `4001` | Port the proxy listens on (must be an integer) |
| `GEMMA_PROXY_LOG_LEVEL` | `info` | Uvicorn log level (`debug`, `info`, `warning`, `error`, ...) |

Example:

```bash
GEMMA_UPSTREAM_URL=http://gpu-host:8000 GEMMA_PROXY_PORT=8080 uv run gemma4-delimiter-proxy
```

## Running with Docker

Build and run the image directly:

```bash
docker build -t gemma4-delimiter-proxy .
docker run --rm -p 4001:4001 \
  -e GEMMA_UPSTREAM_URL=http://host.docker.internal:8000 \
  --add-host host.docker.internal:host-gateway \
  gemma4-delimiter-proxy
```

Or use the provided compose file:

```bash
docker compose up --build
```

The compose service maps `4001:4001` and sets `GEMMA_UPSTREAM_URL` to
`http://host.docker.internal:8000` by default, with
`host.docker.internal:host-gateway` mapped so it works on Linux as well as
macOS/Windows. On Linux without the `host-gateway` mapping, point
`GEMMA_UPSTREAM_URL` at the host's LAN IP (e.g. `http://192.168.1.50:8000`).

The image runs as a non-root user, exposes port `4001`, and includes a
`HEALTHCHECK` that hits `/healthz`. Note that `/healthz` is proxied to the
upstream, so the healthcheck reflects **upstream availability** as well: if
the upstream is down, the container is reported unhealthy even though the
proxy itself is running.

### Docker image

A prebuilt image is published to GitHub Container Registry on every push to
`main` and on release tags:

```
ghcr.io/jessmorgan/gemma4-delimiter-proxy
```

Tag scheme:

- `latest` — built from the latest `main` commit (also updated by release tags)
- `vX.Y.Z` — built from the matching `vX.Y.Z` git tag

Pull and run:

```bash
docker pull ghcr.io/jessmorgan/gemma4-delimiter-proxy:latest
docker run --rm -p 4001:4001 \
  -e GEMMA_UPSTREAM_URL=http://host.docker.internal:8000 \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/jessmorgan/gemma4-delimiter-proxy:latest
```

## Development

```bash
uv sync
uv run ruff check .
uv run pytest
```
