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

The proxy listens on `0.0.0.0:4001`.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `GEMMA_UPSTREAM_URL` | `http://localhost:8000` | Upstream **base URL**; the request path + query string are appended verbatim (e.g. `/v1/models?limit=5` → `http://localhost:8000/v1/models?limit=5`) |

Example:

```bash
GEMMA_UPSTREAM_URL=http://gpu-host:8000 uv run gemma4-delimiter-proxy
```

## Development

```bash
uv sync
uv run ruff check .
uv run pytest
```
