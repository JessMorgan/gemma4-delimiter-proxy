# gemma4-delimiter-proxy

A FastAPI SSE proxy that transpiles Gemma 4 channel/tool-call delimiters into
standard OpenAI chat-completion format.

Gemma 4 streams thinking and tool calls as raw text delimiters
(`<|channel>thought`, `<|tool_call>`, `<tool_call|>`, etc.). This proxy
intercepts the upstream SSE stream and rewrites it so:

- Thinking text is routed to `reasoning_content` / `reasoning` delta fields.
- Tool-call text is accumulated, parsed (`call:func_name{...}` syntax,
  including `<|"|>` string escapes and unquoted JSON keys), and emitted as a
  proper `tool_calls` array with function name and JSON arguments.
- Plain content passes through unchanged, and `data: [DONE]` is forwarded.

## Running

```bash
uv sync
uv run gemma4-delimiter-proxy
```

The proxy listens on `0.0.0.0:4001` and exposes `POST /v1/chat/completions`
(streaming and non-streaming).

## Configuration

| Env var | Default | Description |
|---|---|---|
| `GEMMA_UPSTREAM_URL` | `http://localhost:8000/v1/chat/completions` | Upstream vLLM/SGLang endpoint to proxy to |

Example:

```bash
GEMMA_UPSTREAM_URL=http://gpu-host:8000/v1/chat/completions uv run gemma4-delimiter-proxy
```

## Development

```bash
uv sync
uv run ruff check .
uv run pytest
```
