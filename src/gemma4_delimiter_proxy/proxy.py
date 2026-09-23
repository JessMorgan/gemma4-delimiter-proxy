"""FastAPI SSE proxy that transpiles Gemma 4 channel/tool-call delimiters
into OpenAI chat-completion format."""

import json
import os
import re
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
# Update this to your upstream vLLM or SGLang server endpoint
UPSTREAM_URL = os.environ.get(
    "GEMMA_UPSTREAM_URL", "http://localhost:8000/v1/chat/completions"
)

# Streaming channel state definitions
STATE_CONTENT = "content"
STATE_THINKING = "reasoning"
STATE_TOOL = "tool_calls"

# Delimiters we need to actively catch and isolate
TAG_THINK_START = "<|channel>thought"
TAG_THINK_END = "<|channel>"
TAG_TOOL_START = "<|tool_call>"
TAG_TOOL_END = "<tool_call|>"  # Handles both <tool_call|> and <|tool_call|>


def _is_inside_quotes(text: str, pos: int) -> bool:
    """Return True if position `pos` falls inside a double-quoted string."""
    in_string = False
    i = 0
    while i < pos:
        char = text[i]
        if char == "\\":
            i += 2  # Skip escaped character
            continue
        if char == '"':
            in_string = not in_string
        i += 1
    return in_string


def _quote_unquoted_keys(args_str: str) -> str:
    """Wrap unquoted JSON keys in quotes, ignoring text inside quoted strings."""

    def replace(match: re.Match) -> str:
        if _is_inside_quotes(args_str, match.start(2)):
            return match.group(0)
        return f'{match.group(1)}"{match.group(2)}":'

    return re.sub(r'([{,]\s*)([a-zA-Z0-9_-]+)\s*:', replace, args_str)


def parse_gemma_tool_syntax(accumulated_tool_text: str):
    """
    Transforms Gemma 4 raw text syntax:
    'call:get_weather{location:<|"|>Boston<|"|>}' or 'call:my_func{"arg1": 23}'
    Into standard OpenAI dictionary properties.
    """
    # Clean Gemma's weird string escape sequence if present: <|"|> -> "
    cleaned_text = accumulated_tool_text.replace('<|""|>', '"').replace('<|"|>', '"')

    func_name = "unknown_function"
    args_str = "{}"

    # Extract function name (looks for call:func_name followed by { )
    func_match = re.search(r"call:([a-zA-Z0-9_-]+)\{", cleaned_text)
    if func_match:
        func_name = func_match.group(1)
        # Extract everything from the first curly brace onward
        start_idx = cleaned_text.find("{")
        if start_idx != -1:
            args_str = cleaned_text[start_idx:]

    # Quick fix: Gemma sometimes emits unquoted JSON keys like {location: "Boston"}
    # This simple regex wraps unquoted keys in quotes so standard JSON parsers don't crash
    args_str = _quote_unquoted_keys(args_str)

    return func_name, args_str


async def transpile_stream_generator(upstream_response):
    byte_buffer = b""
    current_state = STATE_CONTENT

    # Stateful buffers to accumulate strings across chunks
    accumulated_tool_text = ""
    # Fresh unique ID per tool call, generated at each tool-start transition
    tool_call_id = ""

    async for chunk in upstream_response.aiter_bytes():
        byte_buffer += chunk

        while b"\n\n" in byte_buffer:
            line_bytes, byte_buffer = byte_buffer.split(b"\n\n", 1)
            line = line_bytes.decode("utf-8", errors="ignore").strip()

            if not line.startswith("data:"):
                if line:  # Keep-alive headers, comments, etc.
                    yield f"{line}\n\n".encode()
                continue

            if line == "data: [DONE]":
                yield b"data: [DONE]\n\n"
                continue

            try:
                payload = json.loads(line[5:])
                delta = payload["choices"][0]["delta"]

                # Intercept content leakage
                raw_text = delta.get("content", "")
                if not raw_text and "tool_calls" not in delta and "reasoning" not in delta:
                    yield f"{line}\n\n".encode()
                    continue

                # Wipe the messy original content key out so we can control routing completely
                delta.pop("content", None)

                # --- STEP 1: EVALUATE STATE TRANSITIONS & CLEAN TAGS ---

                # Check for Thinking State transitions
                if TAG_THINK_START in raw_text:
                    current_state = STATE_THINKING
                    raw_text = raw_text.replace(TAG_THINK_START, "")

                # Check for Tool State transitions
                if TAG_TOOL_START in raw_text:
                    current_state = STATE_TOOL
                    raw_text = raw_text.replace(TAG_TOOL_START, "")
                    accumulated_tool_text = ""  # Reset tool text collector
                    tool_call_id = f"call_{uuid.uuid4().hex[:16]}"  # Fresh id per tool call

                # Check for state terminations
                is_ending_think = (TAG_THINK_END in raw_text and current_state == STATE_THINKING)
                is_ending_tool = (
                    (TAG_TOOL_END in raw_text or "<|tool_call|>" in raw_text)
                    and current_state == STATE_TOOL
                )

                if is_ending_think:
                    raw_text = raw_text.replace(TAG_THINK_END, "")
                if is_ending_tool:
                    raw_text = raw_text.replace(TAG_TOOL_END, "").replace("<|tool_call|>", "")

                # --- STEP 2: PACK PAYLOAD ACCORDING TO STATE ---

                if current_state == STATE_THINKING:
                    # Stream directly into standard Reasoning field
                    if raw_text:
                        delta["reasoning_content"] = raw_text  # OpenWebUI / Litellm standard
                        delta["reasoning"] = raw_text  # Compatibility fallback

                elif current_state == STATE_TOOL:
                    # Accumulate raw syntax text so we can stream parts or evaluate the final syntax cleanly
                    accumulated_tool_text += raw_text

                    # Parse current progress
                    func_name, current_args = parse_gemma_tool_syntax(accumulated_tool_text)

                    # Construct compliant OpenAI Tool Call array structure
                    delta["tool_calls"] = [
                        {
                            "index": 0,
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": current_args,
                            },
                        }
                    ]

                else:
                    # Standard Content State
                    if raw_text:
                        delta["content"] = raw_text

                # Post-processing: If we intercepted an end-tag, return engine state back to standard content
                if is_ending_think or is_ending_tool:
                    current_state = STATE_CONTENT

                # Build mutated payload packet back to server format
                yield f"data: {json.dumps(payload)}\n\n".encode()

            except Exception:  # noqa: BLE001 - intentional: forward raw line on any parse break
                # Fallback to forwarding raw line safely if parsing breaks mid-stream packet
                yield f"{line}\n\n".encode()


@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    body = await request.json()
    headers = dict(request.headers)
    headers.pop("host", None)

    # Shorten timeouts to prevent client dropping connection during reasoning delays
    timeout = httpx.Timeout(60.0, connect=10.0)
    client = httpx.AsyncClient(timeout=timeout)

    if body.get("stream", False):
        req = client.build_request("POST", UPSTREAM_URL, json=body, headers=headers)
        r = await client.send(req, stream=True)

        async def stream_with_cleanup():
            try:
                async for chunk in transpile_stream_generator(r):
                    yield chunk
            finally:
                # Release the upstream stream and the client once the response
                # is fully consumed, cancelled, or the client disconnects
                await r.aclose()
                await client.aclose()

        return StreamingResponse(stream_with_cleanup(), media_type="text/event-stream")
    else:
        # Non-streaming completions don't suffer chunk parsing errors
        try:
            r = await client.post(UPSTREAM_URL, json=body, headers=headers)
            return r.json()
        finally:
            await client.aclose()
