"""Console entry point: run the proxy with uvicorn.

Configuration via environment variables:
  GEMMA_PROXY_HOST      bind address (default 0.0.0.0)
  GEMMA_PROXY_PORT      bind port (default 4001)
  GEMMA_PROXY_LOG_LEVEL uvicorn log level (default info)
"""

import os
from dataclasses import dataclass

import uvicorn


@dataclass(frozen=True)
class ProxyConfig:
    host: str
    port: int
    log_level: str


def get_config() -> ProxyConfig:
    """Build the server config from environment variables."""
    host = os.environ.get("GEMMA_PROXY_HOST", "0.0.0.0")
    port_raw = os.environ.get("GEMMA_PROXY_PORT", "4001")
    try:
        port = int(port_raw)
    except ValueError:
        raise SystemExit(
            f"GEMMA_PROXY_PORT must be an integer, got {port_raw!r}"
        ) from None
    if not 1 <= port <= 65535:
        raise SystemExit(f"GEMMA_PROXY_PORT must be 1-65535, got {port}")
    log_level = os.environ.get("GEMMA_PROXY_LOG_LEVEL", "info")
    return ProxyConfig(host=host, port=port, log_level=log_level)


def main() -> None:
    from gemma4_delimiter_proxy.proxy import app

    config = get_config()
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()
