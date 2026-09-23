FROM python:3.12-slim

# Install uv in its own layer so it is cached across code changes
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy project metadata first and install dependencies: this layer is cached
# until pyproject.toml / uv.lock change, independent of source edits
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and install the project itself
COPY src ./src
RUN uv sync --frozen --no-dev

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 4001

# curl is not available in slim; use the stdlib instead
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, sys, urllib.request; port = os.environ.get('GEMMA_PROXY_PORT', '4001'); r = urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=4); sys.exit(0 if r.status == 200 else 1)"

CMD ["gemma4-delimiter-proxy"]
