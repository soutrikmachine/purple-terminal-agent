FROM python:3.13-slim

# Install system deps: git (for build-time clone), and general tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# ── Install Python dependencies ─────────────────────────────
COPY pyproject.toml ./
# uv sync creates .venv and installs all deps from pyproject.toml
RUN uv sync --no-dev

# ── Build-time: construct task RAG index ────────────────────
# Clone the public terminal-bench-2 repo and extract task metadata.
# Uses python3 from the .venv uv just created.
# Non-fatal: if clone fails (network timeout, repo renamed), we write
# an empty index and the agent starts with RAG disabled — no crash.
COPY scripts/ ./scripts/
RUN mkdir -p /app/data && \
    uv run python3 scripts/build_task_index.py \
        --repo https://github.com/laude-institute/terminal-bench-2 \
        --output /app/data/task_index.json \
    || (echo "Warning: RAG index build failed — starting with empty index" && \
        echo '{"tasks": [], "count": 0}' > /app/data/task_index.json)

# ── Copy source ─────────────────────────────────────────────
COPY src/ ./src/

# ── Runtime config ──────────────────────────────────────────
EXPOSE 9009
ENV PYTHONUNBUFFERED=1
ENV TASK_INDEX_PATH=/app/data/task_index.json

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s \
    CMD curl -f http://localhost:9009/health || exit 1

CMD ["uv", "run", "--no-sync", "src/server.py", "--host", "0.0.0.0", "--port", "9009"]