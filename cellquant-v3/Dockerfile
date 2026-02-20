# ── Stage 1: Build frontend ──────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --ignore-scripts

COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ─────────────────────────────
FROM python:3.12-slim AS runtime

# Avoid interactive prompts and Python buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for scipy, scikit-image, opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (copy only pyproject.toml first for layer caching)
COPY pyproject.toml README.md ./
COPY src/cellquant/__init__.py ./src/cellquant/__init__.py
RUN pip install ".[gpu,tracking]" 2>/dev/null || \
    pip install "."

# Copy backend source
COPY src/ ./src/

# Re-install in editable mode now that full source is present
RUN pip install -e "." --no-deps

# Copy built frontend into static dir
COPY --from=frontend-build /app/frontend/build ./src/cellquant/static/

# Create non-root user
RUN groupadd --gid 1000 cellquant && \
    useradd --uid 1000 --gid cellquant --create-home cellquant && \
    mkdir -p /data /app/sessions && \
    chown -R cellquant:cellquant /app /data

USER cellquant

# Volumes for persistent data
VOLUME ["/data", "/app/sessions"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "-m", "cellquant", "serve", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
