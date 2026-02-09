# ── Stage 1: Build frontend ──────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python app ─────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# System deps for scipy, scikit-image, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[gpu,tracking]" 2>/dev/null || \
    pip install --no-cache-dir -e "."

# Copy backend source
COPY src/ ./src/

# Copy built frontend into static dir
COPY --from=frontend-build /app/frontend/build ./src/cellquant/static/

# Default port
EXPOSE 8000

# Run server
CMD ["python", "-m", "cellquant", "serve", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
