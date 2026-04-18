# ══════════════════════════════════════════════════════════════════
# Stage 1 — Build SvelteKit frontend
# ══════════════════════════════════════════════════════════════════
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ══════════════════════════════════════════════════════════════════
# Stage 2 — Production image
# ══════════════════════════════════════════════════════════════════
FROM python:3.12-slim

# System libraries required by scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security (with home directory for session cache)
RUN addgroup --system cellquant && \
    adduser --system --ingroup cellquant --home /home/cellquant cellquant && \
    mkdir -p /home/cellquant && \
    chown cellquant:cellquant /home/cellquant

WORKDIR /app

# Install base Python dependencies (no napari/display-server deps)
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
        "fastapi>=0.109" "uvicorn[standard]>=0.27" "pydantic[email]>=2.5" \
        "python-multipart>=0.0.6" "aiofiles>=23.2" \
        "aiosqlite>=0.20" "bcrypt>=4.0" "python-jose[cryptography]>=3.3" "httpx>=0.27" \
        "numpy>=1.24" "pandas>=2.1" "pyarrow>=14.0" \
        "tifffile>=2024.1" "scikit-image>=0.22" "scipy>=1.12" "Pillow>=10.2"

# Copy source + frontend (before install so static files are baked in)
COPY README.md ./
COPY src/ ./src/
COPY --from=frontend-builder /app/frontend/build /app/src/cellquant/static/

# Install package (static files are now included)
RUN pip install --no-cache-dir "hatchling>=1.21" && \
    pip install --no-cache-dir .

# Persistent data directory (SQLite DB + output files)
RUN mkdir -p /data && chown cellquant:cellquant /data
VOLUME ["/data"]

USER cellquant

EXPOSE 7860

# Secrets — override all of these via env / docker secrets
ENV CELLQUANT_DB=/data/cellquant.db \
    CELLQUANT_ADMIN_USER=admin \
    CELLQUANT_ADMIN_PASS=changeme \
    JWT_SECRET=change-this-in-production

CMD ["uvicorn", "cellquant.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "7860", \
     "--workers", "1", "--log-level", "info"]
