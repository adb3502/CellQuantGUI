# CellQuant

**High-throughput cell quantification and tracking for microscopy.**

CellQuant is a platform for automated cell segmentation, tracking, and fluorescence quantification of microscopy images. It supports GPU-accelerated [Cellpose](https://github.com/MouseLand/cellpose) segmentation, multi-frame cell tracking via [Trackastra](https://github.com/weigertlab/trackastra), and Corrected Total Cell Fluorescence (CTCF) analysis — all through a modern web interface.

---

## Table of Contents

- [Features](#features)
- [Versions](#versions)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Cell Segmentation** — Automatic nucleus and cytoplasm segmentation using Cellpose (GPU-accelerated)
- **Cell Tracking** — Multi-frame tracking with Trackastra for timelapse experiments
- **CTCF Quantification** — Corrected Total Cell Fluorescence calculation across multiple markers
- **Interactive Viewer** — Tiled image viewer (OpenLayers + DZI) with ROI editing
- **Batch Processing** — Process entire experiments with real-time progress via WebSocket
- **Data Export** — Export results to CSV, Excel, and other formats
- **Interactive Charts** — Scatter plots, histograms, and box plots via Plotly.js

---

## Versions

CellQuant has evolved through several major versions:

### v1 — Desktop Application (`Cellquant_v1.py`)

The original implementation as a standalone Tkinter desktop application.

- **UI**: Tkinter with `ttk` widgets
- **Segmentation**: Cellpose (CPU/GPU) with configurable models, diameter, and flow threshold
- **Analysis**: CTCF calculation with background estimation
- **Channels**: Configurable nuclear, cytoplasm, and marker channel suffixes
- **Output**: Segmentation overlays and CSV quantification results

### v2 — Web UI + Timelapse (`Cellquant_v2.py`)

Extended v1 with a Gradio-based web interface and timelapse tracking.

- **UI**: Gradio web application (port 7860)
- **Tracking**: Cell tracking across timepoints with lineage tree visualization
- **Analysis**: Per-cell intensity profiles over time
- **Compatibility**: Imports and extends all v1 functions
- **Visualization**: Matplotlib/Seaborn plots for tracking data

### v3 — Full-Stack Rebuild (`cellquant-v3/`)

**Current active version.** A complete rewrite with a modern full-stack architecture.

- **Backend**: FastAPI with async endpoints, Pydantic validation, and session management
- **Frontend**: SvelteKit 2 + Svelte 5, TailwindCSS 4, TypeScript
- **Viewer**: OpenLayers with Deep Zoom Image (DZI) tiling for large microscopy images
- **Charts**: Plotly.js (scatter, histogram, box plots)
- **Real-time**: WebSocket progress tracking for long-running operations
- **API**: RESTful API (`/api/v1/`) with 9 resource routers
- **Docker**: Multi-stage build with frontend compilation and Python runtime
- **CLI**: `cellquant serve` and `cellquant cleanup` commands

### Enterprise Edition (`cellquant_enterprise/`)

A variant with Gradio UI and Napari integration for advanced 3D visualization.

- **UI**: Gradio with dual themes (Dark "Deep Focus" / Light "Histology Atlas")
- **Pipeline**: `BatchPipeline` orchestrator for multi-condition experiments
- **Napari Bridge**: Direct integration with Napari for 3D visualization and annotation
- **Batch Processing**: Parallel marker quantification with progress tracking

---

## Quick Start

The fastest way to get running is with Docker:

```bash
cd cellquant-v3
docker compose up --build
```

Then open http://localhost:8000 in your browser.

---

## Installation

### Prerequisites

- Python 3.10+
- Node.js 20+ (for frontend development)
- CUDA-capable GPU (optional, for accelerated segmentation)

### From Source (v3)

```bash
# Clone the repository
git clone https://github.com/adb3502/CellQuantGUI.git
cd CellQuantGUI/cellquant-v3

# Install Python dependencies
pip install -e ".[all,dev]"

# Install frontend dependencies
cd frontend && npm ci && cd ..
```

### Minimal Install (CPU only)

```bash
cd cellquant-v3
pip install -e "."
```

### GPU + Tracking

```bash
cd cellquant-v3
pip install -e ".[gpu,tracking]"
```

### Legacy Versions (v1/v2)

```bash
pip install -r requirements.txt
```

---

## Usage

### Running v3 (Recommended)

**Start the server:**

```bash
cd cellquant-v3
cellquant serve --port 8000
```

Or using the Makefile:

```bash
cd cellquant-v3
make dev          # Frontend + backend with hot reload
make dev-backend  # Backend only (port 8000)
make dev-frontend # Frontend only (port 5173)
```

**CLI options:**

```
cellquant serve [OPTIONS]
  --port PORT        Server port (default: 8000)
  --host HOST        Bind address (default: 127.0.0.1)
  --no-browser       Don't open browser automatically
  --no-gpu           Disable GPU acceleration

cellquant cleanup [OPTIONS]
  --max-age HOURS    Max session age in hours (default: 24)
```

### Workflow

1. **Load Experiment** — Select a folder containing microscopy images. CellQuant detects conditions and channels automatically.
2. **Configure Channels** — Assign nuclear, cytoplasm, and marker channels.
3. **Segment** — Run Cellpose segmentation with adjustable parameters (model, diameter, flow threshold).
4. **Edit Masks** — Interactively select, delete, or merge segmentation masks.
5. **Track** (timelapse) — Track cells across timepoints using Trackastra.
6. **Quantify** — Calculate CTCF and other metrics across all markers.
7. **Export** — Download results as CSV or Excel.

### Running Legacy Versions

```bash
# v1 (Tkinter)
python Cellquant_v1.py

# v2 (Gradio)
python Cellquant_v2.py

# Enterprise (Gradio + Napari)
python cellquant_enterprise/run.py --port 7860
```

---

## Docker

### Using Docker Compose (Recommended)

```bash
cd cellquant-v3

# CPU only
docker compose up --build

# With GPU support
docker compose --profile gpu up --build
```

### Using Docker Directly

```bash
cd cellquant-v3

# Build
docker build -t cellquant:latest .

# Run (CPU)
docker run --rm -p 8000:8000 cellquant:latest

# Run (GPU)
docker run --rm -p 8000:8000 --gpus all cellquant:latest
```

### Persistent Data

Mount a volume for experiment data and session persistence:

```bash
docker run --rm -p 8000:8000 \
  -v /path/to/images:/data \
  -v cellquant-sessions:/app/sessions \
  cellquant:latest
```

---

## Project Structure

```
CellQuantGUI/
├── cellquant-v3/                  # Active version (FastAPI + Svelte)
│   ├── src/cellquant/
│   │   ├── api/                   # FastAPI application
│   │   │   ├── routers/           # API endpoints (9 routers)
│   │   │   ├── schemas/           # Pydantic request/response models
│   │   │   ├── app.py             # Application factory
│   │   │   └── middleware.py      # CORS, error handling
│   │   ├── core/                  # Core processing logic
│   │   ├── config/                # Application configuration
│   │   ├── sessions/              # Session management
│   │   ├── tasks/                 # Background task processing
│   │   ├── tiles/                 # DZI tile generation
│   │   ├── tracking/              # Trackastra integration
│   │   └── cli.py                 # CLI entry point
│   ├── frontend/                  # SvelteKit frontend
│   │   └── src/
│   │       ├── lib/               # Shared code (API client, stores, components)
│   │       └── routes/            # Pages (experiment, segmentation, editor, ...)
│   ├── tests/                     # Test suite
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── Makefile
│   └── pyproject.toml
├── cellquant_enterprise/          # Enterprise variant (Gradio + Napari)
│   ├── core/pipeline.py           # Batch pipeline orchestrator
│   ├── ui/app.py                  # Gradio interface
│   └── run.py                     # Entry point
├── Cellquant_v1.py                # Original Tkinter version
├── Cellquant_v2.py                # Gradio + timelapse version
├── requirements.txt               # Legacy dependencies
├── CHANGELOG.md
└── README.md                      # This file
```

---

## API Reference (v3)

The v3 backend exposes a RESTful API at `/api/v1/`:

| Endpoint               | Description                              |
| ---------------------- | ---------------------------------------- |
| `GET /api/health`      | Health check                             |
| `/api/v1/experiments`  | Experiment loading, scanning, channel config |
| `/api/v1/images`       | Image loading, metadata, thumbnails      |
| `/api/v1/segmentation` | Run Cellpose segmentation                |
| `/api/v1/tracking`     | Trackastra cell tracking                 |
| `/api/v1/masks`        | ROI editing (select, delete, merge)      |
| `/api/v1/quantification` | CTCF calculation and statistics        |
| `/api/v1/export`       | Download results (CSV, Excel)            |
| `/api/v1/napari`       | Napari viewer integration                |
| `/api/v1/ws`           | WebSocket for real-time progress         |

---

## Contributing

Contributions are welcome! Here's how to get started:

### Development Setup

```bash
# Clone and create a branch
git clone https://github.com/adb3502/CellQuantGUI.git
cd CellQuantGUI/cellquant-v3

# Install all dependencies (including dev tools)
pip install -e ".[all,dev]"
cd frontend && npm ci && cd ..

# Start development servers
make dev
```

### Running Tests

```bash
make test          # All tests
make test-core     # Core module tests
make test-api      # API tests
```

### Guidelines

1. **Branch** from `main` and open a pull request when ready.
2. **Test** your changes — add tests for new features or bug fixes.
3. **Follow existing style** — the project uses standard Python conventions and Svelte/TypeScript for the frontend.
4. **Keep commits focused** — one logical change per commit with a descriptive message.
5. **Document** public API changes in the CHANGELOG.

### Building for Production

```bash
make build         # Build frontend + Python wheel
make docker        # Build Docker image
```

---

## License

This project is licensed under the MIT License. See the [pyproject.toml](cellquant-v3/pyproject.toml) for details.
