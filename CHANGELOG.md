# Changelog

All notable changes to CellQuant are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2025

Complete rewrite as a full-stack web application.

### Added
- FastAPI backend with async endpoints and Pydantic validation
- SvelteKit + Svelte 5 frontend with TypeScript
- OpenLayers-based tiled image viewer (DZI format) for large microscopy images
- WebSocket real-time progress tracking for segmentation, tracking, and quantification
- Interactive mask editing (click-to-select, delete, merge)
- Trackastra cell tracking integration with lineage visualization
- Plotly.js charts (scatter, histogram, box plots) for quantification results
- RESTful API (`/api/v1/`) with 9 resource routers
- Session management for multi-user access
- CLI entry point (`cellquant serve`, `cellquant cleanup`)
- Multi-stage Docker build (Node.js frontend + Python runtime)
- Makefile for development workflow
- Data export to CSV and Excel

### Changed
- Migrated from Tkinter/Gradio to FastAPI + SvelteKit architecture
- Replaced matplotlib plots with interactive Plotly.js charts
- Image viewing now uses tiled Deep Zoom Images instead of loading full images into memory
- Segmentation parameters are now configured through a web form instead of CLI arguments

### Removed
- Tkinter desktop interface (preserved in `Cellquant_v1.py`)
- Gradio web interface (preserved in `Cellquant_v2.py`)

## [2.0.0] - 2025

Web-based interface with timelapse tracking support.

### Added
- Gradio web interface for browser-based access
- Timelapse cell tracking across multiple timepoints
- Lineage tree visualization
- Per-cell intensity profiles over time
- Seaborn/matplotlib statistical plots
- Backward compatibility with v1 functions

### Changed
- Moved from desktop-only to web-accessible interface
- Extended single-frame analysis to multi-frame timelapse support

## [1.0.0] - 2025

Initial release as a Tkinter desktop application.

### Added
- Cellpose-based cell segmentation (nucleus and cytoplasm)
- Configurable channel assignment (nuclear, cytoplasm, marker suffixes)
- CTCF (Corrected Total Cell Fluorescence) calculation
- Background estimation for fluorescence correction
- Segmentation overlay visualization
- CSV export of quantification results
- Support for multiple Cellpose models (cpsam, cyto3, nuclei)
- Adjustable segmentation parameters (diameter, flow threshold, min size)
