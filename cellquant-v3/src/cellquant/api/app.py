"""FastAPI application factory."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from cellquant import __version__
from cellquant.api.middleware import setup_middleware
from cellquant.api.dependencies import init_dependencies

from cellquant.api.routers import (
    experiments,
    images,
    segmentation,
    tracking,
    masks,
    quantification,
    export,
    napari,
    ws,
    bharat,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="CellQuant",
        version=__version__,
        description="High-throughput cell quantification and tracking for microscopy",
    )

    # Initialize singletons
    init_dependencies()

    # Middleware
    setup_middleware(app)

    # API routers
    prefix = "/api/v1"
    app.include_router(experiments.router, prefix=prefix)
    app.include_router(images.router, prefix=prefix)
    app.include_router(segmentation.router, prefix=prefix)
    app.include_router(tracking.router, prefix=prefix)
    app.include_router(masks.router, prefix=prefix)
    app.include_router(quantification.router, prefix=prefix)
    app.include_router(export.router, prefix=prefix)
    app.include_router(napari.router, prefix=prefix)
    app.include_router(ws.router, prefix=prefix)
    app.include_router(bharat.router, prefix=prefix)

    # Health check
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": __version__}

    # Serve pre-built frontend (must be LAST, after API routes)
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists() and any(static_dir.iterdir()):
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")

    return app
