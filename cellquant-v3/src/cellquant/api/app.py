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
)

from cellquant.plugins.registry import PluginRegistry


def _register_default_plugins(registry: PluginRegistry) -> None:
    """Register the built-in FUSION plugins."""
    from cellquant.plugins.oxytrack import OxyTrackPlugin
    from cellquant.plugins.senescencedb import SenescenceDBPlugin

    registry.register(OxyTrackPlugin())
    registry.register(SenescenceDBPlugin())


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

    # FUSION plugins
    plugin_registry = PluginRegistry()
    _register_default_plugins(plugin_registry)
    plugin_registry.mount_all(app, prefix=f"{prefix}/plugins")

    # Plugin listing endpoint
    @app.get("/api/v1/plugins", tags=["plugins"])
    async def list_plugins():
        return [
            {
                "name": m.name,
                "slug": m.slug,
                "version": m.version,
                "description": m.description,
            }
            for m in plugin_registry.list_plugins()
        ]

    # Health check
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": __version__}

    # Serve pre-built frontend (must be LAST, after API routes)
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists() and any(static_dir.iterdir()):
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")

    return app
