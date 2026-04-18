"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from cellquant import __version__
from cellquant.api.dependencies import init_dependencies
from cellquant.api.middleware import setup_middleware
from cellquant.api.routers import (
    experiments,
    export,
    images,
    masks,
    napari,
    nellie,
    quantification,
    segmentation,
    tracking,
    training,
    ws,
)
from cellquant.api.routers import auth, admin, projects


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from cellquant.auth.database import init_db
    await init_db()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="CellQuant",
        version=__version__,
        description="High-throughput cell quantification and tracking for microscopy",
        lifespan=_lifespan,
    )

    # Initialize singletons
    init_dependencies()

    # Middleware
    setup_middleware(app)

    # API routers
    prefix = "/api/v1"
    app.include_router(auth.router, prefix=prefix)
    app.include_router(admin.router, prefix=prefix)
    app.include_router(projects.router, prefix=prefix)
    app.include_router(experiments.router, prefix=prefix)
    app.include_router(images.router, prefix=prefix)
    app.include_router(segmentation.router, prefix=prefix)
    app.include_router(tracking.router, prefix=prefix)
    app.include_router(masks.router, prefix=prefix)
    app.include_router(nellie.router, prefix=prefix)
    app.include_router(quantification.router, prefix=prefix)
    app.include_router(export.router, prefix=prefix)
    app.include_router(napari.router, prefix=prefix)
    app.include_router(ws.router, prefix=prefix)
    app.include_router(training.router, prefix=prefix)

    # Health check
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": __version__}

    # Serve pre-built frontend (SPA — must be LAST, after all API routes)
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists() and any(static_dir.iterdir()):
        from fastapi.responses import FileResponse
        from fastapi import Request as _Request

        # Mount static assets (JS/CSS/images) at /_app and other known prefixes
        app.mount("/_app", StaticFiles(directory=str(static_dir / "_app")), name="assets")

        # SPA catch-all: serve index.html for any unmatched path
        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str, request: _Request):
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(str(index))
            return {"detail": "Frontend not built"}

    return app
