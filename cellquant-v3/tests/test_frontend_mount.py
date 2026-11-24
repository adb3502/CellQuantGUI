"""The backend must start whether or not the frontend has been built.

The built frontend is not in git (the Docker build produces it), so a fresh
checkout has static/ holding only .gitkeep. create_app() mounted static/_app
whenever static/ was non-empty and crashed on that checkout.
"""
from fastapi import FastAPI


def _paths(app):
    return {getattr(r, "path", None) for r in app.routes}


def test_unbuilt_frontend_is_skipped(tmp_path):
    from cellquant.api.app import mount_frontend

    (tmp_path / ".gitkeep").write_text("")
    app = FastAPI()
    assert mount_frontend(app, tmp_path) is False
    assert "/_app" not in _paths(app)


def test_built_frontend_is_served(tmp_path):
    from cellquant.api.app import mount_frontend

    (tmp_path / "_app").mkdir()
    (tmp_path / "index.html").write_text("<html></html>")
    app = FastAPI()
    assert mount_frontend(app, tmp_path) is True
    assert "/_app" in _paths(app)
    assert "/{full_path:path}" in _paths(app)
