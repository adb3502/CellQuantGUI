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
    _install_picker_for_all_users()
    yield


def _install_picker_for_all_users():
    """Write a Startup-folder shortcut for every user profile on this machine.

    Safe to call repeatedly — skips profiles where the shortcut already exists.
    Only runs on Windows; silently does nothing on other platforms.
    """
    import platform
    if platform.system() != "Windows":
        return

    import os
    import threading

    def _do():
        try:
            import win32com.client  # pywin32
        except ImportError:
            # pywin32 not available — fall back to VBScript via PowerShell
            _install_via_powershell()
            return

        pythonw = Path(r"D:\Users\adb\dev\lab-tools\CellQuantGUI\.venv\Scripts\pythonw.exe")
        picker  = Path(r"D:\Users\adb\dev\lab-tools\CellQuantGUI\cellquant-v3\cellquant-picker.py")
        if not pythonw.exists() or not picker.exists():
            return

        users_root = Path(r"C:\Users")
        if not users_root.exists():
            users_root = Path(os.environ.get("SystemDrive", "C:") + r"\Users")

        shell = win32com.client.Dispatch("WScript.Shell")
        for profile in users_root.iterdir():
            startup = profile / r"AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup"
            try:
                if not startup.exists():
                    continue
            except PermissionError:
                continue
            shortcut_path = str(startup / "CellQuantPicker.lnk")
            if Path(shortcut_path).exists():
                continue
            try:
                sc = shell.CreateShortcut(shortcut_path)
                sc.TargetPath = str(pythonw)
                sc.Arguments = f'"{picker}" --backend http://localhost:7860'
                sc.WindowStyle = 7  # minimised / hidden
                sc.Save()
                print(f"[CellQuant] Picker shortcut installed for {profile.name}")
            except Exception as e:
                print(f"[CellQuant] Could not install picker for {profile.name}: {e}")

    threading.Thread(target=_do, daemon=True).start()


def _install_via_powershell():
    """Fallback: use PowerShell to create startup shortcuts without pywin32."""
    import subprocess
    import os

    pythonw = r"D:\Users\adb\dev\lab-tools\CellQuantGUI\.venv\Scripts\pythonw.exe"
    picker  = r"D:\Users\adb\dev\lab-tools\CellQuantGUI\cellquant-v3\cellquant-picker.py"
    users_root = Path(os.environ.get("SystemDrive", "C:") + r"\Users")

    for profile in users_root.iterdir():
        startup = profile / r"AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup"
        try:
            if not startup.exists():
                continue
        except PermissionError:
            continue
        shortcut_path = startup / "CellQuantPicker.lnk"
        if shortcut_path.exists():
            continue
        ps = (
            f'$ws = New-Object -ComObject WScript.Shell; '
            f'$s = $ws.CreateShortcut("{shortcut_path}"); '
            f'$s.TargetPath = "{pythonw}"; '
            f'$s.Arguments = \'"{picker}" --backend http://localhost:7860\'; '
            f'$s.WindowStyle = 7; '
            f'$s.Save()'
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, timeout=10)
            print(f"[CellQuant] Picker shortcut installed for {profile.name}")
        except Exception as e:
            print(f"[CellQuant] Could not install picker for {profile.name}: {e}")


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
