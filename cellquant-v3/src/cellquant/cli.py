"""CellQuant CLI entry point."""

import argparse
import sys
import webbrowser
from pathlib import Path


def main():
    from cellquant import __version__

    parser = argparse.ArgumentParser(
        description=f"CellQuant v{__version__} - Cell quantification and tracking"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the CellQuant server")
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port (default: 8000)"
    )
    serve_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser on start"
    )
    serve_parser.add_argument(
        "--no-gpu", action="store_true", help="Disable GPU acceleration"
    )

    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up stale sessions")
    cleanup_parser.add_argument(
        "--max-age", type=int, default=24, help="Max session age in hours (default: 24)"
    )

    args = parser.parse_args()

    if args.command is None:
        # Default to serve
        args.command = "serve"
        args.port = 8000
        args.host = "127.0.0.1"
        args.no_browser = False
        args.no_gpu = False

    if args.command == "serve":
        _run_server(args)
    elif args.command == "cleanup":
        _run_cleanup(args)


def _run_server(args):
    from cellquant import __version__

    if args.no_gpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"""
    ================================================================

       CELLQUANT v{__version__}

       High-throughput cell quantification & tracking
       FastAPI + Svelte | Tiled Viewer | GPU Segmentation

    ================================================================
    """)

    url = f"http://{args.host}:{args.port}"
    print(f"  Starting server at {url}")
    print("  Press Ctrl+C to stop\n")

    if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run(
        "cellquant.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=False,
    )


def _run_cleanup(args):
    from cellquant.sessions.manager import SessionManager

    manager = SessionManager()
    count = manager.cleanup_sync(max_age_hours=args.max_age)
    print(f"Cleaned up {count} stale session(s).")


if __name__ == "__main__":
    main()
