"""
cellquant-picker.py — Native Windows folder-picker agent for CellQuant.

Run this script in your Windows session (NOT inside Docker):
    python cellquant-picker.py

It starts a tiny HTTP server on port 7861 (configurable via --port).
When the CellQuant backend calls POST /pick, it opens a native Windows
folder-selection dialog in your session and returns the chosen path.

It also registers itself with the backend via /picker-register so the
backend knows which port to reach it on.

IMPORTANT: registration is keyed by the CellQuant *login* name, which is what
/browse looks up — not the Windows account name. When they differ (e.g. Windows
"adb" vs CellQuant "admin"), pass --username explicitly or Browse will report
"picker not running". Every auto-start entry must pass it too; a single one that
omits it will overwrite a correct registration via the 30s heartbeat.
"""

import argparse
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen
import json

# Windows-only: tkinter for the folder dialog
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    print("ERROR: tkinter not available. Install Python with tkinter support.")
    sys.exit(1)


DEFAULT_PORT = 0  # 0 = auto-assign a free port
BACKEND_URL = os.environ.get("CELLQUANT_BACKEND", "http://localhost:7860")


def pick_folder() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(parent=root, title="Select folder")
    root.destroy()
    return path if path else None


class PickerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress access logs

    def do_POST(self):
        if self.path == "/pick":
            path = pick_folder()
            body = json.dumps({"path": path}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def register_with_backend(port: int, username: str, retries: int = 5):
    url = f"{BACKEND_URL}/api/v1/experiments/picker-register"
    payload = json.dumps({"username": username, "port": port}).encode()
    for attempt in range(retries):
        try:
            req = Request(url, data=payload, headers={"Content-Type": "application/json"})
            urlopen(req, timeout=5)
            print(f"Registered with backend at {BACKEND_URL} (port={port}, user={username})")
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"Warning: could not register with backend: {e}")
    return False


def heartbeat_loop(port: int, username: str, interval: int = 30):
    """Re-register every interval seconds so backend restarts don't break Browse."""
    while True:
        time.sleep(interval)
        try:
            register_with_backend(port, username, retries=1)
        except Exception:
            pass


def main():
    global BACKEND_URL
    parser = argparse.ArgumentParser(description="CellQuant folder-picker agent")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--backend", default=BACKEND_URL)
    # NOTE: this must match the CellQuant *login* name, which /browse uses as the
    # registry key — not necessarily the Windows account name. Pass --username
    # explicitly when they differ (e.g. Windows "adb" vs CellQuant "admin").
    parser.add_argument("--username", default=os.environ.get("USERNAME", "").lower())
    # Loopback by default: the handler is unauthenticated, so a wider bind would
    # let anything on the LAN pop a file dialog on this desktop. The backend
    # runs with network_mode: host and so reaches the picker over loopback.
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default: loopback; widen only with a trusted network)",
    )
    args = parser.parse_args()
    BACKEND_URL = args.backend

    # The backend must run with network_mode: host for /browse to reach this
    # over loopback; on a bridge network the callback is refused. See the
    # network_mode comment in docker-compose.yml.
    server = HTTPServer((args.host, args.port), PickerHandler)
    actual_port = server.server_address[1]  # resolved port when args.port==0
    print(f"CellQuant picker agent listening on port {actual_port} (user: {args.username})")
    print("Keep this window open while using CellQuant. Press Ctrl+C to stop.")

    # Register immediately, then keep re-registering so backend restarts don't break Browse
    t = threading.Thread(
        target=register_with_backend,
        args=(actual_port, args.username),
        daemon=True,
    )
    t.start()
    hb = threading.Thread(
        target=heartbeat_loop,
        args=(actual_port, args.username),
        daemon=True,
    )
    hb.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
