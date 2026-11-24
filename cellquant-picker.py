"""
CellQuant Host Picker Agent
============================
Runs on the Windows HOST machine (not in Docker).
Each user runs this automatically at login (via the scheduled task
installed by install-picker.bat) or via start_cellquant.bat.

The script:
  1. Detects the current Windows username and session ID
  2. Binds to a stable default port (7861) when available
  3. Registers itself with the CellQuant backend, retrying until it succeeds
  4. Listens for folder-picker requests and opens the native Windows dialog

Requirements: Python 3.x with tkinter (standard on Windows).
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "127.0.0.1"
BACKEND_URL = "http://localhost:7860/api/v1/experiments/picker-register"
DEFAULT_PORT = 7861


def get_session_id() -> int:
    """Return the current Windows logon session ID."""
    try:
        result = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Process -Id $PID).SessionId"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        return int(result)
    except Exception:
        return 1  # fallback


def get_username() -> str:
    return os.environ.get("USERNAME", os.environ.get("USER", "unknown")).lower()


def pick_folder() -> str | None:
    """Open a native Windows folder picker and return the selected path."""
    import tkinter as tk
    from tkinter import filedialog

    result: list[str | None] = [None]

    def _run():
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        root.focus_force()
        path = filedialog.askdirectory(title="Select Folder — CellQuant")
        root.destroy()
        result[0] = path or None

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=300)
    return result[0]


def register_with_backend(username: str, port: int, retry_delay: float = 3.0) -> None:
    """POST username+port to the backend so it knows which port to call.

    Docker/backend startup can lag behind the picker process, so keep retrying
    until registration succeeds instead of giving up and leaving the UI unable
    to reach the native Windows dialog.
    """
    payload = json.dumps({"username": username, "port": port}).encode()
    warned = False
    while True:
        try:
            req = urllib.request.Request(
                BACKEND_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                print(f"[picker] Registered with backend as '{username}' on port {port}")
                return
        except Exception as e:
            if not warned:
                print(f"[picker] Waiting for backend registration endpoint ({e})...")
                warned = True
            time.sleep(retry_delay)


def choose_port(session_id: int) -> int:
    """Prefer a stable port used by the app, then fall back to a session port."""
    env_port = os.environ.get("CELLQUANT_PICKER_PORT", "").strip()
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass
    return DEFAULT_PORT


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence access log

    def do_POST(self):
        if self.path == "/pick":
            try:
                path = pick_folder()
                body = json.dumps({"path": path}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()


if __name__ == "__main__":
    if sys.platform != "win32":
        print("ERROR: cellquant-picker.py must run on the Windows host machine.")
        sys.exit(1)

    username = get_username()
    session_id = get_session_id()
    port = choose_port(session_id)

    try:
        server = HTTPServer((HOST, port), Handler)
    except OSError as e:
        fallback_port = 7860 + session_id
        if fallback_port == port:
            raise
        print(f"[picker] Port {port} unavailable ({e}); falling back to session port {fallback_port}.")
        port = fallback_port
        server = HTTPServer((HOST, port), Handler)

    print(f"CellQuant Picker Agent")
    print(f"  User:    {username}")
    print(f"  Session: {session_id}")
    print(f"  Port:    {port}")
    print(f"Keep this window open while CellQuant is running.")

    # Register in background so the server starts immediately.
    threading.Thread(
        target=register_with_backend, args=(username, port), daemon=True
    ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
