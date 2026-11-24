"""The Windows folder-picker agent must only answer the CellQuant backend.

It listens on localhost and opens a native folder dialog on POST /pick.
Any web page open in the user's browser can send a "simple" cross-site POST
to localhost without a CORS preflight, so an unguarded /pick lets an
arbitrary site pop dialogs on the user's screen. Requiring a custom header
closes that: a browser cannot attach one cross-site unless the agent
approves a preflight, which it never does. The backend sends the header.
"""
import asyncio
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

pytest.importorskip("tkinter")

PICKER = Path(__file__).resolve().parents[1] / "cellquant-picker.py"


def _load_picker():
    spec = importlib.util.spec_from_file_location("cellquant_picker", PICKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def agent(monkeypatch):
    mod = _load_picker()
    calls = []

    def fake_pick():
        calls.append(1)
        return "D:/data/run1"

    monkeypatch.setattr(mod, "pick_folder", fake_pick)
    server = HTTPServer(("127.0.0.1", 0), mod.PickerHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield mod, f"http://127.0.0.1:{server.server_address[1]}", calls
    server.shutdown()


def _post(url, headers=None):
    req = urllib.request.Request(url, data=b"", method="POST", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_pick_without_header_is_refused_and_opens_nothing(agent):
    mod, base, calls = agent
    status, _, _ = _post(base + "/pick", {"Origin": "https://evil.example"})
    assert status == 403
    assert calls == [], "no dialog may open for an unauthenticated request"


def test_pick_with_backend_header_works(agent):
    mod, base, calls = agent
    status, _, body = _post(base + "/pick", {mod.PICKER_HEADER: "1"})
    assert status == 200
    assert json.loads(body) == {"path": "D:/data/run1"}
    assert calls == [1]


def test_never_grants_cross_origin_access(agent):
    mod, base, _ = agent
    _, headers, _ = _post(base + "/pick", {mod.PICKER_HEADER: "1", "Origin": "https://evil.example"})
    assert not any(k.lower() == "access-control-allow-origin" for k in headers)


def test_backend_sends_the_header_the_agent_requires(monkeypatch):
    """The two sides are separate programs; the header name must match."""
    from cellquant.api.routers import experiments

    mod = _load_picker()
    assert experiments.PICKER_HEADER == mod.PICKER_HEADER

    sent = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"path": "D:/x"}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, **k):
            sent["headers"] = headers or {}
            return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    experiments._picker_registry.clear()
    experiments._picker_registry["amritha"] = 7863

    class User:
        username = "Amritha"

    out = asyncio.run(experiments.browse_folder({}, User()))
    assert out == {"path": "D:/x"}
    assert sent["headers"].get(mod.PICKER_HEADER) == "1"
