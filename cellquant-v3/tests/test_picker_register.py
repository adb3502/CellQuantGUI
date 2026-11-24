"""/picker-register must only accept the local machine.

The server listens on 0.0.0.0 with host networking, so this endpoint was
reachable from the whole lab network despite its "host only" comment. Any
machine could register a port under any username; /browse would then POST
to that port and hand the user whatever folder path it returned.
"""
import asyncio

import pytest
from fastapi import HTTPException


class _Client:
    def __init__(self, host):
        self.host = host


class _Req:
    def __init__(self, host):
        self.client = _Client(host) if host else None


def _register(host, body):
    from cellquant.api.routers import experiments

    experiments._picker_registry.clear()
    return asyncio.run(experiments.picker_register(body, _Req(host)))


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "127.0.0.5"])
def test_loopback_may_register(host):
    from cellquant.api.routers import experiments

    assert _register(host, {"username": "amritha", "port": 7863}) == {"ok": True}
    assert experiments._picker_registry["amritha"] == 7863


@pytest.mark.parametrize("host", ["10.0.0.7", "192.168.1.20", "172.17.0.2", None])
def test_other_machines_are_refused(host):
    from cellquant.api.routers import experiments

    with pytest.raises(HTTPException) as e:
        _register(host, {"username": "amritha", "port": 7863})
    assert e.value.status_code == 403
    assert "amritha" not in experiments._picker_registry


def test_bad_port_is_rejected():
    with pytest.raises(HTTPException) as e:
        _register("127.0.0.1", {"username": "a", "port": "not-a-number"})
    assert e.value.status_code == 400
