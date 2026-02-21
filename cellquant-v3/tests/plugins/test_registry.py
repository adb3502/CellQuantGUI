"""Tests for the FUSION plugin registry."""

import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cellquant.plugins.registry import PluginRegistry
from cellquant.plugins.oxytrack import OxyTrackPlugin
from cellquant.plugins.senescencedb import SenescenceDBPlugin


@pytest.fixture()
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestPluginRegistry:
    def test_register_and_list(self):
        registry = PluginRegistry()
        registry.register(OxyTrackPlugin())
        registry.register(SenescenceDBPlugin())

        plugins = registry.list_plugins()
        assert len(plugins) == 2
        slugs = {p.slug for p in plugins}
        assert slugs == {"oxytrack", "senescencedb"}

    def test_duplicate_registration_raises(self):
        registry = PluginRegistry()
        registry.register(OxyTrackPlugin())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(OxyTrackPlugin())

    def test_get_plugin(self):
        registry = PluginRegistry()
        registry.register(OxyTrackPlugin())
        assert registry.get("oxytrack") is not None
        assert registry.get("nonexistent") is None

    def test_mount_all(self, tmp_dir):
        app = FastAPI()
        registry = PluginRegistry()
        registry._base_data_dir = tmp_dir
        registry.register(OxyTrackPlugin())
        registry.register(SenescenceDBPlugin())
        registry.mount_all(app, prefix="/api/v1/plugins")

        client = TestClient(app)

        # OxyTrack endpoints should exist
        resp = client.get("/api/v1/plugins/oxytrack/experiments")
        assert resp.status_code == 200

        # SenescenceDB endpoints should exist
        resp = client.get("/api/v1/plugins/senescencedb/papers")
        assert resp.status_code == 200
        resp = client.get("/api/v1/plugins/senescencedb/stats")
        assert resp.status_code == 200

    def test_plugin_metadata(self):
        oxy = OxyTrackPlugin()
        meta = oxy.metadata()
        assert meta.name == "OxyTrack"
        assert meta.slug == "oxytrack"
        assert meta.version == "1.0.0"

        sdb = SenescenceDBPlugin()
        meta = sdb.metadata()
        assert meta.name == "SenescenceDB"
        assert meta.slug == "senescencedb"
        assert meta.version == "1.0.0"
