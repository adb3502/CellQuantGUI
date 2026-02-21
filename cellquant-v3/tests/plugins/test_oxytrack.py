"""Tests for OxyTrack FUSION plugin."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cellquant.plugins.oxytrack.store import OxyTrackStore
from cellquant.plugins.oxytrack.plugin import OxyTrackPlugin
from cellquant.plugins.registry import PluginRegistry


@pytest.fixture()
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def store(tmp_dir):
    return OxyTrackStore(tmp_dir)


@pytest.fixture()
def client(tmp_dir):
    """Create a test client with OxyTrack mounted."""
    app = FastAPI()
    registry = PluginRegistry()
    plugin = OxyTrackPlugin()
    registry.register(plugin)
    # Override base data dir to use temp
    registry._base_data_dir = tmp_dir
    registry.mount_all(app, prefix="/api/v1/plugins")
    return TestClient(app)


# -- Store unit tests --------------------------------------------------------

class TestOxyTrackStore:
    def test_create_experiment(self, store):
        exp = store.create_experiment(
            name="7KC dose response",
            cell_line="IMR-90",
            passage=15,
            treatments=[
                {"name": "Vehicle", "oxysterol": "none", "concentration_uM": 0},
                {"name": "7KC 10μM", "oxysterol": "7-ketocholesterol", "concentration_uM": 10},
            ],
            timepoints=[{"hours": 24}, {"hours": 48}, {"hours": 72}],
            marker_panel={"markers": ["SA-β-Gal", "p21", "γH2AX"]},
            tags=["oxysterol", "dose-response"],
        )
        assert exp.id
        assert exp.name == "7KC dose response"
        assert exp.cell_line == "IMR-90"
        assert len(exp.treatments) == 2
        assert len(exp.timepoints) == 3
        assert exp.marker_panel.markers == ["SA-β-Gal", "p21", "γH2AX"]

    def test_list_experiments(self, store):
        store.create_experiment(name="Exp1", cell_line="WI-38")
        store.create_experiment(name="Exp2", cell_line="IMR-90")
        exps = store.list_experiments()
        assert len(exps) == 2

    def test_get_experiment(self, store):
        exp = store.create_experiment(name="Test", cell_line="HDF")
        loaded = store.get_experiment(exp.id)
        assert loaded is not None
        assert loaded.name == "Test"

    def test_update_experiment(self, store):
        exp = store.create_experiment(name="Original", cell_line="WI-38")
        updated = store.update_experiment(exp.id, {"name": "Updated", "passage": 20})
        assert updated.name == "Updated"
        assert updated.passage == 20

    def test_delete_experiment(self, store):
        exp = store.create_experiment(name="ToDelete", cell_line="WI-38")
        assert store.delete_experiment(exp.id) is True
        assert store.get_experiment(exp.id) is None
        assert store.delete_experiment("nonexistent") is False

    def test_add_observation(self, store):
        exp = store.create_experiment(
            name="Obs test",
            cell_line="IMR-90",
            treatments=[{"name": "7KC", "oxysterol": "7-ketocholesterol", "concentration_uM": 10}],
        )
        obs = store.add_observation(exp.id, {
            "treatment_id": "T00",
            "timepoint_hours": 24,
            "marker": "p21",
            "value": 1500.0,
            "n_cells": 200,
            "std_dev": 320.5,
        })
        assert obs is not None
        assert obs.marker == "p21"
        assert obs.value == 1500.0

        reloaded = store.get_experiment(exp.id)
        assert len(reloaded.observations) == 1

    def test_delete_observation(self, store):
        exp = store.create_experiment(name="DelObs", cell_line="IMR-90")
        obs = store.add_observation(exp.id, {
            "treatment_id": "T00",
            "timepoint_hours": 24,
            "marker": "p21",
            "value": 100.0,
        })
        assert store.delete_observation(exp.id, obs.id) is True
        reloaded = store.get_experiment(exp.id)
        assert len(reloaded.observations) == 0

    def test_persistence(self, tmp_dir):
        """Data survives store re-instantiation."""
        store1 = OxyTrackStore(tmp_dir)
        exp = store1.create_experiment(name="Persist", cell_line="HDF")

        store2 = OxyTrackStore(tmp_dir)
        loaded = store2.get_experiment(exp.id)
        assert loaded is not None
        assert loaded.name == "Persist"


# -- API endpoint tests ------------------------------------------------------

class TestOxyTrackAPI:
    def test_create_and_list_experiments(self, client):
        resp = client.post("/api/v1/plugins/oxytrack/experiments", json={
            "name": "7KC time-course",
            "cell_line": "IMR-90",
            "passage": 12,
            "treatments": [
                {"name": "Vehicle", "oxysterol": "none", "concentration_uM": 0},
                {"name": "7KC", "oxysterol": "7-ketocholesterol", "concentration_uM": 10},
            ],
            "timepoints": [{"hours": 24}, {"hours": 48}],
            "marker_panel": {"markers": ["p21", "SA-β-Gal"]},
            "tags": ["timecourse"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "7KC time-course"
        assert len(data["treatments"]) == 2
        exp_id = data["id"]

        # List
        resp = client.get("/api/v1/plugins/oxytrack/experiments")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        # Get
        resp = client.get(f"/api/v1/plugins/oxytrack/experiments/{exp_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == exp_id

    def test_update_experiment(self, client):
        resp = client.post("/api/v1/plugins/oxytrack/experiments", json={
            "name": "Orig", "cell_line": "WI-38",
        })
        exp_id = resp.json()["id"]

        resp = client.patch(f"/api/v1/plugins/oxytrack/experiments/{exp_id}", json={
            "name": "Updated",
        })
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    def test_delete_experiment(self, client):
        resp = client.post("/api/v1/plugins/oxytrack/experiments", json={
            "name": "ToDelete", "cell_line": "WI-38",
        })
        exp_id = resp.json()["id"]

        resp = client.delete(f"/api/v1/plugins/oxytrack/experiments/{exp_id}")
        assert resp.status_code == 200

        resp = client.get(f"/api/v1/plugins/oxytrack/experiments/{exp_id}")
        assert resp.status_code == 404

    def test_add_observation(self, client):
        resp = client.post("/api/v1/plugins/oxytrack/experiments", json={
            "name": "Obs", "cell_line": "IMR-90",
            "treatments": [{"name": "7KC", "oxysterol": "7KC", "concentration_uM": 10}],
        })
        exp_id = resp.json()["id"]

        resp = client.post(f"/api/v1/plugins/oxytrack/experiments/{exp_id}/observations", json={
            "treatment_id": "T00",
            "timepoint_hours": 24,
            "marker": "p21",
            "value": 1200.0,
            "n_cells": 150,
            "std_dev": 250.0,
        })
        assert resp.status_code == 201
        assert resp.json()["marker"] == "p21"

    def test_timecourse(self, client):
        # Create experiment with observations
        resp = client.post("/api/v1/plugins/oxytrack/experiments", json={
            "name": "TC", "cell_line": "IMR-90",
            "treatments": [{"name": "7KC", "oxysterol": "7KC", "concentration_uM": 10}],
            "timepoints": [{"hours": 24}, {"hours": 48}],
        })
        exp_id = resp.json()["id"]

        for hours, val in [(24, 1000.0), (48, 1800.0)]:
            client.post(f"/api/v1/plugins/oxytrack/experiments/{exp_id}/observations", json={
                "treatment_id": "T00",
                "timepoint_hours": hours,
                "marker": "p21",
                "value": val,
                "std_dev": 100.0,
            })

        resp = client.get(f"/api/v1/plugins/oxytrack/experiments/{exp_id}/timecourse")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["series"]) == 1
        assert data["series"][0]["hours"] == [24, 48]

    def test_dose_response(self, client):
        resp = client.post("/api/v1/plugins/oxytrack/experiments", json={
            "name": "DR", "cell_line": "IMR-90",
            "treatments": [
                {"name": "Veh", "oxysterol": "7KC", "concentration_uM": 0},
                {"name": "Low", "oxysterol": "7KC", "concentration_uM": 5},
                {"name": "High", "oxysterol": "7KC", "concentration_uM": 20},
            ],
        })
        exp_id = resp.json()["id"]

        for tid, val in [("T00", 500.0), ("T01", 1200.0), ("T02", 2500.0)]:
            client.post(f"/api/v1/plugins/oxytrack/experiments/{exp_id}/observations", json={
                "treatment_id": tid,
                "timepoint_hours": 48,
                "marker": "p21",
                "value": val,
                "n_cells": 100,
            })

        resp = client.get(
            f"/api/v1/plugins/oxytrack/experiments/{exp_id}/dose-response",
            params={"timepoint_hours": 48},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["series"]) == 1
        assert len(data["series"][0]["points"]) == 3

    def test_not_found(self, client):
        resp = client.get("/api/v1/plugins/oxytrack/experiments/nonexistent")
        assert resp.status_code == 404
