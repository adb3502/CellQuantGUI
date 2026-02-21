"""
Tests for the OxyTrack FUSION plugin.
"""

import json
import tempfile
from pathlib import Path

import pytest

from cellquant_enterprise.plugins.oxytrack.models import (
    Experiment,
    OxysterolTreatment,
    SenescenceMarker,
)
from cellquant_enterprise.plugins.oxytrack.storage import OxyTrackStore


# ── Model tests ──────────────────────────────────────────────────────


class TestOxysterolTreatment:
    def test_label(self):
        t = OxysterolTreatment(compound="7-ketocholesterol", concentration_um=10.0, duration_hours=24.0)
        assert "7-ketocholesterol" in t.label()
        assert "10" in t.label()
        assert "24" in t.label()


class TestExperiment:
    def test_roundtrip_dict(self):
        exp = Experiment(
            name="Test Experiment",
            cell_line="IMR90",
            passage="P12",
            treatments=[
                OxysterolTreatment(compound="7KC", concentration_um=5.0, duration_hours=48.0),
            ],
            markers=[
                SenescenceMarker(name="SA-beta-gal", channel_suffix="C2"),
            ],
            tags=["test", "7KC"],
        )
        d = exp.to_dict()
        restored = Experiment.from_dict(d)

        assert restored.name == "Test Experiment"
        assert restored.cell_line == "IMR90"
        assert len(restored.treatments) == 1
        assert restored.treatments[0].compound == "7KC"
        assert len(restored.markers) == 1
        assert restored.markers[0].name == "SA-beta-gal"
        assert restored.tags == ["test", "7KC"]

    def test_defaults(self):
        exp = Experiment()
        assert exp.status == "draft"
        assert exp.id  # auto-generated
        assert exp.treatments == []


# ── Storage tests ────────────────────────────────────────────────────


class TestOxyTrackStore:
    @pytest.fixture
    def store(self, tmp_path):
        return OxyTrackStore(path=tmp_path / "oxytrack.json")

    def test_add_and_list(self, store):
        exp = Experiment(name="Exp1", cell_line="BJ")
        store.add(exp)
        exps = store.list_experiments()
        assert len(exps) == 1
        assert exps[0].name == "Exp1"

    def test_get(self, store):
        exp = Experiment(name="Exp2")
        store.add(exp)
        fetched = store.get(exp.id)
        assert fetched is not None
        assert fetched.name == "Exp2"

    def test_update(self, store):
        exp = Experiment(name="Original")
        store.add(exp)
        exp.name = "Updated"
        store.update(exp)
        fetched = store.get(exp.id)
        assert fetched.name == "Updated"

    def test_delete(self, store):
        exp = Experiment(name="ToDelete")
        store.add(exp)
        assert store.delete(exp.id) is True
        assert store.get(exp.id) is None
        assert len(store.list_experiments()) == 0

    def test_delete_nonexistent(self, store):
        assert store.delete("nonexistent") is False

    def test_search(self, store):
        store.add(Experiment(name="7KC dose response", cell_line="IMR90"))
        store.add(Experiment(name="25HC time course", cell_line="BJ"))

        results = store.search("IMR90")
        assert len(results) == 1
        assert results[0].name == "7KC dose response"

        results = store.search("time course")
        assert len(results) == 1

    def test_persistence(self, tmp_path):
        path = tmp_path / "oxytrack.json"
        store1 = OxyTrackStore(path=path)
        store1.add(Experiment(name="Persistent", cell_line="HCA2"))

        # Create a new store pointing to the same file
        store2 = OxyTrackStore(path=path)
        exps = store2.list_experiments()
        assert len(exps) == 1
        assert exps[0].name == "Persistent"
