"""Tests for SenescenceDB FUSION plugin."""

import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cellquant.plugins.senescencedb.store import SenescenceDBStore
from cellquant.plugins.senescencedb.plugin import SenescenceDBPlugin
from cellquant.plugins.registry import PluginRegistry


@pytest.fixture()
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def store(tmp_dir):
    return SenescenceDBStore(tmp_dir)


@pytest.fixture()
def client(tmp_dir):
    app = FastAPI()
    registry = PluginRegistry()
    plugin = SenescenceDBPlugin()
    registry.register(plugin)
    registry._base_data_dir = tmp_dir
    registry.mount_all(app, prefix="/api/v1/plugins")
    return TestClient(app)


# -- Store unit tests --------------------------------------------------------

class TestSenescenceDBStore:
    def test_create_paper(self, store):
        p = store.create_paper({
            "title": "Oxysterols trigger senescence in fibroblasts",
            "authors": ["Smith J", "Doe A"],
            "journal": "Cell",
            "year": 2024,
            "doi": "10.1000/test",
            "key_findings": ["7KC induces p21", "Dose-dependent SA-β-Gal"],
            "senescence_types": ["stress-induced"],
            "markers_mentioned": ["p21", "SA-β-Gal"],
            "tags": ["oxysterol", "fibroblast"],
        })
        assert p["id"]
        assert p["title"] == "Oxysterols trigger senescence in fibroblasts"
        assert p["year"] == 2024

    def test_create_marker(self, store):
        m = store.create_marker({
            "name": "SA-β-Gal",
            "gene_symbol": "GLB1",
            "marker_type": "enzymatic",
            "description": "Senescence-associated beta-galactosidase",
            "detection_methods": ["X-Gal staining", "C12FDG flow cytometry"],
            "senescence_types": ["replicative", "stress-induced"],
            "upregulated": True,
            "tags": ["gold-standard"],
        })
        assert m["name"] == "SA-β-Gal"
        assert m["marker_type"] == "enzymatic"

    def test_create_pathway(self, store):
        p = store.create_pathway({
            "name": "p53/p21 pathway",
            "description": "DNA damage response leading to cell cycle arrest",
            "key_genes": ["TP53", "CDKN1A", "MDM2"],
            "senescence_role": "pro-senescence",
            "tags": ["DDR"],
        })
        assert p["name"] == "p53/p21 pathway"
        assert len(p["key_genes"]) == 3

    def test_create_protocol(self, store):
        p = store.create_protocol({
            "title": "SA-β-Gal staining (pH 6.0)",
            "objective": "Detect senescent cells using X-Gal",
            "cell_lines": ["IMR-90", "WI-38"],
            "reagents": ["X-Gal", "citric acid buffer pH 6.0", "K3Fe(CN)6"],
            "steps": [
                "Fix cells in 2% formaldehyde for 5 min",
                "Wash 3x with PBS",
                "Add staining solution",
                "Incubate overnight at 37°C (no CO2)",
                "Count blue cells under brightfield",
            ],
            "tags": ["staining", "senescence"],
        })
        assert len(p["steps"]) == 5

    def test_create_note(self, store):
        n = store.create_note({
            "title": "7KC concentration optimisation",
            "body": "Tested 5, 10, 20 μM. 10 μM gives best senescence induction without toxicity.",
            "tags": ["oxysterol", "optimisation"],
        })
        assert n["title"] == "7KC concentration optimisation"

    def test_update_and_delete(self, store):
        p = store.create_paper({"title": "Draft paper"})
        updated = store.update_paper(p["id"], {"title": "Final paper", "year": 2025})
        assert updated["title"] == "Final paper"
        assert updated["year"] == 2025

        assert store.delete_paper(p["id"]) is True
        assert store.get_paper(p["id"]) is None

    def test_search(self, store):
        store.create_paper({
            "title": "7-ketocholesterol senescence",
            "abstract": "We show that 7KC induces senescence via oxidative stress.",
            "tags": ["oxysterol"],
        })
        store.create_marker({
            "name": "p21",
            "description": "Cyclin-dependent kinase inhibitor",
            "tags": ["cell-cycle"],
        })
        store.create_note({
            "title": "Random note",
            "body": "Unrelated content about apoptosis.",
        })

        # Search for "senescence"
        results = store.search("senescence")
        assert len(results) >= 1
        assert results[0]["entry_type"] == "paper"

        # Search with tag filter
        results = store.search("", tags=["oxysterol"])
        assert len(results) >= 1

        # Search within specific type
        results = store.search("p21", entry_types=["markers"])
        assert len(results) == 1
        assert results[0]["entry_type"] == "marker"

    def test_stats(self, store):
        store.create_paper({"title": "P1", "tags": ["a"]})
        store.create_marker({"name": "M1", "tags": ["b"]})
        store.create_pathway({"name": "W1", "tags": ["a", "c"]})

        stats = store.stats()
        assert stats["n_papers"] == 1
        assert stats["n_markers"] == 1
        assert stats["n_pathways"] == 1
        assert stats["n_protocols"] == 0
        assert stats["n_notes"] == 0
        assert set(stats["all_tags"]) == {"a", "b", "c"}

    def test_persistence(self, tmp_dir):
        store1 = SenescenceDBStore(tmp_dir)
        store1.create_paper({"title": "Persistent paper"})

        store2 = SenescenceDBStore(tmp_dir)
        papers = store2.list_papers()
        assert len(papers) == 1
        assert papers[0]["title"] == "Persistent paper"


# -- API endpoint tests ------------------------------------------------------

class TestSenescenceDBAPI:
    def test_papers_crud(self, client):
        # Create
        resp = client.post("/api/v1/plugins/senescencedb/papers", json={
            "title": "Oxysterols and senescence",
            "authors": ["Smith J"],
            "journal": "Nature",
            "year": 2024,
            "tags": ["oxysterol"],
        })
        assert resp.status_code == 201
        paper_id = resp.json()["id"]

        # List
        resp = client.get("/api/v1/plugins/senescencedb/papers")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        # Get
        resp = client.get(f"/api/v1/plugins/senescencedb/papers/{paper_id}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Oxysterols and senescence"

        # Update
        resp = client.patch(f"/api/v1/plugins/senescencedb/papers/{paper_id}", json={
            "year": 2025,
        })
        assert resp.status_code == 200
        assert resp.json()["year"] == 2025

        # Delete
        resp = client.delete(f"/api/v1/plugins/senescencedb/papers/{paper_id}")
        assert resp.status_code == 200

        resp = client.get(f"/api/v1/plugins/senescencedb/papers/{paper_id}")
        assert resp.status_code == 404

    def test_markers_crud(self, client):
        resp = client.post("/api/v1/plugins/senescencedb/markers", json={
            "name": "p21",
            "gene_symbol": "CDKN1A",
            "marker_type": "nuclear",
            "description": "CDK inhibitor upregulated in senescence",
            "upregulated": True,
        })
        assert resp.status_code == 201
        marker_id = resp.json()["id"]

        resp = client.get(f"/api/v1/plugins/senescencedb/markers/{marker_id}")
        assert resp.status_code == 200
        assert resp.json()["gene_symbol"] == "CDKN1A"

    def test_pathways_crud(self, client):
        resp = client.post("/api/v1/plugins/senescencedb/pathways", json={
            "name": "p53/p21",
            "key_genes": ["TP53", "CDKN1A"],
            "senescence_role": "pro-senescence",
        })
        assert resp.status_code == 201

        resp = client.get("/api/v1/plugins/senescencedb/pathways")
        assert len(resp.json()) == 1

    def test_protocols_crud(self, client):
        resp = client.post("/api/v1/plugins/senescencedb/protocols", json={
            "title": "SA-β-Gal staining",
            "objective": "Detect senescent cells",
            "steps": ["Fix", "Wash", "Stain", "Count"],
        })
        assert resp.status_code == 201
        pid = resp.json()["id"]

        resp = client.patch(f"/api/v1/plugins/senescencedb/protocols/{pid}", json={
            "tips": ["Use pH 6.0 buffer exactly"],
        })
        assert resp.status_code == 200
        assert resp.json()["tips"] == ["Use pH 6.0 buffer exactly"]

    def test_notes_crud(self, client):
        resp = client.post("/api/v1/plugins/senescencedb/notes", json={
            "title": "Lab meeting notes",
            "body": "Discussed 7KC dose optimization results",
            "tags": ["meeting"],
        })
        assert resp.status_code == 201
        nid = resp.json()["id"]

        resp = client.get(f"/api/v1/plugins/senescencedb/notes/{nid}")
        assert resp.json()["body"] == "Discussed 7KC dose optimization results"

    def test_search(self, client):
        client.post("/api/v1/plugins/senescencedb/papers", json={
            "title": "Oxysterol-induced senescence",
            "abstract": "7-ketocholesterol drives stress-induced senescence",
        })
        client.post("/api/v1/plugins/senescencedb/markers", json={
            "name": "p16",
            "description": "Cell cycle inhibitor in senescence",
        })

        resp = client.post("/api/v1/plugins/senescencedb/search", json={
            "query": "senescence",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_hits"] >= 2

    def test_stats(self, client):
        client.post("/api/v1/plugins/senescencedb/papers", json={
            "title": "P1", "tags": ["tag1"],
        })
        client.post("/api/v1/plugins/senescencedb/markers", json={
            "name": "M1", "tags": ["tag2"],
        })

        resp = client.get("/api/v1/plugins/senescencedb/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_papers"] == 1
        assert data["n_markers"] == 1
        assert set(data["all_tags"]) == {"tag1", "tag2"}

    def test_not_found(self, client):
        resp = client.get("/api/v1/plugins/senescencedb/papers/nonexistent")
        assert resp.status_code == 404
        resp = client.get("/api/v1/plugins/senescencedb/markers/nonexistent")
        assert resp.status_code == 404
        resp = client.get("/api/v1/plugins/senescencedb/pathways/nonexistent")
        assert resp.status_code == 404
        resp = client.get("/api/v1/plugins/senescencedb/protocols/nonexistent")
        assert resp.status_code == 404
        resp = client.get("/api/v1/plugins/senescencedb/notes/nonexistent")
        assert resp.status_code == 404
