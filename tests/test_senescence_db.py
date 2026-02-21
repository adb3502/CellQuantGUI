"""
Tests for the SenescenceDB FUSION plugin.
"""

import pytest

from cellquant_enterprise.plugins.senescence_db.models import Paper, Protocol, Finding
from cellquant_enterprise.plugins.senescence_db.storage import SenescenceDBStore


# ── Model tests ──────────────────────────────────────────────────────


class TestPaper:
    def test_roundtrip(self):
        p = Paper(
            title="Oxysterols and senescence",
            authors="Smith J, Doe A",
            journal="J Cell Biol",
            year=2023,
            doi="10.1234/jcb.2023",
            tags=["oxysterol", "senescence"],
        )
        d = p.to_dict()
        restored = Paper.from_dict(d)
        assert restored.title == p.title
        assert restored.year == 2023
        assert restored.tags == ["oxysterol", "senescence"]

    def test_citation(self):
        p = Paper(authors="Smith J", year=2023, title="Test Paper", journal="Nature")
        c = p.citation()
        assert "Smith J" in c
        assert "(2023)" in c


class TestProtocol:
    def test_roundtrip(self):
        p = Protocol(
            title="SA-beta-gal Staining",
            category="staining",
            steps="1. Fix cells\n2. Stain at pH 6.0",
            reagents="X-gal, citric acid",
        )
        d = p.to_dict()
        restored = Protocol.from_dict(d)
        assert restored.title == p.title
        assert restored.category == "staining"


class TestFinding:
    def test_roundtrip(self):
        f = Finding(
            title="7KC induces p21 upregulation",
            summary="7-ketocholesterol treatment causes p21 increase in IMR90 cells",
            category="mechanism",
            confidence="high",
            tags=["7KC", "p21"],
        )
        d = f.to_dict()
        restored = Finding.from_dict(d)
        assert restored.title == f.title
        assert restored.confidence == "high"


# ── Storage tests ────────────────────────────────────────────────────


class TestSenescenceDBStore:
    @pytest.fixture
    def store(self, tmp_path):
        return SenescenceDBStore(path=tmp_path / "senescence_db.json")

    # -- Papers --

    def test_add_and_list_papers(self, store):
        store.add_paper(Paper(title="Paper 1"))
        assert len(store.list_papers()) == 1

    def test_update_paper(self, store):
        p = Paper(title="Original")
        store.add_paper(p)
        p.title = "Updated"
        store.update_paper(p)
        assert store.get_paper(p.id).title == "Updated"

    def test_delete_paper(self, store):
        p = Paper(title="Delete Me")
        store.add_paper(p)
        assert store.delete_paper(p.id) is True
        assert store.get_paper(p.id) is None

    # -- Protocols --

    def test_add_and_list_protocols(self, store):
        store.add_protocol(Protocol(title="Protocol 1"))
        assert len(store.list_protocols()) == 1

    def test_update_protocol(self, store):
        p = Protocol(title="Original")
        store.add_protocol(p)
        p.title = "Updated"
        store.update_protocol(p)
        assert store.get_protocol(p.id).title == "Updated"

    def test_delete_protocol(self, store):
        p = Protocol(title="Delete Me")
        store.add_protocol(p)
        assert store.delete_protocol(p.id) is True
        assert store.get_protocol(p.id) is None

    # -- Findings --

    def test_add_and_list_findings(self, store):
        store.add_finding(Finding(title="Finding 1"))
        assert len(store.list_findings()) == 1

    def test_update_finding(self, store):
        f = Finding(title="Original")
        store.add_finding(f)
        f.title = "Updated"
        store.update_finding(f)
        assert store.get_finding(f.id).title == "Updated"

    def test_delete_finding(self, store):
        f = Finding(title="Delete Me")
        store.add_finding(f)
        assert store.delete_finding(f.id) is True
        assert store.get_finding(f.id) is None

    # -- Search --

    def test_search_papers(self, store):
        store.add_paper(Paper(title="Oxysterol review", tags=["review"]))
        store.add_paper(Paper(title="Senescence mechanisms"))
        results = store.search("oxysterol")
        assert len(results["papers"]) == 1
        assert results["papers"][0].title == "Oxysterol review"

    def test_search_protocols(self, store):
        store.add_protocol(Protocol(title="SA-beta-gal staining", category="staining"))
        results = store.search("beta-gal")
        assert len(results["protocols"]) == 1

    def test_search_findings(self, store):
        store.add_finding(Finding(title="p21 upregulation by 7KC", category="mechanism"))
        results = store.search("p21")
        assert len(results["findings"]) == 1

    # -- Stats --

    def test_stats(self, store):
        store.add_paper(Paper(title="P1", tags=["tag1", "tag2"]))
        store.add_protocol(Protocol(title="Pr1", tags=["tag2", "tag3"]))
        store.add_finding(Finding(title="F1", tags=["tag1"]))
        s = store.stats()
        assert s["papers"] == 1
        assert s["protocols"] == 1
        assert s["findings"] == 1
        assert s["tags"] == 3  # tag1, tag2, tag3

    # -- Persistence --

    def test_persistence(self, tmp_path):
        path = tmp_path / "senescence_db.json"
        store1 = SenescenceDBStore(path=path)
        store1.add_paper(Paper(title="Persistent Paper"))
        store1.add_protocol(Protocol(title="Persistent Protocol"))
        store1.add_finding(Finding(title="Persistent Finding"))

        store2 = SenescenceDBStore(path=path)
        assert len(store2.list_papers()) == 1
        assert len(store2.list_protocols()) == 1
        assert len(store2.list_findings()) == 1
