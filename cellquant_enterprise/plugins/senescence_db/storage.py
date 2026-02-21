"""
JSON-file storage backend for SenescenceDB.
"""

import json
from pathlib import Path
from typing import List, Optional

from cellquant_enterprise.plugins.senescence_db.models import Paper, Protocol, Finding


class SenescenceDBStore:
    """Persists the knowledge base to a JSON file."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = Path.home() / ".cellquant_enterprise" / "senescence_db.json"
        self.path = path
        self._papers: List[Paper] = []
        self._protocols: List[Protocol] = []
        self._findings: List[Finding] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            self._papers = [Paper.from_dict(d) for d in data.get("papers", [])]
            self._protocols = [Protocol.from_dict(d) for d in data.get("protocols", [])]
            self._findings = [Finding.from_dict(d) for d in data.get("findings", [])]
        else:
            self._papers = []
            self._protocols = []
            self._findings = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "papers": [p.to_dict() for p in self._papers],
            "protocols": [p.to_dict() for p in self._protocols],
            "findings": [f.to_dict() for f in self._findings],
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Papers
    # ------------------------------------------------------------------

    def list_papers(self) -> List[Paper]:
        return list(self._papers)

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        for p in self._papers:
            if p.id == paper_id:
                return p
        return None

    def add_paper(self, paper: Paper) -> Paper:
        self._papers.append(paper)
        self._save()
        return paper

    def update_paper(self, paper: Paper) -> None:
        for i, existing in enumerate(self._papers):
            if existing.id == paper.id:
                self._papers[i] = paper
                break
        self._save()

    def delete_paper(self, paper_id: str) -> bool:
        before = len(self._papers)
        self._papers = [p for p in self._papers if p.id != paper_id]
        if len(self._papers) < before:
            self._save()
            return True
        return False

    # ------------------------------------------------------------------
    # Protocols
    # ------------------------------------------------------------------

    def list_protocols(self) -> List[Protocol]:
        return list(self._protocols)

    def get_protocol(self, protocol_id: str) -> Optional[Protocol]:
        for p in self._protocols:
            if p.id == protocol_id:
                return p
        return None

    def add_protocol(self, protocol: Protocol) -> Protocol:
        self._protocols.append(protocol)
        self._save()
        return protocol

    def update_protocol(self, protocol: Protocol) -> None:
        for i, existing in enumerate(self._protocols):
            if existing.id == protocol.id:
                self._protocols[i] = protocol
                break
        self._save()

    def delete_protocol(self, protocol_id: str) -> bool:
        before = len(self._protocols)
        self._protocols = [p for p in self._protocols if p.id != protocol_id]
        if len(self._protocols) < before:
            self._save()
            return True
        return False

    # ------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------

    def list_findings(self) -> List[Finding]:
        return list(self._findings)

    def get_finding(self, finding_id: str) -> Optional[Finding]:
        for f in self._findings:
            if f.id == finding_id:
                return f
        return None

    def add_finding(self, finding: Finding) -> Finding:
        self._findings.append(finding)
        self._save()
        return finding

    def update_finding(self, finding: Finding) -> None:
        for i, existing in enumerate(self._findings):
            if existing.id == finding.id:
                self._findings[i] = finding
                break
        self._save()

    def delete_finding(self, finding_id: str) -> bool:
        before = len(self._findings)
        self._findings = [f for f in self._findings if f.id != finding_id]
        if len(self._findings) < before:
            self._save()
            return True
        return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str) -> dict:
        """Search across all collections. Returns dict with papers, protocols, findings."""
        q = query.lower()

        def _match_paper(p: Paper) -> bool:
            text = " ".join([p.title, p.authors, p.journal, p.abstract, p.notes, " ".join(p.tags)]).lower()
            return q in text

        def _match_protocol(p: Protocol) -> bool:
            text = " ".join([p.title, p.category, p.steps, p.reagents, p.tips, " ".join(p.tags)]).lower()
            return q in text

        def _match_finding(f: Finding) -> bool:
            text = " ".join([f.title, f.summary, f.details, f.category, " ".join(f.tags)]).lower()
            return q in text

        return {
            "papers": [p for p in self._papers if _match_paper(p)],
            "protocols": [p for p in self._protocols if _match_protocol(p)],
            "findings": [f for f in self._findings if _match_finding(f)],
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        all_tags = set()
        for p in self._papers:
            all_tags.update(p.tags)
        for p in self._protocols:
            all_tags.update(p.tags)
        for f in self._findings:
            all_tags.update(f.tags)

        return {
            "papers": len(self._papers),
            "protocols": len(self._protocols),
            "findings": len(self._findings),
            "tags": len(all_tags),
        }
