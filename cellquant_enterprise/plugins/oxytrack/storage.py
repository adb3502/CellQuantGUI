"""
JSON-file storage backend for OxyTrack experiments.
"""

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from cellquant_enterprise.plugins.oxytrack.models import Experiment


class OxyTrackStore:
    """Persists experiments to a JSON file."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = Path.home() / ".cellquant_enterprise" / "oxytrack.json"
        self.path = path
        self._experiments: List[Experiment] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            self._experiments = [Experiment.from_dict(d) for d in data]
        else:
            self._experiments = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([e.to_dict() for e in self._experiments], f, indent=2)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_experiments(self) -> List[Experiment]:
        return list(self._experiments)

    def get(self, experiment_id: str) -> Optional[Experiment]:
        for exp in self._experiments:
            if exp.id == experiment_id:
                return exp
        return None

    def add(self, experiment: Experiment) -> Experiment:
        experiment.date_modified = datetime.now().isoformat(timespec="seconds")
        self._experiments.append(experiment)
        self._save()
        return experiment

    def update(self, experiment: Experiment) -> None:
        experiment.date_modified = datetime.now().isoformat(timespec="seconds")
        for i, existing in enumerate(self._experiments):
            if existing.id == experiment.id:
                self._experiments[i] = experiment
                break
        self._save()

    def delete(self, experiment_id: str) -> bool:
        before = len(self._experiments)
        self._experiments = [e for e in self._experiments if e.id != experiment_id]
        if len(self._experiments) < before:
            self._save()
            return True
        return False

    def search(self, query: str) -> List[Experiment]:
        """Simple text search across name, description, cell_line, tags."""
        query_lower = query.lower()
        results = []
        for exp in self._experiments:
            searchable = " ".join([
                exp.name, exp.description, exp.cell_line,
                " ".join(exp.tags), exp.notes,
                " ".join(t.compound for t in exp.treatments),
            ]).lower()
            if query_lower in searchable:
                results.append(exp)
        return results
