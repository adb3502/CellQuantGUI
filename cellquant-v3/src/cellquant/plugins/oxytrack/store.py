"""JSON-file backed storage for OxyTrack experiments."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from cellquant.plugins.oxytrack.models import (
    Experiment,
    Observation,
    TimePoint,
    TreatmentArm,
    MarkerPanel,
)


class OxyTrackStore:
    """Persist experiments as individual JSON files under a data directory."""

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / "experiments"
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- helpers -------------------------------------------------------------

    def _path(self, experiment_id: str) -> Path:
        return self._dir / f"{experiment_id}.json"

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:12]

    def _save(self, exp: Experiment) -> None:
        with open(self._path(exp.id), "w") as f:
            json.dump(asdict(exp), f, indent=2, default=str)

    def _load(self, experiment_id: str) -> Optional[Experiment]:
        path = self._path(experiment_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return _dict_to_experiment(data)

    # -- CRUD ----------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        cell_line: str,
        passage: int = 0,
        description: str = "",
        treatments: Optional[List[dict]] = None,
        timepoints: Optional[List[dict]] = None,
        marker_panel: Optional[dict] = None,
        tags: Optional[List[str]] = None,
    ) -> Experiment:
        exp_id = self._new_id()
        now = datetime.utcnow().isoformat()

        treatment_objs = []
        for i, t in enumerate(treatments or []):
            treatment_objs.append(
                TreatmentArm(id=f"T{i:02d}", **t)
            )

        tp_objs = [TimePoint(**tp) for tp in (timepoints or [])]
        mp = MarkerPanel(**(marker_panel or {}))

        exp = Experiment(
            id=exp_id,
            name=name,
            cell_line=cell_line,
            passage=passage,
            description=description,
            treatments=treatment_objs,
            timepoints=tp_objs,
            marker_panel=mp,
            created_at=now,
            updated_at=now,
            tags=tags or [],
        )
        self._save(exp)
        return exp

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        return self._load(experiment_id)

    def list_experiments(self) -> List[Experiment]:
        experiments = []
        for path in sorted(self._dir.glob("*.json")):
            exp = self._load(path.stem)
            if exp:
                experiments.append(exp)
        return experiments

    def update_experiment(self, experiment_id: str, updates: dict) -> Optional[Experiment]:
        exp = self._load(experiment_id)
        if exp is None:
            return None

        if "name" in updates and updates["name"] is not None:
            exp.name = updates["name"]
        if "cell_line" in updates and updates["cell_line"] is not None:
            exp.cell_line = updates["cell_line"]
        if "passage" in updates and updates["passage"] is not None:
            exp.passage = updates["passage"]
        if "description" in updates and updates["description"] is not None:
            exp.description = updates["description"]
        if "tags" in updates and updates["tags"] is not None:
            exp.tags = updates["tags"]

        if "treatments" in updates and updates["treatments"] is not None:
            exp.treatments = [
                TreatmentArm(id=f"T{i:02d}", **t)
                for i, t in enumerate(updates["treatments"])
            ]
        if "timepoints" in updates and updates["timepoints"] is not None:
            exp.timepoints = [TimePoint(**tp) for tp in updates["timepoints"]]
        if "marker_panel" in updates and updates["marker_panel"] is not None:
            exp.marker_panel = MarkerPanel(**updates["marker_panel"])

        exp.updated_at = datetime.utcnow().isoformat()
        self._save(exp)
        return exp

    def delete_experiment(self, experiment_id: str) -> bool:
        path = self._path(experiment_id)
        if path.exists():
            path.unlink()
            return True
        return False

    # -- observations --------------------------------------------------------

    def add_observation(self, experiment_id: str, obs_data: dict) -> Optional[Observation]:
        exp = self._load(experiment_id)
        if exp is None:
            return None

        obs = Observation(
            id=self._new_id(),
            experiment_id=experiment_id,
            **obs_data,
        )
        exp.observations.append(obs)
        exp.updated_at = datetime.utcnow().isoformat()
        self._save(exp)
        return obs

    def delete_observation(self, experiment_id: str, observation_id: str) -> bool:
        exp = self._load(experiment_id)
        if exp is None:
            return False
        before = len(exp.observations)
        exp.observations = [o for o in exp.observations if o.id != observation_id]
        if len(exp.observations) < before:
            exp.updated_at = datetime.utcnow().isoformat()
            self._save(exp)
            return True
        return False


def _dict_to_experiment(data: dict) -> Experiment:
    """Reconstruct an Experiment from a JSON-loaded dict."""
    treatments = [TreatmentArm(**t) for t in data.get("treatments", [])]
    timepoints = [TimePoint(**tp) for tp in data.get("timepoints", [])]
    mp_data = data.get("marker_panel", {})
    marker_panel = MarkerPanel(**mp_data)
    observations = [Observation(**o) for o in data.get("observations", [])]

    return Experiment(
        id=data["id"],
        name=data["name"],
        cell_line=data["cell_line"],
        passage=data.get("passage", 0),
        description=data.get("description", ""),
        treatments=treatments,
        timepoints=timepoints,
        marker_panel=marker_panel,
        observations=observations,
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        tags=data.get("tags", []),
    )
