"""Per-session state model, replacing the v2 global AppState."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import json
import time


@dataclass
class Session:
    """Isolated state for a single user session."""

    id: str
    directory: Path
    created_at: float = field(default_factory=time.time)

    # Experiment state
    experiment_path: Optional[Path] = None
    conditions: Dict[str, dict] = field(default_factory=dict)

    # Masks: {condition_name: {base_name: np.ndarray}}
    masks: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    # Tracking: {condition_name: {base_name: np.ndarray}} (tracked masks)
    tracked_masks: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    track_graphs: Dict[str, dict] = field(default_factory=dict)

    # Results
    results_df: Optional[pd.DataFrame] = None

    def get_tile_dir(self, condition: str, base_name: str, channel: str) -> Path:
        p = self.directory / "tiles" / condition / base_name / channel
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_mask_path(self, condition: str, base_name: str) -> Path:
        p = self.directory / "masks" / condition
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{base_name}_masks.npy"

    def get_thumbnail_path(self, condition: str, base_name: str, channel: str) -> Path:
        p = self.directory / "thumbnails" / condition
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{base_name}_{channel}.jpg"

    def get_results_path(self) -> Path:
        p = self.directory / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p / "ctcf_results.parquet"

    def get_export_dir(self) -> Path:
        p = self.directory / "exports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_state(self):
        """Persist session metadata to JSON for recovery."""
        state = {
            "id": self.id,
            "created_at": self.created_at,
            "experiment_path": str(self.experiment_path) if self.experiment_path else None,
            "conditions": {
                k: {
                    "name": v.get("name", k),
                    "path": str(v.get("path", "")),
                    "n_images": v.get("n_images", 0),
                }
                for k, v in self.conditions.items()
            },
        }
        with open(self.directory / "session.json", "w") as f:
            json.dump(state, f, indent=2)

    def save_results(self):
        """Save results DataFrame as Parquet."""
        if self.results_df is not None and len(self.results_df) > 0:
            self.results_df.to_parquet(self.get_results_path(), index=False)

    def load_results(self) -> Optional[pd.DataFrame]:
        """Load results from Parquet if available."""
        path = self.get_results_path()
        if path.exists():
            self.results_df = pd.read_parquet(path)
            return self.results_df
        return None
