"""Training data collection for Cellpose fine-tuning.

Collects (image, corrected_mask) pairs from user edits for use as
training data in Cellpose model fine-tuning.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
import numpy as np


class TrainingDataCollector:
    """Manages training data directory for a session."""

    def __init__(self, session_dir: Path):
        self._dir = session_dir / "training_data"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._dir / "manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f)
        return {"pairs": [], "created_at": time.time()}

    def _save_manifest(self):
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def collect_pair(
        self,
        condition: str,
        base_name: str,
        image_path: str,
        mask: np.ndarray,
    ) -> bool:
        """Save a corrected (image, mask) pair for training.

        Args:
            condition: Condition name
            base_name: Image set base name
            image_path: Path to the raw intensity image
            mask: Corrected mask array

        Returns:
            True if pair was saved (new or updated)
        """
        pair_dir = self._dir / condition
        pair_dir.mkdir(parents=True, exist_ok=True)

        mask_path = pair_dir / f"{base_name}_mask.npy"
        np.save(mask_path, mask)

        # Update manifest
        key = f"{condition}/{base_name}"
        existing = next(
            (p for p in self._manifest["pairs"] if p["key"] == key), None
        )
        entry = {
            "key": key,
            "condition": condition,
            "base_name": base_name,
            "image_path": image_path,
            "mask_path": str(mask_path),
            "collected_at": time.time(),
        }

        if existing:
            self._manifest["pairs"] = [
                entry if p["key"] == key else p
                for p in self._manifest["pairs"]
            ]
        else:
            self._manifest["pairs"].append(entry)

        self._save_manifest()
        return True

    def remove_pair(self, condition: str, base_name: str) -> bool:
        """Remove a training pair."""
        key = f"{condition}/{base_name}"
        before = len(self._manifest["pairs"])
        self._manifest["pairs"] = [
            p for p in self._manifest["pairs"] if p["key"] != key
        ]
        self._save_manifest()

        # Remove mask file
        mask_path = self._dir / condition / f"{base_name}_mask.npy"
        if mask_path.exists():
            mask_path.unlink()

        return len(self._manifest["pairs"]) < before

    def get_pairs(self) -> List[dict]:
        """Get all collected training pairs."""
        return self._manifest["pairs"]

    def pair_count(self) -> int:
        return len(self._manifest["pairs"])

    def get_training_arrays(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load all training pairs as numpy arrays.

        Returns:
            (images, masks) lists of numpy arrays
        """
        from cellquant.core.io.image_loader import load_image

        images = []
        masks = []
        for pair in self._manifest["pairs"]:
            img_path = pair["image_path"]
            mask_path = pair["mask_path"]
            if Path(img_path).exists() and Path(mask_path).exists():
                images.append(load_image(img_path))
                masks.append(np.load(mask_path))
        return images, masks
