"""Undo system for mask editing with original mask tracking."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import time
import numpy as np


MAX_UNDO_DEPTH = 20


@dataclass
class MaskEdit:
    """A single mask edit operation snapshot."""
    timestamp: float
    operation: str
    condition: str
    base_name: str
    previous_masks: np.ndarray


class EditHistory:
    """Per-image undo stack with original mask preservation."""

    def __init__(self, session_dir: Path):
        self._session_dir = session_dir
        # {(condition, base_name): [MaskEdit, ...]}
        self._stacks: Dict[tuple, List[MaskEdit]] = {}
        # Track which images have been edited
        self._edited: set = set()

    def push(self, condition: str, base_name: str, operation: str,
             current_masks: np.ndarray):
        """Save snapshot before a mutation. Call BEFORE applying the edit."""
        key = (condition, base_name)

        # On first edit, save original mask
        if key not in self._edited:
            self._save_original(condition, base_name, current_masks)
            self._edited.add(key)

        if key not in self._stacks:
            self._stacks[key] = []

        stack = self._stacks[key]
        stack.append(MaskEdit(
            timestamp=time.time(),
            operation=operation,
            condition=condition,
            base_name=base_name,
            previous_masks=current_masks.copy(),
        ))

        # Trim to max depth
        if len(stack) > MAX_UNDO_DEPTH:
            self._stacks[key] = stack[-MAX_UNDO_DEPTH:]

    def pop(self, condition: str, base_name: str) -> Optional[np.ndarray]:
        """Undo the last edit. Returns the previous mask state, or None."""
        key = (condition, base_name)
        stack = self._stacks.get(key, [])
        if not stack:
            return None
        edit = stack.pop()
        return edit.previous_masks

    def can_undo(self, condition: str, base_name: str) -> bool:
        key = (condition, base_name)
        return bool(self._stacks.get(key))

    def undo_depth(self, condition: str, base_name: str) -> int:
        key = (condition, base_name)
        return len(self._stacks.get(key, []))

    def is_edited(self, condition: str, base_name: str) -> bool:
        return (condition, base_name) in self._edited

    def edited_images(self) -> List[dict]:
        """Return list of all edited images."""
        return [
            {"condition": c, "base_name": b}
            for c, b in sorted(self._edited)
        ]

    def _save_original(self, condition: str, base_name: str,
                       masks: np.ndarray):
        """Save original mask before first edit."""
        original_dir = self._session_dir / "masks" / condition
        original_dir.mkdir(parents=True, exist_ok=True)
        original_path = original_dir / f"{base_name}_masks_original.npy"
        if not original_path.exists():
            np.save(original_path, masks)

    def get_original_path(self, condition: str, base_name: str) -> Path:
        return self._session_dir / "masks" / condition / f"{base_name}_masks_original.npy"
