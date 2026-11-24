"""
Nellie organelle analysis wrapper.

Wraps nellie.run() to:
  - Accept a raw TIFF path + cell masks + output directory
  - Run the full nellie pipeline (filter → label → network → markers → track → hierarchy)
  - Aggregate per-organelle features up to per-cell rows that merge cleanly into
    the CellQuant results DataFrame (same Condition / ImageSet / CellID index)

The wrapper is intentionally thin.  All nellie-specific tuning (threshold,
remove_edges, etc.) flows through NellieParams so the task worker and API
schema stay decoupled from nellie internals.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Parameter dataclass ────────────────────────────────────────────────────────

@dataclass
class NellieParams:
    """Settings forwarded to nellie.run()."""

    channel_index: int = 0                      # which channel to analyse
    remove_edges: bool = False                  # remove border organelles
    otsu_thresh_intensity: bool = False         # Otsu instead of adaptive threshold
    threshold: Optional[float] = None           # manual threshold (None = auto)
    device: str = "auto"                         # "auto" | "cpu" | "gpu" (gpu/auto requires CuPy)
    low_memory: bool = False

    # Pixel / voxel sizes for non-OME TIFFs (µm).  None = let nellie detect.
    pixel_size_xy: Optional[float] = None       # µm per pixel (Y and X)
    pixel_size_z: Optional[float] = None        # µm per Z-slice (None for 2D)
    time_interval: Optional[float] = None       # seconds per frame (None for single)

    # Which hierarchy levels to include in the per-cell summary.
    # Choices: "organelle", "branch", "image"  (voxel/node are too granular)
    include_levels: list[str] = field(
        default_factory=lambda: ["organelle", "branch"]
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def run_nellie_on_image(
    image_path: str | Path,
    cell_masks: np.ndarray,
    output_dir: str | Path,
    condition: str,
    image_set: str,
    params: NellieParams,
) -> pd.DataFrame:
    """Run nellie on one field-of-view and return a per-cell DataFrame.

    Parameters
    ----------
    image_path:
        Path to a TIFF / ND2 file containing the organelle channel(s).
    cell_masks:
        2-D uint16 integer array where each non-zero value is a cell ID.
        Used to map organelles → parent cells.
    output_dir:
        Directory where nellie may write intermediate files and CSVs.
    condition, image_set:
        Metadata labels carried into the returned DataFrame.
    params:
        NellieParams controlling nellie behaviour.

    Returns
    -------
    pd.DataFrame with columns:
        Condition, ImageSet, CellID, <nellie feature columns ...>
    Rows are one per cell (organelle metrics aggregated as mean/std/count).
    Returns an empty DataFrame (with schema) if nellie finds no organelles.
    """
    from nellie.im_info.verifier import FileInfo
    from nellie.run import run as nellie_run

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build FileInfo ──────────────────────────────────────────────────────
    file_info = FileInfo(str(image_path), output_dir=str(output_dir))
    file_info.find_metadata()
    file_info.load_metadata()

    # Override voxel sizes if the image has no embedded metadata
    if params.pixel_size_xy is not None:
        file_info.change_dim_res("X", params.pixel_size_xy)
        file_info.change_dim_res("Y", params.pixel_size_xy)
    if params.pixel_size_z is not None:
        file_info.change_dim_res("Z", params.pixel_size_z)
    if params.time_interval is not None:
        file_info.change_dim_res("T", params.time_interval)

    # Select channel
    if "C" in (file_info.axes or ""):
        try:
            file_info.change_selected_channel(params.channel_index)
        except (IndexError, ValueError, KeyError):
            pass  # single-channel image — channel selection not needed

    if not file_info.good_dims or not file_info.good_axes:
        errs = file_info.get_validation_errors()
        raise ValueError(
            f"Nellie could not validate image metadata for {image_path.name}: "
            + "; ".join(errs)
        )

    # ── 2. Run nellie pipeline ─────────────────────────────────────────────────
    # Resolve device: gpu/auto requires CuPy; fall back to cpu if not available.
    device = params.device
    if device in ("gpu", "auto"):
        try:
            import cupy  # noqa: F401
        except ImportError:
            if device == "gpu":
                print("[Nellie] Warning: GPU requested but CuPy is not installed — falling back to CPU")
            device = "cpu"

    im_info = nellie_run(
        file_info,
        remove_edges=params.remove_edges,
        otsu_thresh_intensity=params.otsu_thresh_intensity,
        threshold=params.threshold,
        device=device,
        low_memory=params.low_memory,
    )

    # ── 3. Read output CSVs ────────────────────────────────────────────────────
    organelle_csv = im_info.pipeline_paths.get("features_organelles")
    branch_csv = im_info.pipeline_paths.get("features_branches")
    image_csv = im_info.pipeline_paths.get("features_image")

    level_dfs: dict[str, pd.DataFrame] = {}
    if "organelle" in params.include_levels and organelle_csv and os.path.exists(organelle_csv):
        level_dfs["organelle"] = pd.read_csv(organelle_csv)
    if "branch" in params.include_levels and branch_csv and os.path.exists(branch_csv):
        level_dfs["branch"] = pd.read_csv(branch_csv)
    if "image" in params.include_levels and image_csv and os.path.exists(image_csv):
        level_dfs["image"] = pd.read_csv(image_csv)

    if not level_dfs:
        return _empty_df(condition, image_set)

    # ── 4. Load organelle label mask produced by nellie ───────────────────────
    skel_relabelled_path = im_info.pipeline_paths.get("im_skel_relabelled")
    obj_label_path = im_info.pipeline_paths.get("im_instance_label")

    organelle_masks: Optional[np.ndarray] = None
    for candidate in (skel_relabelled_path, obj_label_path):
        if candidate and os.path.exists(candidate):
            try:
                import tifffile
                arr = tifffile.imread(candidate)
                # Collapse T / Z if present — take max-projection for 2-D assignment
                while arr.ndim > 2:
                    arr = arr.max(axis=0)
                organelle_masks = arr.astype(np.int32)
                break
            except Exception:
                pass

    # ── 5. Aggregate to per-cell ──────────────────────────────────────────────
    return _aggregate_to_cells(
        cell_masks=cell_masks,
        organelle_masks=organelle_masks,
        level_dfs=level_dfs,
        condition=condition,
        image_set=image_set,
    )


# ── Aggregation helpers ────────────────────────────────────────────────────────

def _aggregate_to_cells(
    cell_masks: np.ndarray,
    organelle_masks: Optional[np.ndarray],
    level_dfs: dict[str, pd.DataFrame],
    condition: str,
    image_set: str,
) -> pd.DataFrame:
    """
    Map each nellie organelle label to its parent cell via the cell_masks array,
    then aggregate per-organelle features into per-cell statistics.

    If organelle_masks is unavailable we fall back to image-level features only,
    assigning the same values to every cell in the image (marked with a flag).
    """
    cell_ids = sorted(int(i) for i in np.unique(cell_masks) if i != 0)
    if not cell_ids:
        return _empty_df(condition, image_set)

    rows = []

    # ── Case A: organelle-level mapping available ──────────────────────────────
    if organelle_masks is not None and "organelle" in level_dfs:
        org_df = level_dfs["organelle"]
        # Build organelle_label → cell_id mapping via spatial overlap
        org_to_cell = _map_organelles_to_cells(organelle_masks, cell_masks)

        # Collapse time dimension in DataFrame: take mean across t per label
        if "t" in org_df.columns:
            org_df = org_df.groupby("label", as_index=False).mean(numeric_only=True)

        # Prefix organelle column names
        feature_cols = [c for c in org_df.columns if c not in ("label", "t")]
        org_df = org_df.rename(columns={c: f"nellie_org_{c}" for c in feature_cols})

        # Add cell assignment
        org_df["CellID"] = org_df["label"].map(org_to_cell)
        org_df = org_df.dropna(subset=["CellID"])
        org_df["CellID"] = org_df["CellID"].astype(int)

        prefixed_feature_cols = [f"nellie_org_{c}" for c in feature_cols]

        for cell_id in cell_ids:
            cell_orgs = org_df[org_df["CellID"] == cell_id]
            row: dict = {"Condition": condition, "ImageSet": image_set, "CellID": cell_id}
            row["nellie_n_organelles"] = len(cell_orgs)

            if len(cell_orgs) > 0:
                for col in prefixed_feature_cols:
                    vals = cell_orgs[col].dropna()
                    if len(vals) > 0:
                        row[f"{col}_mean"] = float(vals.mean())
                        row[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
                        row[f"{col}_sum"] = float(vals.sum())
                    else:
                        row[f"{col}_mean"] = np.nan
                        row[f"{col}_std"] = np.nan
                        row[f"{col}_sum"] = np.nan
            else:
                for col in prefixed_feature_cols:
                    row[f"{col}_mean"] = np.nan
                    row[f"{col}_std"] = np.nan
                    row[f"{col}_sum"] = np.nan

            # Append branch-level aggregates if available
            if "branch" in level_dfs:
                row.update(
                    _image_level_features(level_dfs["branch"], prefix="nellie_branch_")
                )

            rows.append(row)

    # ── Case B: image-level only ───────────────────────────────────────────────
    else:
        img_features: dict = {}
        if "image" in level_dfs:
            img_features = _image_level_features(level_dfs["image"], prefix="nellie_img_")
        elif "branch" in level_dfs:
            img_features = _image_level_features(level_dfs["branch"], prefix="nellie_branch_")

        for cell_id in cell_ids:
            row = {"Condition": condition, "ImageSet": image_set, "CellID": cell_id,
                   "nellie_n_organelles": np.nan, "nellie_cell_level": "image_fallback"}
            row.update(img_features)
            rows.append(row)

    if not rows:
        return _empty_df(condition, image_set)

    return pd.DataFrame(rows)


def _map_organelles_to_cells(
    organelle_masks: np.ndarray,
    cell_masks: np.ndarray,
) -> dict[int, int]:
    """Return {organelle_label: cell_id} by max-overlap assignment (vectorized)."""
    # Resize organelle mask to match cell mask if shapes differ
    if organelle_masks.shape != cell_masks.shape:
        from skimage.transform import resize
        organelle_masks = resize(
            organelle_masks,
            cell_masks.shape,
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.int32)

    # Keep only pixels where both masks are non-zero
    flat_org = organelle_masks.ravel()
    flat_cell = cell_masks.ravel()
    valid = (flat_org != 0) & (flat_cell != 0)
    org_px = flat_org[valid]
    cell_px = flat_cell[valid]

    if len(org_px) == 0:
        return {}

    # Build a 2-D co-occurrence count matrix: rows=organelle labels, cols=cell IDs
    n_org = int(org_px.max()) + 1
    n_cell = int(cell_px.max()) + 1
    # Encode pair as a single integer and use bincount for speed
    pair_idx = org_px.astype(np.int64) * n_cell + cell_px.astype(np.int64)
    counts = np.bincount(pair_idx, minlength=n_org * n_cell).reshape(n_org, n_cell)

    # For each organelle label, pick the cell with the highest overlap
    mapping: dict[int, int] = {}
    org_labels = np.unique(org_px)
    best_cells = counts[org_labels].argmax(axis=1)
    for org_label, cell_id in zip(org_labels.tolist(), best_cells.tolist()):
        if cell_id != 0:
            mapping[int(org_label)] = int(cell_id)

    return mapping


def _image_level_features(df: pd.DataFrame, prefix: str) -> dict:
    """Flatten an image-level or branch-level DataFrame into a flat dict."""
    if df.empty:
        return {}
    # Collapse time: use mean
    num_df = df.select_dtypes(include="number")
    skip = {"t", "label"}
    cols = [c for c in num_df.columns if c not in skip]
    mean_row = num_df[cols].mean()
    return {f"{prefix}{c}": float(v) for c, v in mean_row.items()}


def _empty_df(condition: str, image_set: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["Condition", "ImageSet", "CellID", "nellie_n_organelles"])
