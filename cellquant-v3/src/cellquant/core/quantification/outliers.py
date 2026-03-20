"""
Outlier detection and hierarchical statistical aggregation.

Implements MAD-based modified Z-score (Iglewicz & Hoaglin) for robust
outlier flagging, per-FOV aggregation, and the proper cells → FOVs →
conditions summary hierarchy described in the CTCF specification.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


# ── MAD-based outlier detection ────────────────────────────────────


def mad_outlier_detection(
    values: np.ndarray,
    threshold: float = 3.5,
) -> np.ndarray:
    """Flag outliers using the Modified Z-Score.

    Modified Z_i = 0.6745 · (x_i − median) / MAD

    The constant 0.6745 makes MAD consistent with the standard
    deviation for Gaussian data.  A threshold of 3.5 is recommended
    by Iglewicz & Hoaglin.

    Args:
        values: 1-D numeric array.
        threshold: Modified Z-score cut-off (default 3.5).

    Returns:
        Boolean array (``True`` = outlier).
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return np.zeros(len(values), dtype=bool)

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad < 1e-12:
        # MAD is zero — all values are essentially identical.
        return np.zeros(len(values), dtype=bool)

    modified_z = 0.6745 * (values - median) / mad
    return np.abs(modified_z) > threshold


def flag_outliers_in_dataframe(
    df: pd.DataFrame,
    ctcf_columns: Optional[List[str]] = None,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Add ``is_outlier_*`` flag columns to a results DataFrame.

    Applies MAD outlier detection **per condition** to:
    - ``Area`` → ``is_outlier_area``
    - Each CTCF column → ``is_outlier_{marker}``

    Args:
        df: Cell-level results DataFrame (must contain ``Condition``
            and ``Area`` columns).
        ctcf_columns: CTCF columns to check.  Auto-detected from
            columns ending with ``_CTCF`` if ``None``.
        threshold: MAD threshold.

    Returns:
        DataFrame with added boolean flag columns (in-place).
    """
    if len(df) == 0:
        return df

    if ctcf_columns is None:
        ctcf_columns = [c for c in df.columns if c.endswith("_CTCF")]

    # Area outliers (per condition)
    df["is_outlier_area"] = False
    for cond, grp in df.groupby("Condition"):
        idx = grp.index
        df.loc[idx, "is_outlier_area"] = mad_outlier_detection(
            grp["Area"].values, threshold
        )

    # Intensity outliers (per condition per marker)
    for col in ctcf_columns:
        marker = col.replace("_CTCF", "")
        flag_col = f"is_outlier_{marker}"
        df[flag_col] = False
        for cond, grp in df.groupby("Condition"):
            idx = grp.index
            df.loc[idx, flag_col] = mad_outlier_detection(
                grp[col].values, threshold
            )

    return df


# ── Per-FOV aggregation ────────────────────────────────────────────


def per_fov_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cell-level results to per-FOV (ImageSet) level.

    Groups by ``Condition`` × ``ImageSet`` and computes:
    - ``n_cells``: cell count
    - ``median_area``: median cell area
    - For each ``*_CTCF`` column: ``{marker}_median_CTCF``,
      ``{marker}_mean_CTCF``

    Returns a DataFrame with one row per FOV.
    """
    if len(df) == 0:
        return pd.DataFrame()

    ctcf_cols = [c for c in df.columns if c.endswith("_CTCF")]

    agg_dict = {"Area": ["count", "median"]}
    for col in ctcf_cols:
        agg_dict[col] = ["median", "mean"]

    grouped = df.groupby(["Condition", "ImageSet"]).agg(agg_dict)
    grouped.columns = [
        "n_cells" if (a, b) == ("Area", "count")
        else "median_area" if (a, b) == ("Area", "median")
        else f"{a.replace('_CTCF', '')}_median_CTCF" if b == "median"
        else f"{a.replace('_CTCF', '')}_mean_CTCF"
        for a, b in grouped.columns
    ]
    return grouped.reset_index()


# ── Hierarchical summary ──────────────────────────────────────────


def hierarchical_summary(
    df: pd.DataFrame,
    fov_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Produce the cells → FOVs → conditions summary.

    For each condition and each CTCF marker:
    - **N_cells**: total cell count
    - **N_FOVs**: number of image sets
    - **mean_of_fov_medians**: mean of per-FOV median CTCF values
    - **sd_of_fov_medians**: standard deviation of per-FOV medians
    - **cv_percent**: coefficient of variation (SD / Mean × 100)
    - **sem**: standard error (SD / √N_FOVs)

    This avoids pseudoreplication: the statistical unit is the FOV,
    not the individual cell.

    Args:
        df: Cell-level results DataFrame.
        fov_df: Pre-computed per-FOV table (computed if ``None``).

    Returns:
        Summary DataFrame with one row per condition × marker.
    """
    if len(df) == 0:
        return pd.DataFrame()

    if fov_df is None:
        fov_df = per_fov_aggregation(df)

    ctcf_cols = [c for c in df.columns if c.endswith("_CTCF")]
    rows = []

    for cond, grp in fov_df.groupby("Condition"):
        n_fovs = len(grp)
        n_cells = int(grp["n_cells"].sum())

        for ctcf_col in ctcf_cols:
            marker = ctcf_col.replace("_CTCF", "")
            med_col = f"{marker}_median_CTCF"
            if med_col not in grp.columns:
                continue

            medians = grp[med_col].values.astype(np.float64)
            mean_val = float(np.mean(medians))
            sd_val = float(np.std(medians, ddof=1)) if n_fovs > 1 else 0.0
            cv_pct = (sd_val / mean_val * 100) if mean_val > 0 else 0.0
            sem = sd_val / np.sqrt(n_fovs) if n_fovs > 0 else 0.0

            rows.append(
                {
                    "Condition": cond,
                    "Marker": marker,
                    "N_cells": n_cells,
                    "N_FOVs": n_fovs,
                    "mean_of_fov_medians": round(mean_val, 2),
                    "sd_of_fov_medians": round(sd_val, 2),
                    "cv_percent": round(cv_pct, 1),
                    "sem": round(sem, 2),
                }
            )

    return pd.DataFrame(rows)
