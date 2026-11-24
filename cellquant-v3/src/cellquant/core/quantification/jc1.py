"""
JC-1 mitochondrial membrane potential analysis.

JC-1 (5,5',6,6'-tetrachloro-1,1',3,3'-tetraethylbenzimidazolylcarbocyanine iodide)
is a cationic dye that accumulates in mitochondria in proportion to the membrane
potential (ΔΨm).  At high ΔΨm the dye forms J-aggregates with red fluorescence
(~590 nm); at low ΔΨm it remains as monomers with green fluorescence (~529 nm).

Primary metric:  JC1_Ratio = Red_CTCF / Green_CTCF
  - Higher ratio → healthy mitochondria (high ΔΨm)
  - Lower ratio  → depolarised / unhealthy mitochondria (low ΔΨm)

CRITICAL: Zero and negative CTCF values in either channel MUST be removed before
computing the ratio.  Division by zero / negative denominators produce nonsensical
ratios that cannot be recovered downstream.  This is not optional.

Ported from: _archive/jc-1.py (standalone analysis script)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class JC1Config:
    """Configuration for JC-1 ratio analysis.

    Args:
        red_marker: Name of the aggregate (red/RFP) marker channel.
            Must match a key in the CTCF results (e.g. "mRFP", "JC1_Red").
        green_marker: Name of the monomer (green/GFP) marker channel.
            Must match a key in the CTCF results (e.g. "GFP", "JC1_Green").
        min_area: Minimum cell area (pixels) to include.  Cells below this
            are typically debris. Default 50 matches the original analysis script.
        require_positive: If True (default, mandatory), remove any cell where
            either channel CTCF is <= 0 before computing the ratio.
        red_ctcf_col: DataFrame column for red CTCF.  Auto-derived from
            ``red_marker`` if not specified.
        green_ctcf_col: DataFrame column for green CTCF.  Auto-derived from
            ``green_marker`` if not specified.
    """

    red_marker: str = "JC1_Red"
    green_marker: str = "JC1_Green"
    min_area: int = 50
    require_positive: bool = True
    red_ctcf_col: Optional[str] = None
    green_ctcf_col: Optional[str] = None

    def __post_init__(self):
        if self.red_ctcf_col is None:
            self.red_ctcf_col = f"{self.red_marker}_CTCF"
        if self.green_ctcf_col is None:
            self.green_ctcf_col = f"{self.green_marker}_CTCF"


# ── Core calculation ───────────────────────────────────────────────


def calculate_jc1_ratio(
    red_ctcf: np.ndarray,
    green_ctcf: np.ndarray,
) -> np.ndarray:
    """Compute per-cell JC-1 Red/Green ratio.

    Returns NaN where green_ctcf <= 0 (avoids inf/division-by-zero).
    Callers should run ``filter_jc1_cells`` first to remove those cells.

    Args:
        red_ctcf: Array of red-channel CTCF values.
        green_ctcf: Array of green-channel CTCF values.

    Returns:
        Float array of ratios, same length as inputs.
    """
    red = np.asarray(red_ctcf, dtype=np.float64)
    green = np.asarray(green_ctcf, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(green > 0, red / green, np.nan)

    return ratio


# ── Filtering ─────────────────────────────────────────────────────


def filter_jc1_cells(
    df: pd.DataFrame,
    config: JC1Config,
) -> Tuple[pd.DataFrame, int]:
    """Remove cells that are invalid for JC-1 ratio analysis.

    Applies two mandatory filters:
      1. Area < min_area  (debris / fragments)
      2. Either channel CTCF <= 0  (log-scale breaks; ratio undefined)

    Args:
        df: Cell-level results DataFrame (must contain the CTCF columns
            specified in ``config``).
        config: JC-1 configuration.

    Returns:
        Tuple of (filtered DataFrame, number of cells removed).
    """
    n_before = len(df)
    mask = pd.Series(True, index=df.index)

    # Area filter
    if "Area" in df.columns:
        mask &= df["Area"] >= config.min_area

    # Zero / negative CTCF filter (mandatory for ratio calculation)
    if config.require_positive:
        if config.red_ctcf_col in df.columns:
            mask &= df[config.red_ctcf_col] > 0
        if config.green_ctcf_col in df.columns:
            mask &= df[config.green_ctcf_col] > 0

    filtered = df[mask].copy()
    n_removed = n_before - len(filtered)
    return filtered, n_removed


# ── DataFrame integration ──────────────────────────────────────────


def add_jc1_ratio_column(
    df: pd.DataFrame,
    config: JC1Config,
) -> pd.DataFrame:
    """Compute JC-1 ratio and append it as a column to the DataFrame.

    Performs mandatory cell filtering first (zero/negative CTCF removal),
    then computes ``JC1_Ratio = red_CTCF / green_CTCF`` for remaining cells.

    The returned DataFrame has fewer rows than the input when cells are
    filtered out.  A ``jc1_valid`` boolean column is added to indicate
    which cells passed filtering (useful for auditing in the frontend).

    Args:
        df: Cell-level results DataFrame from ``results_to_dataframe``.
        config: JC-1 configuration.

    Returns:
        DataFrame with added columns: ``JC1_Ratio``, ``jc1_valid``.
        Rows with invalid JC-1 values are dropped.
    """
    if config.red_ctcf_col not in df.columns:
        raise KeyError(
            f"Red CTCF column '{config.red_ctcf_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    if config.green_ctcf_col not in df.columns:
        raise KeyError(
            f"Green CTCF column '{config.green_ctcf_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    filtered_df, n_removed = filter_jc1_cells(df, config)

    if n_removed > 0:
        print(
            f"  JC-1: removed {n_removed} cells with zero/negative CTCF or "
            f"area < {config.min_area}px (mandatory for ratio calculation)"
        )

    if len(filtered_df) == 0:
        print("  JC-1: Warning — no valid cells remain after filtering")
        filtered_df["JC1_Ratio"] = pd.Series(dtype=np.float64)
        filtered_df["jc1_valid"] = pd.Series(dtype=bool)
        return filtered_df

    filtered_df["JC1_Ratio"] = calculate_jc1_ratio(
        filtered_df[config.red_ctcf_col].values,
        filtered_df[config.green_ctcf_col].values,
    )
    filtered_df["jc1_valid"] = True

    return filtered_df


# ── Summary statistics ─────────────────────────────────────────────


def jc1_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-condition JC-1 ratio summary statistics.

    Args:
        df: Cell-level DataFrame with ``JC1_Ratio`` and ``Condition`` columns.

    Returns:
        Summary DataFrame with one row per condition:
        n_cells, mean_ratio, median_ratio, std_ratio, sem_ratio, cv_percent.
    """
    if "JC1_Ratio" not in df.columns or "Condition" not in df.columns:
        return pd.DataFrame()

    rows = []
    for cond, grp in df.groupby("Condition"):
        ratios = grp["JC1_Ratio"].dropna().values
        n = len(ratios)
        if n == 0:
            continue
        mean_r = float(np.mean(ratios))
        std_r = float(np.std(ratios, ddof=1)) if n > 1 else 0.0
        rows.append({
            "Condition": cond,
            "n_cells": n,
            "mean_ratio": round(mean_r, 4),
            "median_ratio": round(float(np.median(ratios)), 4),
            "std_ratio": round(std_r, 4),
            "sem_ratio": round(std_r / np.sqrt(n), 4) if n > 0 else 0.0,
            "cv_percent": round(std_r / mean_r * 100, 1) if mean_r > 0 else 0.0,
        })

    return pd.DataFrame(rows)
