"""
Cohort data manager — handles import, storage, and demographic analysis.

Supports CSV/XLSX import with flexible column mapping.  All data is
held in-memory via pandas DataFrames and indexed by cohort_id.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cellquant.bharat.aging_clock import (
    BIOMARKER_REFS,
    AgingClockPrediction,
    compute_cohort_aging_stats,
    predict_biological_age,
)

# Column name aliases the importer will recognise
_COLUMN_ALIASES: dict[str, list[str]] = {
    "subject_id":        ["subject_id", "id", "participant_id", "pid", "sample_id"],
    "age":               ["age", "age_years", "chronological_age"],
    "sex":               ["sex", "gender"],
    "state":             ["state", "indian_state", "province", "region"],
    "district":          ["district", "city"],
    "hba1c":             ["hba1c", "hb_a1c", "glycated_hb"],
    "fasting_glucose":   ["fasting_glucose", "fbg", "fbs", "fasting_blood_glucose"],
    "total_cholesterol": ["total_cholesterol", "tc", "cholesterol"],
    "hdl":               ["hdl", "hdl_cholesterol", "hdl_c"],
    "ldl":               ["ldl", "ldl_cholesterol", "ldl_c"],
    "triglycerides":     ["triglycerides", "tg", "trigs"],
    "creatinine":        ["creatinine", "serum_creatinine", "scr"],
    "egfr":              ["egfr", "gfr", "estimated_gfr"],
    "albumin":           ["albumin", "serum_albumin", "alb"],
    "crp":               ["crp", "c_reactive_protein", "hs_crp"],
    "hemoglobin":        ["hemoglobin", "hb", "hgb"],
    "wbc_count":         ["wbc_count", "wbc", "white_blood_cells", "leukocytes"],
    "platelet_count":    ["platelet_count", "platelets", "plt"],
    "systolic_bp":       ["systolic_bp", "sbp", "sys_bp"],
    "diastolic_bp":      ["diastolic_bp", "dbp", "dia_bp"],
    "bmi":               ["bmi", "body_mass_index"],
    "waist_circumference": ["waist_circumference", "waist", "waist_cm"],
    "vitamin_d":         ["vitamin_d", "vit_d", "25_oh_d"],
    "tsh":               ["tsh", "thyroid_stimulating_hormone"],
    "uric_acid":         ["uric_acid", "ua", "serum_uric_acid"],
    "ast":               ["ast", "sgot", "aspartate_aminotransferase"],
    "alt":               ["alt", "sgpt", "alanine_aminotransferase"],
}


class CohortStore:
    """In-memory store for imported cohort data."""

    def __init__(self) -> None:
        self._cohorts: dict[str, pd.DataFrame] = {}
        self._predictions: dict[str, list[AgingClockPrediction]] = {}
        self._names: dict[str, str] = {}

    # ── Import ────────────────────────────────────────────

    def import_file(self, file_path: str, name: Optional[str] = None) -> str:
        """Import a CSV or XLSX file and return the cohort_id."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if p.suffix.lower() in (".xlsx", ".xls"):
            raw = pd.read_excel(p)
        else:
            raw = pd.read_csv(p)

        df = self._normalise_columns(raw)
        cohort_id = uuid.uuid4().hex[:12]
        self._cohorts[cohort_id] = df
        self._names[cohort_id] = name or p.stem
        return cohort_id

    def _normalise_columns(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map raw column names to canonical names."""
        lower_cols = {c.lower().strip(): c for c in raw.columns}
        mapping: dict[str, str] = {}
        for canonical, aliases in _COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in lower_cols:
                    mapping[lower_cols[alias]] = canonical
                    break
        df = raw.rename(columns=mapping)
        # Ensure subject_id exists
        if "subject_id" not in df.columns:
            df["subject_id"] = [f"S{i:05d}" for i in range(len(df))]
        # Normalise sex
        if "sex" in df.columns:
            df["sex"] = df["sex"].astype(str).str.lower().str.strip()
            df["sex"] = df["sex"].replace({"m": "male", "f": "female", "1": "male", "0": "female"})
        return df

    # ── Getters ───────────────────────────────────────────

    def get_df(self, cohort_id: str) -> pd.DataFrame:
        if cohort_id not in self._cohorts:
            raise KeyError(f"Unknown cohort: {cohort_id}")
        return self._cohorts[cohort_id]

    def list_cohorts(self) -> list[dict]:
        result = []
        for cid, df in self._cohorts.items():
            result.append({
                "cohort_id": cid,
                "name": self._names.get(cid, ""),
                "n_subjects": len(df),
            })
        return result

    # ── Demographics ──────────────────────────────────────

    def demographics(self, cohort_id: str) -> dict:
        df = self.get_df(cohort_id)
        total = len(df)

        by_sex = {}
        if "sex" in df.columns:
            by_sex = df["sex"].value_counts().to_dict()

        by_age_group: dict[str, int] = {}
        age_dist: list[dict] = []
        if "age" in df.columns:
            bins = [0, 18, 30, 40, 50, 60, 70, 80, 120]
            labels = ["<18", "18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
            df["_age_grp"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
            by_age_group = df["_age_grp"].value_counts().sort_index().to_dict()

            # Population pyramid bins
            pyramid_bins = list(range(0, 101, 5))
            for i in range(len(pyramid_bins) - 1):
                lo, hi = pyramid_bins[i], pyramid_bins[i + 1]
                mask = (df["age"] >= lo) & (df["age"] < hi)
                male = int(((df["sex"] == "male") & mask).sum()) if "sex" in df.columns else 0
                female = int(((df["sex"] == "female") & mask).sum()) if "sex" in df.columns else 0
                age_dist.append({"bin_start": lo, "bin_end": hi, "male": male, "female": female})

            df.drop(columns=["_age_grp"], inplace=True, errors="ignore")

        by_state: list[dict] = []
        if "state" in df.columns:
            state_counts = df["state"].dropna().value_counts()
            for state, count in state_counts.items():
                by_state.append({
                    "state": state,
                    "count": int(count),
                    "pct": round(count / total * 100, 1),
                })

        bmi_dist: list[dict] = []
        if "bmi" in df.columns:
            bmi_vals = df["bmi"].dropna()
            categories = [
                ("Underweight (<18.5)", bmi_vals < 18.5),
                ("Normal (18.5–22.9)", (bmi_vals >= 18.5) & (bmi_vals < 23)),
                ("Overweight (23–24.9)", (bmi_vals >= 23) & (bmi_vals < 25)),
                ("Obese (≥25)", bmi_vals >= 25),
            ]
            for cat_name, mask in categories:
                n = int(mask.sum())
                bmi_dist.append({
                    "category": cat_name,
                    "count": n,
                    "pct": round(n / max(len(bmi_vals), 1) * 100, 1),
                })

        return {
            "total": total,
            "by_sex": by_sex,
            "by_age_group": by_age_group,
            "by_state": by_state,
            "age_distribution": age_dist,
            "bmi_distribution": bmi_dist,
        }

    # ── Biomarker Profile ─────────────────────────────────

    def biomarker_profile(self, cohort_id: str) -> list[dict]:
        df = self.get_df(cohort_id)
        results = []
        for key, ref in BIOMARKER_REFS.items():
            if key not in df.columns:
                continue
            vals = pd.to_numeric(df[key], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            arr = vals.values
            n_abnormal = int(((arr < ref["low"]) | (arr > ref["high"])).sum())
            results.append({
                "name": key,
                "display_name": ref["display"],
                "unit": ref["unit"],
                "n_available": len(vals),
                "mean": round(float(np.mean(arr)), 2),
                "median": round(float(np.median(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "p5": round(float(np.percentile(arr, 5)), 2),
                "p25": round(float(np.percentile(arr, 25)), 2),
                "p75": round(float(np.percentile(arr, 75)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "reference_low": ref["low"],
                "reference_high": ref["high"],
                "pct_abnormal": round(n_abnormal / len(vals) * 100, 1),
            })
        return results

    # ── AgingClock ────────────────────────────────────────

    def run_aging_clock(self, cohort_id: str) -> dict:
        """Run AgingClock India on every subject in the cohort."""
        df = self.get_df(cohort_id)
        predictions: list[AgingClockPrediction] = []
        skipped = 0

        for _, row in df.iterrows():
            biomarker_dict = {}
            for key in BIOMARKER_REFS:
                if key in row.index:
                    val = row[key]
                    biomarker_dict[key] = float(val) if pd.notna(val) else None

            age = int(row["age"]) if "age" in row.index and pd.notna(row["age"]) else None
            sex = str(row["sex"]) if "sex" in row.index and pd.notna(row["sex"]) else "male"
            sid = str(row["subject_id"]) if "subject_id" in row.index else f"S{_}"

            if age is None:
                skipped += 1
                continue

            pred = predict_biological_age(sid, age, sex, biomarker_dict)
            if pred is None:
                skipped += 1
            else:
                predictions.append(pred)

        self._predictions[cohort_id] = predictions

        # Population stats
        stats = compute_cohort_aging_stats(predictions)
        stats["n_skipped"] = skipped

        # By-sex breakdown
        by_sex: dict[str, dict] = {}
        for sex_val in ("male", "female"):
            sex_preds = [p for p in predictions
                         if df.loc[df["subject_id"] == p.subject_id, "sex"].iloc[0] == sex_val]
            if sex_preds:
                gaps = [p.age_gap for p in sex_preds]
                by_sex[sex_val] = {
                    "n": len(sex_preds),
                    "mean_age_gap": round(float(np.mean(gaps)), 2),
                    "median_age_gap": round(float(np.median(gaps)), 2),
                }
        stats["by_sex"] = by_sex

        # By age-group
        by_age_group: dict[str, dict] = {}
        age_groups = [("18-30", 18, 30), ("31-50", 31, 50), ("51-65", 51, 65), ("65+", 65, 200)]
        for label, lo, hi in age_groups:
            grp = [p for p in predictions if lo <= p.chronological_age < hi]
            if grp:
                gaps = [p.age_gap for p in grp]
                bio = [p.biological_age for p in grp]
                by_age_group[label] = {
                    "age_group": label,
                    "n": len(grp),
                    "mean_age_gap": round(float(np.mean(gaps)), 2),
                    "mean_bio_age": round(float(np.mean(bio)), 1),
                }
        stats["by_age_group"] = by_age_group

        # By state
        by_state: list[dict] = []
        if "state" in df.columns:
            pred_ids = {p.subject_id for p in predictions}
            pred_df = df[df["subject_id"].isin(pred_ids)].copy()
            pred_map = {p.subject_id: p for p in predictions}
            pred_df["bio_age"] = pred_df["subject_id"].map(lambda s: pred_map[s].biological_age)
            pred_df["age_gap"] = pred_df["subject_id"].map(lambda s: pred_map[s].age_gap)

            for state, grp in pred_df.groupby("state"):
                by_state.append({
                    "state": str(state),
                    "n": len(grp),
                    "mean_age_gap": round(float(grp["age_gap"].mean()), 2),
                    "mean_bio_age": round(float(grp["bio_age"].mean()), 1),
                    "mean_chrono_age": round(float(grp["age"].mean()), 1),
                })
        stats["by_state"] = by_state

        return stats

    def get_predictions(self, cohort_id: str) -> list[AgingClockPrediction]:
        return self._predictions.get(cohort_id, [])

    def get_subject_prediction(
        self, cohort_id: str, subject_id: str
    ) -> Optional[AgingClockPrediction]:
        for p in self._predictions.get(cohort_id, []):
            if p.subject_id == subject_id:
                return p
        return None


# Singleton
_store: Optional[CohortStore] = None


def get_cohort_store() -> CohortStore:
    global _store
    if _store is None:
        _store = CohortStore()
    return _store
