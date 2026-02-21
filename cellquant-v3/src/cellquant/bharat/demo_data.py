"""
Generate synthetic demo data for the BHARAT Cohort Analytics portal.

Creates a realistic cohort of Indian subjects with age/sex/state-appropriate
biomarker distributions.  This allows the portal to display meaningful
visualisations without requiring real patient data.
"""

from __future__ import annotations

import io
import random
from typing import Optional

import numpy as np
import pandas as pd

# Indian states weighted by approximate population share
_STATE_WEIGHTS: list[tuple[str, float]] = [
    ("Uttar Pradesh", 0.17), ("Maharashtra", 0.09), ("Bihar", 0.09),
    ("West Bengal", 0.07), ("Madhya Pradesh", 0.06), ("Tamil Nadu", 0.06),
    ("Rajasthan", 0.06), ("Karnataka", 0.05), ("Gujarat", 0.05),
    ("Andhra Pradesh", 0.04), ("Odisha", 0.04), ("Telangana", 0.03),
    ("Kerala", 0.03), ("Jharkhand", 0.03), ("Assam", 0.03),
    ("Punjab", 0.02), ("Chhattisgarh", 0.02), ("Haryana", 0.02),
    ("Delhi", 0.02), ("Uttarakhand", 0.01), ("Himachal Pradesh", 0.005),
    ("Goa", 0.002), ("Sikkim", 0.001),
]


def _pick_state(rng: random.Random) -> str:
    states, weights = zip(*_STATE_WEIGHTS)
    return rng.choices(states, weights=weights, k=1)[0]


def _age_adjusted(base: float, std: float, age: int, age_coeff: float,
                  rng: np.random.Generator) -> float:
    """Sample a biomarker value adjusted for age."""
    return base + age_coeff * (age - 40) + rng.normal(0, std)


def generate_demo_cohort(
    n: int = 500,
    seed: int = 42,
    name: str = "BHARAT Demo Cohort",
) -> pd.DataFrame:
    """Return a DataFrame with n synthetic Indian subjects."""
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    rows: list[dict] = []
    for i in range(n):
        age = int(rng_np.integers(20, 85))
        sex = rng_py.choice(["male", "female"])
        state = _pick_state(rng_py)
        is_male = sex == "male"

        # Build biomarker profile — values are age/sex adjusted
        row: dict = {
            "subject_id": f"BH{i + 1:05d}",
            "age": age,
            "sex": sex,
            "state": state,
        }

        row["hba1c"] = round(max(3.5, _age_adjusted(5.3, 0.5, age, 0.015, rng_np)), 1)
        row["fasting_glucose"] = round(max(50, _age_adjusted(90, 12, age, 0.4, rng_np)), 0)
        row["total_cholesterol"] = round(max(100, _age_adjusted(185, 30, age, 0.6, rng_np)), 0)
        row["hdl"] = round(max(20, _age_adjusted(50 if is_male else 58, 10, age, -0.1, rng_np)), 0)
        row["ldl"] = round(max(30, _age_adjusted(110, 25, age, 0.5, rng_np)), 0)
        row["triglycerides"] = round(max(30, _age_adjusted(130, 40, age, 0.8, rng_np)), 0)
        row["creatinine"] = round(max(0.3, _age_adjusted(0.9 if is_male else 0.75, 0.15, age, 0.004, rng_np)), 2)
        row["egfr"] = round(max(15, _age_adjusted(105, 12, age, -0.8, rng_np)), 0)
        row["albumin"] = round(max(2.0, _age_adjusted(4.3, 0.3, age, -0.01, rng_np)), 1)
        row["crp"] = round(max(0.0, _age_adjusted(1.5, 1.5, age, 0.04, rng_np)), 1)
        row["hemoglobin"] = round(max(6, _age_adjusted(14.5 if is_male else 12.5, 1.2, age, -0.02, rng_np)), 1)
        row["wbc_count"] = round(max(2, _age_adjusted(7.0, 1.5, age, 0.01, rng_np)), 1)
        row["platelet_count"] = round(max(50, _age_adjusted(250, 50, age, -0.5, rng_np)), 0)
        row["systolic_bp"] = round(max(80, _age_adjusted(118, 12, age, 0.5, rng_np)), 0)
        row["diastolic_bp"] = round(max(50, _age_adjusted(75, 8, age, 0.2, rng_np)), 0)
        row["bmi"] = round(max(14, _age_adjusted(24.0, 3.5, age, 0.05, rng_np)), 1)
        row["waist_circumference"] = round(max(50, _age_adjusted(82 if is_male else 76, 8, age, 0.15, rng_np)), 0)
        row["vitamin_d"] = round(max(5, _age_adjusted(28, 12, age, -0.1, rng_np)), 0)
        row["tsh"] = round(max(0.1, _age_adjusted(2.5, 1.2, age, 0.02, rng_np)), 1)
        row["uric_acid"] = round(max(1.5, _age_adjusted(5.5 if is_male else 4.2, 1.0, age, 0.02, rng_np)), 1)
        row["ast"] = round(max(5, _age_adjusted(24, 8, age, 0.15, rng_np)), 0)
        row["alt"] = round(max(5, _age_adjusted(26 if is_male else 20, 10, age, 0.1, rng_np)), 0)

        # Randomly null out a few fields to simulate incomplete data
        if rng_py.random() < 0.1:
            drop_keys = rng_py.sample(
                [k for k in row if k not in ("subject_id", "age", "sex", "state")],
                k=rng_py.randint(1, 4),
            )
            for k in drop_keys:
                row[k] = None

        rows.append(row)

    return pd.DataFrame(rows)


def generate_demo_csv() -> str:
    """Return demo cohort as a CSV string."""
    df = generate_demo_cohort()
    return df.to_csv(index=False)
