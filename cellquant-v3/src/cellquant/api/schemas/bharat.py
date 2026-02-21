"""Pydantic schemas for BHARAT Cohort Analytics & AgingClock India."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────

class Sex(str, Enum):
    male = "male"
    female = "female"


class AgeGroup(str, Enum):
    young = "18-30"
    middle = "31-50"
    senior = "51-65"
    elderly = "65+"


class IndianState(str, Enum):
    andhra_pradesh = "Andhra Pradesh"
    arunachal_pradesh = "Arunachal Pradesh"
    assam = "Assam"
    bihar = "Bihar"
    chhattisgarh = "Chhattisgarh"
    goa = "Goa"
    gujarat = "Gujarat"
    haryana = "Haryana"
    himachal_pradesh = "Himachal Pradesh"
    jharkhand = "Jharkhand"
    karnataka = "Karnataka"
    kerala = "Kerala"
    madhya_pradesh = "Madhya Pradesh"
    maharashtra = "Maharashtra"
    manipur = "Manipur"
    meghalaya = "Meghalaya"
    mizoram = "Mizoram"
    nagaland = "Nagaland"
    odisha = "Odisha"
    punjab = "Punjab"
    rajasthan = "Rajasthan"
    sikkim = "Sikkim"
    tamil_nadu = "Tamil Nadu"
    telangana = "Telangana"
    tripura = "Tripura"
    uttar_pradesh = "Uttar Pradesh"
    uttarakhand = "Uttarakhand"
    west_bengal = "West Bengal"
    delhi = "Delhi"
    jammu_kashmir = "Jammu & Kashmir"
    ladakh = "Ladakh"
    chandigarh = "Chandigarh"
    puducherry = "Puducherry"


# ── Subject / Participant ─────────────────────────────────

class SubjectBiomarkers(BaseModel):
    """Core blood / clinical biomarkers used by AgingClock India."""

    hba1c: Optional[float] = Field(None, description="HbA1c (%)")
    fasting_glucose: Optional[float] = Field(None, description="Fasting glucose (mg/dL)")
    total_cholesterol: Optional[float] = Field(None, description="Total cholesterol (mg/dL)")
    hdl: Optional[float] = Field(None, description="HDL cholesterol (mg/dL)")
    ldl: Optional[float] = Field(None, description="LDL cholesterol (mg/dL)")
    triglycerides: Optional[float] = Field(None, description="Triglycerides (mg/dL)")
    creatinine: Optional[float] = Field(None, description="Serum creatinine (mg/dL)")
    egfr: Optional[float] = Field(None, description="eGFR (mL/min/1.73m²)")
    albumin: Optional[float] = Field(None, description="Serum albumin (g/dL)")
    crp: Optional[float] = Field(None, description="C-reactive protein (mg/L)")
    hemoglobin: Optional[float] = Field(None, description="Hemoglobin (g/dL)")
    wbc_count: Optional[float] = Field(None, description="WBC count (×10³/µL)")
    platelet_count: Optional[float] = Field(None, description="Platelet count (×10³/µL)")
    systolic_bp: Optional[float] = Field(None, description="Systolic blood pressure (mmHg)")
    diastolic_bp: Optional[float] = Field(None, description="Diastolic blood pressure (mmHg)")
    bmi: Optional[float] = Field(None, description="Body mass index (kg/m²)")
    waist_circumference: Optional[float] = Field(None, description="Waist circumference (cm)")
    vitamin_d: Optional[float] = Field(None, description="25-OH Vitamin D (ng/mL)")
    tsh: Optional[float] = Field(None, description="Thyroid-stimulating hormone (mIU/L)")
    uric_acid: Optional[float] = Field(None, description="Uric acid (mg/dL)")
    ast: Optional[float] = Field(None, description="AST (U/L)")
    alt: Optional[float] = Field(None, description="ALT (U/L)")


class SubjectRecord(BaseModel):
    """A single participant in the BHARAT cohort."""

    subject_id: str = Field(..., description="Unique participant ID")
    age: int = Field(..., ge=0, le=120, description="Chronological age (years)")
    sex: Sex
    state: Optional[str] = Field(None, description="Indian state of residence")
    district: Optional[str] = None
    collection_date: Optional[date] = None
    biomarkers: SubjectBiomarkers = Field(default_factory=SubjectBiomarkers)


# ── Cohort Upload / Import ────────────────────────────────

class CohortUploadRequest(BaseModel):
    """Upload cohort CSV/Excel data."""

    file_path: str = Field(..., description="Path to cohort data file (CSV or XLSX)")


class CohortInfo(BaseModel):
    """Summary of an imported cohort."""

    cohort_id: str
    name: str
    n_subjects: int
    n_male: int
    n_female: int
    age_range: tuple[int, int]
    mean_age: float
    states_represented: int
    biomarker_completeness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of biomarker fields filled"
    )


# ── AgingClock Results ────────────────────────────────────

class AgingClockResult(BaseModel):
    """Biological age prediction for a single subject."""

    subject_id: str
    chronological_age: int
    biological_age: float = Field(..., description="Predicted biological age")
    age_gap: float = Field(..., description="biological_age - chronological_age")
    percentile: float = Field(
        ..., ge=0.0, le=100.0,
        description="Percentile within same age-sex group (lower = younger)"
    )
    top_accelerators: list[BiomarkerContribution] = Field(
        default_factory=list,
        description="Top biomarkers accelerating aging"
    )
    top_decelerators: list[BiomarkerContribution] = Field(
        default_factory=list,
        description="Top biomarkers decelerating aging"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class BiomarkerContribution(BaseModel):
    """How much a single biomarker contributes to age gap."""

    name: str
    value: float
    contribution: float = Field(..., description="Signed contribution in years")
    reference_range: str = Field(..., description="Population reference range")
    status: str = Field(..., description="normal | elevated | low")


class AgingClockRunRequest(BaseModel):
    """Request to run AgingClock analysis on a cohort."""

    cohort_id: str


class AgingClockSummary(BaseModel):
    """Population-level AgingClock summary for a cohort."""

    cohort_id: str
    n_analyzed: int
    n_skipped: int
    mean_age_gap: float
    median_age_gap: float
    std_age_gap: float
    pct_accelerated: float = Field(
        ..., description="Percentage with positive age gap"
    )
    pct_decelerated: float = Field(
        ..., description="Percentage with negative age gap"
    )
    by_sex: dict[str, SexAgingSummary]
    by_age_group: dict[str, AgeGroupSummary]
    by_state: list[StateAgingSummary]


class SexAgingSummary(BaseModel):
    n: int
    mean_age_gap: float
    median_age_gap: float


class AgeGroupSummary(BaseModel):
    age_group: str
    n: int
    mean_age_gap: float
    mean_bio_age: float


class StateAgingSummary(BaseModel):
    state: str
    n: int
    mean_age_gap: float
    mean_bio_age: float
    mean_chrono_age: float


# ── Demographics ──────────────────────────────────────────

class DemographicsSummary(BaseModel):
    """Demographics breakdown for the cohort."""

    total: int
    by_sex: dict[str, int]
    by_age_group: dict[str, int]
    by_state: list[StateDemographics]
    age_distribution: list[AgeDistBin]
    bmi_distribution: list[BmiBin]


class StateDemographics(BaseModel):
    state: str
    count: int
    pct: float


class AgeDistBin(BaseModel):
    bin_start: int
    bin_end: int
    male: int
    female: int


class BmiBin(BaseModel):
    category: str
    count: int
    pct: float


# ── Biomarker Population Stats ────────────────────────────

class BiomarkerPopulationStats(BaseModel):
    """Population-level statistics for a single biomarker."""

    name: str
    display_name: str
    unit: str
    n_available: int
    mean: float
    median: float
    std: float
    p5: float
    p25: float
    p75: float
    p95: float
    reference_low: float
    reference_high: float
    pct_abnormal: float


class CohortBiomarkerProfile(BaseModel):
    """Full biomarker profile for a cohort."""

    cohort_id: str
    biomarkers: list[BiomarkerPopulationStats]
