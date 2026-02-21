"""OxyTrack domain models – experiments, treatments, and observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class SenescenceMarker(str, Enum):
    """Common senescence markers quantified in oxysterol experiments."""

    SA_BETA_GAL = "SA-β-Gal"
    P21 = "p21"
    P16 = "p16"
    GAMMA_H2AX = "γH2AX"
    IL6 = "IL-6"
    IL8 = "IL-8"
    LAMIN_B1 = "Lamin B1"
    HMGB1 = "HMGB1"
    KI67 = "Ki67"
    OTHER = "other"


class OxysterolType(str, Enum):
    """Oxysterol species commonly used in senescence research."""

    SEVEN_KC = "7-ketocholesterol"
    SEVEN_BETA_HC = "7β-hydroxycholesterol"
    TWENTY_FIVE_HC = "25-hydroxycholesterol"
    TWENTY_SEVEN_HC = "27-hydroxycholesterol"
    CHOLESTEROL_EPOXIDE = "cholesterol-5,6-epoxide"
    SEVEN_ALPHA_HC = "7α-hydroxycholesterol"
    CUSTOM = "custom"


@dataclass
class TreatmentArm:
    """A single treatment condition within an experiment."""

    id: str
    name: str
    oxysterol: str  # OxysterolType value or custom name
    concentration_uM: float
    vehicle: str = "ethanol"
    vehicle_pct: float = 0.1
    notes: str = ""


@dataclass
class TimePoint:
    """A sampled time-point in a time-course experiment."""

    hours: float
    label: str = ""  # e.g. "24h", "Day 3"
    cellquant_session_id: Optional[str] = None  # link to CellQuant session

    def __post_init__(self):
        if not self.label:
            if self.hours < 24:
                self.label = f"{self.hours:.0f}h"
            else:
                self.label = f"Day {self.hours / 24:.0f}"


@dataclass
class MarkerPanel:
    """Set of senescence markers being measured in an experiment."""

    markers: List[str] = field(default_factory=list)
    channel_mapping: Dict[str, str] = field(default_factory=dict)
    # channel_mapping: marker_name -> CellQuant channel suffix (e.g. "C2")


@dataclass
class Observation:
    """A quantitative observation linking a time-point to results."""

    id: str
    experiment_id: str
    treatment_id: str
    timepoint_hours: float
    marker: str
    value: float
    unit: str = "CTCF"
    n_cells: int = 0
    std_dev: float = 0.0
    cellquant_session_id: Optional[str] = None
    notes: str = ""
    recorded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Experiment:
    """Top-level oxysterol-senescence experiment."""

    id: str
    name: str
    cell_line: str
    passage: int = 0
    description: str = ""
    treatments: List[TreatmentArm] = field(default_factory=list)
    timepoints: List[TimePoint] = field(default_factory=list)
    marker_panel: MarkerPanel = field(default_factory=MarkerPanel)
    observations: List[Observation] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: List[str] = field(default_factory=list)
