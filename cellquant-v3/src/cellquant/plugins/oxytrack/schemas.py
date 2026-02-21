"""Pydantic schemas for OxyTrack API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# -- Treatment ---------------------------------------------------------------

class TreatmentArmSchema(BaseModel):
    name: str
    oxysterol: str
    concentration_uM: float
    vehicle: str = "ethanol"
    vehicle_pct: float = 0.1
    notes: str = ""


# -- Time-point --------------------------------------------------------------

class TimePointSchema(BaseModel):
    hours: float
    label: str = ""
    cellquant_session_id: Optional[str] = None


# -- Marker panel ------------------------------------------------------------

class MarkerPanelSchema(BaseModel):
    markers: List[str] = []
    channel_mapping: Dict[str, str] = {}


# -- Observation -------------------------------------------------------------

class ObservationCreate(BaseModel):
    treatment_id: str
    timepoint_hours: float
    marker: str
    value: float
    unit: str = "CTCF"
    n_cells: int = 0
    std_dev: float = 0.0
    cellquant_session_id: Optional[str] = None
    notes: str = ""


class ObservationResponse(ObservationCreate):
    id: str
    experiment_id: str
    recorded_at: str


# -- Experiment CRUD ---------------------------------------------------------

class ExperimentCreate(BaseModel):
    name: str
    cell_line: str
    passage: int = 0
    description: str = ""
    treatments: List[TreatmentArmSchema] = []
    timepoints: List[TimePointSchema] = []
    marker_panel: MarkerPanelSchema = MarkerPanelSchema()
    tags: List[str] = []


class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    cell_line: Optional[str] = None
    passage: Optional[int] = None
    description: Optional[str] = None
    treatments: Optional[List[TreatmentArmSchema]] = None
    timepoints: Optional[List[TimePointSchema]] = None
    marker_panel: Optional[MarkerPanelSchema] = None
    tags: Optional[List[str]] = None


class ExperimentSummary(BaseModel):
    id: str
    name: str
    cell_line: str
    n_treatments: int
    n_timepoints: int
    n_observations: int
    created_at: str
    tags: List[str] = []


class ExperimentDetail(BaseModel):
    id: str
    name: str
    cell_line: str
    passage: int
    description: str
    treatments: List[TreatmentArmSchema]
    timepoints: List[TimePointSchema]
    marker_panel: MarkerPanelSchema
    observations: List[ObservationResponse]
    created_at: str
    updated_at: str
    tags: List[str] = []


# -- Import from CellQuant --------------------------------------------------

class ImportFromSessionRequest(BaseModel):
    """Pull quantification results from a CellQuant session into OxyTrack."""
    cellquant_session_id: str
    treatment_id: str
    timepoint_hours: float


# -- Analysis responses ------------------------------------------------------

class TimeCourseSeries(BaseModel):
    treatment: str
    marker: str
    hours: List[float]
    means: List[float]
    std_devs: List[float]


class TimeCourseResponse(BaseModel):
    experiment_id: str
    series: List[TimeCourseSeries]


class DoseResponsePoint(BaseModel):
    concentration_uM: float
    mean: float
    std_dev: float
    n_cells: int


class DoseResponseSeries(BaseModel):
    oxysterol: str
    marker: str
    points: List[DoseResponsePoint]


class DoseResponseResponse(BaseModel):
    experiment_id: str
    series: List[DoseResponseSeries]
