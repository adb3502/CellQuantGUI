"""Request/response models for video analytics."""
from typing import List, Literal, Optional

from pydantic import BaseModel


class VideoRunRequest(BaseModel):
    session_id: str
    mode: Literal["static_roi", "tracked", "whole_field"] = "static_roi"
    condition: Optional[str] = None
    baseline_mode: Literal["fraction", "stim_frame", "auto"] = "fraction"
    baseline_fraction: float = 0.10
    stim_frame: Optional[int] = None
    interval_s: Optional[float] = None
    detrend: bool = False
    event_threshold_sigma: float = 3.0
    event_refractory_frames: int = 5
    diameter: float = 30.0
    model_type: str = "cpsam"


class VideoRunResponse(BaseModel):
    task_id: str


class ParsedDoseOut(BaseModel):
    source: str
    parsed: bool
    molar: Optional[float] = None
    unit: Optional[str] = None
    volume_ul: Optional[float] = None
    ambiguous: bool = False
    note: str = ""


class DoseTableResponse(BaseModel):
    doses: List[ParsedDoseOut]
    needs_confirmation: bool
