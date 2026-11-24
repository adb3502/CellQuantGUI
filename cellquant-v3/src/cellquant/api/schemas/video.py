"""Request/response models for video analytics."""
from typing import List, Literal, Optional

from pydantic import BaseModel


class VideoRunRequest(BaseModel):
    session_id: str
    # "tracked" is deliberately absent: the Trackastra wiring (spec 4.3,
    # plan step 6) is not implemented, and silently running static-ROI
    # instead would give migrating cells wrong traces with no warning.
    mode: Literal["static_roi", "whole_field"] = "static_roi"
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
    # How the (T,Y,X) recording is collapsed before segmentation. "mean" was
    # the original default and dilutes exactly the transient cells the
    # analysis is looking for; measured on real dose-series data, max gives
    # better foreground contrast and never lost a cell mean found.
    projection: Literal["max", "percentile", "std", "mean"] = "max"
    projection_percentile: float = 99.0


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
    group_key: str = ""


class DoseTableResponse(BaseModel):
    doses: List[ParsedDoseOut]
    needs_confirmation: bool
