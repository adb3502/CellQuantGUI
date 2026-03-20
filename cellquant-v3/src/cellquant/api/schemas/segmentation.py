"""Pydantic schemas for segmentation endpoints."""

from pydantic import BaseModel
from typing import Dict, List, Optional


class ConditionOverride(BaseModel):
    """Per-condition parameter overrides."""
    diameter: Optional[float] = None
    flow_threshold: Optional[float] = None
    cellprob_threshold: Optional[float] = None
    min_size: Optional[int] = None
    segmentation_suffixes: Optional[List[str]] = None
    model_type: Optional[str] = None
    pre_smooth_sigma: Optional[float] = None


class SegmentationRequest(BaseModel):
    session_id: str
    model_type: str = "cpsam"
    diameter: Optional[float] = 30.0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    channels: List[int] = [0, 0]
    use_gpu: bool = True
    batch_size: int = 4
    skip_existing: bool = False
    custom_model_path: Optional[str] = None
    segmentation_suffixes: Optional[List[str]] = None
    condition_overrides: Optional[Dict[str, ConditionOverride]] = None


class SegmentationStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    elapsed_seconds: float = 0.0
    result: Optional[dict] = None


class TaskResponse(BaseModel):
    task_id: str
    status: str = "submitted"


class ConditionMaskStatus(BaseModel):
    name: str
    mask_count: int
    base_names: List[str]


class MaskStatusResponse(BaseModel):
    conditions: List[ConditionMaskStatus]
    total_masks: int
    expected_total: int
    is_complete: bool
    has_results: bool = False
    results_n_cells: int = 0
    has_nuclear: bool = False
