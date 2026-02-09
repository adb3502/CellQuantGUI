"""Pydantic schemas for segmentation endpoints."""

from pydantic import BaseModel
from typing import List, Optional


class SegmentationRequest(BaseModel):
    session_id: str
    model_type: str = "cyto3"
    diameter: float = 30.0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    channels: List[int] = [0, 0]
    use_gpu: bool = True
    batch_size: int = 4


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
