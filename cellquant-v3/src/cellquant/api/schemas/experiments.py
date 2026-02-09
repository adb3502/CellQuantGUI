"""Pydantic schemas for experiment endpoints."""

from pydantic import BaseModel
from typing import Dict, List, Optional


class ScanRequest(BaseModel):
    path: str


class ChannelConfigSchema(BaseModel):
    nuclear_suffix: Optional[str] = None
    cyto_suffix: Optional[str] = None
    marker_suffixes: List[str] = []
    marker_names: List[str] = []
    mitochondrial_markers: List[str] = []


class ImageSetInfo(BaseModel):
    base_name: str
    channels: Dict[str, str]  # suffix -> filepath


class ConditionInfo(BaseModel):
    name: str
    path: str
    n_image_sets: int
    image_sets: List[ImageSetInfo] = []


class DetectionResult(BaseModel):
    channel_suffixes: List[str]
    n_channels: int
    n_image_sets: int
    n_complete: int
    n_incomplete: int
    confidence: float
    suggested_nuclear: Optional[str] = None
    suggested_cyto: Optional[str] = None
    suggested_markers: List[str] = []


class ScanResponse(BaseModel):
    session_id: str
    conditions: List[ConditionInfo]
    detection: Optional[DetectionResult] = None
