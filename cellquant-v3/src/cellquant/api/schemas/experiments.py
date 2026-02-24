"""Pydantic schemas for experiment endpoints."""

from pydantic import BaseModel
from typing import Dict, List, Optional


class ScanRequest(BaseModel):
    path: str
    output_path: Optional[str] = None


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
    suggested_nuclear: Optional[str] = None
    suggested_cyto: Optional[str] = None
    suggested_markers: List[str] = []
    channel_wavelengths: Dict[str, float] = {}   # suffix -> nm
    channel_colors: Dict[str, str] = {}           # suffix -> hex color from wavelength


class ScanResponse(BaseModel):
    session_id: str
    conditions: List[ConditionInfo]
    detection: Optional[DetectionResult] = None
    output_path: Optional[str] = None


class SetOutputRequest(BaseModel):
    output_path: str


class PreprocessingRequest(BaseModel):
    dark_frame_paths: List[str] = []
    flat_field_paths: List[str] = []
