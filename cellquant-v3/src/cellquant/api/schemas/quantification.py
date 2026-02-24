"""Pydantic schemas for quantification endpoints."""

from pydantic import BaseModel
from typing import Dict, List, Optional


class QCFilterParams(BaseModel):
    """Post-segmentation quality-control filter settings."""

    enabled: bool = True
    remove_border_objects: bool = True
    min_area: Optional[int] = None
    max_area: Optional[int] = None
    area_iqr_factor: float = 1.5
    min_solidity: Optional[float] = None
    max_eccentricity: Optional[float] = None
    min_circularity: Optional[float] = None
    max_aspect_ratio: Optional[float] = None


class QuantificationRequest(BaseModel):
    session_id: str
    background_method: str = "auto"
    marker_suffixes: List[str] = []
    marker_names: List[str] = []
    mitochondrial_markers: List[str] = []
    qc_filters: QCFilterParams = QCFilterParams()
    negative_control_path: Optional[str] = None
    manual_background_value: Optional[float] = None
    outlier_threshold: float = 3.5


class QuantificationResultRow(BaseModel):
    condition: str
    image_set: str
    cell_id: int
    area: float
    markers: Dict[str, float]


class ResultsPageResponse(BaseModel):
    page: int
    per_page: int
    total_rows: int
    total_pages: int
    columns: List[str]
    data: List[dict]


class SummaryStatsResponse(BaseModel):
    total_cells: int
    n_conditions: int
    n_image_sets: int
    per_condition: List[dict]


class QCSummaryResponse(BaseModel):
    """Hierarchical summary (cells → FOVs → conditions)."""

    summary: List[dict]
    fov_data: List[dict]
