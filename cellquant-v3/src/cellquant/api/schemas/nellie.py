"""Pydantic schemas for nellie organelle analysis endpoints."""

from typing import List, Optional
from pydantic import BaseModel


class NellieParams(BaseModel):
    """Settings for the nellie pipeline."""

    channel_suffix: str = ""                    # channel suffix to analyse (e.g. "C1")
    channel_index: int = 0                      # index within multichannel file (fallback)
    remove_edges: bool = False
    otsu_thresh_intensity: bool = False
    threshold: Optional[float] = None           # manual threshold; None = auto
    device: str = "auto"                        # "auto" | "cpu" | "gpu"
    low_memory: bool = False

    pixel_size_xy: Optional[float] = None       # µm / pixel; None = read from metadata
    pixel_size_z: Optional[float] = None        # µm / Z-slice; None = 2-D or from metadata
    time_interval: Optional[float] = None       # seconds / frame; None = static

    include_levels: List[str] = ["organelle", "branch"]


class NellieRequest(BaseModel):
    session_id: str
    nellie_params: NellieParams = NellieParams()


class NellieResultsResponse(BaseModel):
    page: int
    per_page: int
    total_rows: int
    total_pages: int
    columns: List[str]
    data: List[dict]


class NellieSummaryResponse(BaseModel):
    total_cells: int
    n_conditions: int
    n_image_sets: int
    per_condition: List[dict]
