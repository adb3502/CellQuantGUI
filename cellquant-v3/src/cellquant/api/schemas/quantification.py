"""Pydantic schemas for quantification endpoints."""

from pydantic import BaseModel
from typing import Dict, List, Optional


class QuantificationRequest(BaseModel):
    session_id: str
    background_method: str = "median"
    marker_suffixes: List[str] = []
    marker_names: List[str] = []
    mitochondrial_markers: List[str] = []


class QuantificationResultRow(BaseModel):
    condition: str
    image_set: str
    cell_id: int
    area: float
    markers: Dict[str, float]  # marker_name -> CTCF value


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
