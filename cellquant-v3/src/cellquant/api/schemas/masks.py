"""Pydantic schemas for mask editing endpoints."""

from pydantic import BaseModel
from typing import List, Optional


class DeleteCellRequest(BaseModel):
    cell_id: int


class MergeCellsRequest(BaseModel):
    cell_ids: List[int]


class MaskStatsResponse(BaseModel):
    n_cells: int
    min_area: float
    max_area: float
    mean_area: float
