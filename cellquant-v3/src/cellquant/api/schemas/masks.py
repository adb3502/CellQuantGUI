"""Pydantic schemas for mask editing endpoints."""

from pydantic import BaseModel
from typing import List, Optional


class DeleteCellRequest(BaseModel):
    cell_id: int


class MergeCellsRequest(BaseModel):
    cell_ids: List[int]


class AddCellPolygonRequest(BaseModel):
    polygon_coords: List[List[float]]  # [[row, col], ...]
    overwrite: bool = False


class AddCellFloodRequest(BaseModel):
    row: int
    col: int
    threshold: float = 0.3


class DilateCellsRequest(BaseModel):
    cell_ids: List[int]
    iterations: int = 1


class ErodeCellsRequest(BaseModel):
    cell_ids: List[int]
    iterations: int = 1


class SmoothRequest(BaseModel):
    cell_ids: Optional[List[int]] = None
    sigma: float = 1.0


class FillHolesRequest(BaseModel):
    max_hole_size: Optional[int] = None


class CleanSmallRequest(BaseModel):
    min_size: int = 50


class MaskStatsResponse(BaseModel):
    n_cells: int
    min_area: float
    max_area: float
    mean_area: float


class MaskEditResponse(BaseModel):
    success: bool
    n_cells: int
