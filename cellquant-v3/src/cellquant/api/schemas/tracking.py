"""Pydantic schemas for tracking endpoints."""

from pydantic import BaseModel
from typing import Optional


class TrackingRequest(BaseModel):
    session_id: str
    condition: str
    model: str = "general_2d"
    mode: str = "greedy"  # greedy, greedy_nodiv, ilp
    device: str = "automatic"


class TrackingResponse(BaseModel):
    task_id: str
    status: str = "submitted"


class TrackResult(BaseModel):
    n_tracks: int
    n_frames: int
    condition: str
