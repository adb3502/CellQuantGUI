"""Pydantic schemas for WebSocket progress messages."""

from pydantic import BaseModel
from typing import Optional


class ProgressMessage(BaseModel):
    type: str = "progress"  # progress | task_complete
    task_id: str
    task_type: str
    status: str
    progress: float = 0.0
    current: int = 0
    total: int = 0
    stage: str = ""
    condition: str = ""
    image_set: str = ""
    message: str = ""
    elapsed_seconds: float = 0.0
    data: Optional[dict] = None
    error: Optional[str] = None
