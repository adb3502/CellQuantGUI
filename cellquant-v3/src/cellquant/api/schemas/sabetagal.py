"""Request/response models for SA-beta-gal analytics."""
from typing import List, Optional

from pydantic import BaseModel


class SABetaGalRunRequest(BaseModel):
    session_id: str
    condition: Optional[str] = None
    white_percentile: float = 95.0
    chromogen_index: int = 0
    threshold: Optional[float] = None  # None -> Otsu
    stain_vectors: Optional[List[List[float]]] = None
    min_area: int = 10
    diameter: float = 30.0
    model_type: str = "cyto3"


class SABetaGalRunResponse(BaseModel):
    task_id: str
