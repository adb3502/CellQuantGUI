"""Pydantic schemas for training endpoints."""

from pydantic import BaseModel
from typing import List, Optional


class TrainingPairInfo(BaseModel):
    key: str
    condition: str
    base_name: str
    image_path: str
    mask_path: str
    collected_at: float


class TrainingDataResponse(BaseModel):
    pair_count: int
    pairs: List[TrainingPairInfo]


class CollectPairRequest(BaseModel):
    condition: str
    base_name: str


class FinetuneRequest(BaseModel):
    session_id: str
    base_model: str = "cpsam"
    model_name: str = "custom_senescent"
    n_epochs: int = 100
    learning_rate: float = 1e-5
    batch_size: int = 2


class FinetuneStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str


class ModelInfo(BaseModel):
    name: str
    path: str
    base_model: str
    n_training_images: int
    created_at: float
    final_loss: Optional[float] = None


class ModelListResponse(BaseModel):
    models: List[ModelInfo]
