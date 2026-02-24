"""Training data collection and Cellpose fine-tuning endpoints."""

from pathlib import Path
from fastapi import APIRouter, HTTPException

from cellquant.api.dependencies import get_session, get_task_queue
from cellquant.api.schemas.training import (
    TrainingDataResponse,
    TrainingPairInfo,
    CollectPairRequest,
    FinetuneRequest,
    FinetuneStatusResponse,
    ModelInfo,
    ModelListResponse,
)

router = APIRouter(prefix="/training", tags=["training"])


# ── Training data collection ─────────────────────────────


@router.get("/data/{session_id}", response_model=TrainingDataResponse)
async def get_training_data(session_id: str):
    """Get collected training data summary."""
    session = get_session(session_id)
    collector = session.training_collector
    pairs = collector.get_pairs()
    return TrainingDataResponse(
        pair_count=len(pairs),
        pairs=[TrainingPairInfo(**p) for p in pairs],
    )


@router.post("/data/{session_id}/collect")
async def collect_training_pair(session_id: str, req: CollectPairRequest):
    """Manually collect a training pair from the current mask state."""
    session = get_session(session_id)

    # Get current mask
    if req.condition not in session.masks or req.base_name not in session.masks[req.condition]:
        raise HTTPException(404, f"No masks loaded for {req.condition}/{req.base_name}")

    mask = session.masks[req.condition][req.base_name]

    # Find intensity image path
    image_path = _get_image_path(session, req.condition, req.base_name)

    session.training_collector.collect_pair(
        req.condition, req.base_name, str(image_path), mask
    )

    return {
        "success": True,
        "pair_count": session.training_collector.pair_count(),
    }


@router.delete("/data/{session_id}/{condition}/{base_name}")
async def remove_training_pair(session_id: str, condition: str, base_name: str):
    """Remove a training pair."""
    session = get_session(session_id)
    removed = session.training_collector.remove_pair(condition, base_name)
    if not removed:
        raise HTTPException(404, "Training pair not found")
    return {"success": True, "pair_count": session.training_collector.pair_count()}


# ── Fine-tuning ──────────────────────────────────────────


@router.post("/finetune")
async def start_finetune(req: FinetuneRequest):
    """Start Cellpose fine-tuning as a background task."""
    session = get_session(req.session_id)
    collector = session.training_collector

    if collector.pair_count() < 1:
        raise HTTPException(400, "Need at least 1 training pair")

    queue = get_task_queue()

    def finetune_task(task, *args, **kwargs):
        from cellquant.core.segmentation.finetune import run_finetune

        images, masks = collector.get_training_arrays()
        if not images:
            raise ValueError("No valid training images found on disk")

        def progress_cb(message, pct):
            queue.update_progress(task, message=message, progress=pct)

        result = run_finetune(
            images=images,
            masks=masks,
            base_model=req.base_model,
            model_name=req.model_name,
            n_epochs=req.n_epochs,
            learning_rate=req.learning_rate,
            batch_size=req.batch_size,
            progress_callback=progress_cb,
        )
        return result

    task_id = queue.submit(req.session_id, "finetune", finetune_task)
    return {"task_id": task_id, "status": "submitted"}


@router.get("/status/{task_id}", response_model=FinetuneStatusResponse)
async def get_finetune_status(task_id: str):
    """Get fine-tuning task status."""
    queue = get_task_queue()
    task = queue.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return FinetuneStatusResponse(
        task_id=task.id,
        status=task.status,
        progress=task.progress,
        message=task.message,
    )


# ── Model library ────────────────────────────────────────


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all custom fine-tuned models."""
    from cellquant.core.segmentation.finetune import list_custom_models

    models = list_custom_models()
    return ModelListResponse(
        models=[ModelInfo(**m) for m in models],
    )


# ── Helpers ──────────────────────────────────────────────


def _get_image_path(session, condition: str, base_name: str) -> str:
    """Get the first available intensity image path for a base_name."""
    cond_data = session.conditions.get(condition, {})
    image_sets = cond_data.get("image_sets", {})
    channels = image_sets.get(base_name, {})
    for suffix, ch_path in channels.items():
        if ch_path and Path(ch_path).exists():
            return ch_path
    raise HTTPException(404, f"No intensity image found: {condition}/{base_name}")
