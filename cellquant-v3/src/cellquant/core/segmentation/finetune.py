"""Cellpose fine-tuning pipeline.

Uses Cellpose 4.0+ training API to fine-tune models on user-corrected masks.
"""

from pathlib import Path
from typing import List, Optional, Callable
import json
import time
import numpy as np


# Global model library location
MODEL_LIBRARY = Path.home() / ".cellquant" / "models"


def get_model_library() -> Path:
    MODEL_LIBRARY.mkdir(parents=True, exist_ok=True)
    return MODEL_LIBRARY


def list_custom_models() -> list:
    """List all custom fine-tuned models."""
    lib = get_model_library()
    models = []
    for meta_path in sorted(lib.glob("*/metadata.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        models.append(meta)
    return models


def run_finetune(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    base_model: str = "cpsam",
    model_name: str = "custom_senescent",
    n_epochs: int = 100,
    learning_rate: float = 1e-5,
    batch_size: int = 2,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """Run Cellpose fine-tuning on collected training data.

    Args:
        images: List of training images
        masks: List of corrected mask arrays
        base_model: Base Cellpose model to fine-tune from
        model_name: Name for the fine-tuned model
        n_epochs: Number of training epochs
        learning_rate: Training learning rate
        batch_size: Training batch size
        progress_callback: Optional (message, progress%) callback

    Returns:
        dict with model_path, metadata
    """
    from cellpose.models import CellposeModel
    from cellpose import train

    if len(images) < 1:
        raise ValueError("Need at least 1 training image")

    if progress_callback:
        progress_callback("Loading base model...", 5.0)

    model = CellposeModel(gpu=True, pretrained_model=base_model)

    # Train/test split (80/20, min 1 test)
    n = len(images)
    n_test = max(1, n // 5)
    n_train = n - n_test

    # Shuffle indices
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_images = [images[i] for i in train_idx]
    train_masks = [masks[i] for i in train_idx]
    test_images = [images[i] for i in test_idx]
    test_masks = [masks[i] for i in test_idx]

    # Save path
    save_dir = get_model_library() / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(f"Training on {n_train} images ({n_test} test)...", 10.0)

    # Run training
    model_path, train_losses, test_losses = train.train_seg(
        net=model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=test_images if test_images else None,
        test_labels=test_masks if test_masks else None,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        save_path=str(save_dir),
        min_train_masks=1,
    )

    if progress_callback:
        progress_callback("Saving model metadata...", 95.0)

    # Save metadata
    final_loss = float(train_losses[-1]) if train_losses else None
    metadata = {
        "name": model_name,
        "path": str(model_path),
        "base_model": base_model,
        "n_training_images": n_train,
        "n_test_images": n_test,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "final_loss": final_loss,
        "created_at": time.time(),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if progress_callback:
        progress_callback("Training complete!", 100.0)

    return metadata
