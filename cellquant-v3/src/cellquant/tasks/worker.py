"""Task workers that wrap core pipeline operations."""

from pathlib import Path
from typing import Optional

from cellquant.tasks.queue import TaskInfo, TaskQueue


def run_segmentation_task(
    task: TaskInfo,
    queue: TaskQueue,
    session,
    seg_params: dict,
):
    """Run segmentation in background thread."""
    from cellquant.core.segmentation.cellpose_engine import (
        CellposeEngine,
        SegmentationParams,
    )
    from cellquant.core.io.image_loader import load_image, normalize_image
    from cellquant.core.io.mask_io import save_mask
    import numpy as np

    params = SegmentationParams(**seg_params)
    engine = CellposeEngine(
        model_type=params.model_type,
        use_gpu=params.use_gpu,
        batch_size=params.batch_size,
    )

    total_images = sum(
        len(cond.get("image_sets", {})) for cond in session.conditions.values()
    )
    task.total = total_images
    processed = 0
    total_cells = 0

    for cond_name, cond_data in session.conditions.items():
        if task.status == "cancelled":
            break

        image_sets = cond_data.get("image_sets", {})
        nuclear_suffix = cond_data.get("nuclear_suffix", "C0")
        cyto_suffix = cond_data.get("cyto_suffix", "C1")

        for base_name, channels in image_sets.items():
            if task.status == "cancelled":
                break

            queue.update_progress(
                task,
                current=processed,
                stage="segmenting",
                condition=cond_name,
                image_set=base_name,
                message=f"Segmenting {cond_name}/{base_name}",
            )

            nuc_path = channels.get(nuclear_suffix) or channels.get(nuclear_suffix.upper())
            if not nuc_path:
                processed += 1
                continue

            nuc_img = load_image(nuc_path)
            nuc_norm = normalize_image(nuc_img)

            cyto_path = channels.get(cyto_suffix) or channels.get(cyto_suffix.upper())
            if cyto_path:
                cyto_img = load_image(cyto_path)
                cyto_norm = normalize_image(cyto_img)
                seg_input = np.stack([nuc_norm, cyto_norm], axis=0)
            else:
                seg_input = nuc_norm

            result = engine.segment_single(
                seg_input,
                diameter=params.diameter,
                flow_threshold=params.flow_threshold,
                min_size=params.min_size,
                channels=params.channels,
            )

            masks = result.masks
            n_cells = result.n_cells
            total_cells += n_cells

            # Store in session
            if cond_name not in session.masks:
                session.masks[cond_name] = {}
            session.masks[cond_name][base_name] = masks

            # Save to disk
            mask_path = session.get_mask_path(cond_name, base_name)
            np.save(mask_path, masks)

            processed += 1

    engine.clear_gpu_memory()
    return {"total_cells": total_cells, "images_processed": processed}


def run_quantification_task(
    task: TaskInfo,
    queue: TaskQueue,
    session,
    quant_params: dict,
):
    """Run quantification in background thread."""
    from cellquant.core.io.image_loader import load_image
    from cellquant.core.quantification.ctcf import (
        calculate_ctcf_vectorized,
        quantify_multiple_markers,
        results_to_dataframe,
    )
    from cellquant.core.quantification.background import estimate_background
    import pandas as pd
    import numpy as np

    bg_method = quant_params.get("background_method", "median")
    marker_suffixes = quant_params.get("marker_suffixes", [])
    marker_names = quant_params.get("marker_names", [])
    mitochondrial_markers = quant_params.get("mitochondrial_markers", [])

    total = sum(len(masks) for masks in session.masks.values())
    task.total = total
    processed = 0
    all_dfs = []

    for cond_name, cond_masks in session.masks.items():
        if task.status == "cancelled":
            break

        cond_data = session.conditions.get(cond_name, {})
        image_sets = cond_data.get("image_sets", {})

        for base_name, masks in cond_masks.items():
            if task.status == "cancelled":
                break

            queue.update_progress(
                task,
                current=processed,
                stage="quantifying",
                condition=cond_name,
                image_set=base_name,
                message=f"Quantifying {cond_name}/{base_name}",
            )

            channels = image_sets.get(base_name, {})
            marker_images = {}
            for suffix, name in zip(marker_suffixes, marker_names):
                path = channels.get(suffix) or channels.get(suffix.upper())
                if path:
                    marker_images[name] = load_image(path)

            if not marker_images:
                processed += 1
                continue

            backgrounds = {
                name: estimate_background(img, masks, method=bg_method)
                for name, img in marker_images.items()
            }

            results = quantify_multiple_markers(
                marker_images=marker_images,
                masks=masks,
                backgrounds=backgrounds,
                mitochondrial_markers=mitochondrial_markers,
            )

            df = results_to_dataframe(
                results=results,
                condition=cond_name,
                image_set=base_name,
            )
            if len(df) > 0:
                all_dfs.append(df)

            processed += 1

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        session.results_df = combined
        session.save_results()
        return {"total_cells": len(combined), "conditions": len(session.masks)}

    return {"total_cells": 0, "conditions": 0}


def run_tracking_task(
    task: TaskInfo,
    queue: TaskQueue,
    session,
    tracking_params: dict,
):
    """Run Trackastra cell tracking in background thread."""
    try:
        from trackastra.model import Trackastra
        from trackastra.tracking import graph_to_ctc
    except ImportError:
        raise ImportError("trackastra required. Install with: pip install trackastra")

    import numpy as np
    from cellquant.core.io.image_loader import load_image

    model_name = tracking_params.get("model", "general_2d")
    mode = tracking_params.get("mode", "greedy")
    condition = tracking_params.get("condition")
    device = tracking_params.get("device", "automatic")

    queue.update_progress(task, stage="loading_model", message="Loading Trackastra model...")

    model = Trackastra.from_pretrained(model_name, device=device)

    # Get timelapse images and masks for this condition
    cond_data = session.conditions.get(condition, {})
    cond_masks = session.masks.get(condition, {})

    # Sort by base_name to get temporal order
    sorted_names = sorted(cond_masks.keys())
    task.total = len(sorted_names)

    queue.update_progress(
        task,
        stage="preparing",
        message=f"Preparing {len(sorted_names)} frames for tracking",
    )

    # Stack masks and images into timelapse arrays
    masks_list = [cond_masks[name] for name in sorted_names]
    masks_stack = np.stack(masks_list, axis=0)

    image_sets = cond_data.get("image_sets", {})
    nuclear_suffix = cond_data.get("nuclear_suffix", "C0")
    imgs_list = []
    for name in sorted_names:
        channels = image_sets.get(name, {})
        path = channels.get(nuclear_suffix) or channels.get(nuclear_suffix.upper())
        if path:
            imgs_list.append(load_image(path))
        else:
            imgs_list.append(np.zeros_like(masks_list[0], dtype=np.uint16))
    imgs_stack = np.stack(imgs_list, axis=0)

    queue.update_progress(task, stage="tracking", message="Running Trackastra tracking...")

    track_graph, masks_tracked = model.track(imgs_stack, masks_stack, mode=mode)

    # Store tracked results
    if condition not in session.tracked_masks:
        session.tracked_masks[condition] = {}
    for i, name in enumerate(sorted_names):
        session.tracked_masks[condition][name] = masks_tracked[i]

    # Export CTC format
    export_dir = session.get_export_dir() / f"tracks_{condition}"
    export_dir.mkdir(parents=True, exist_ok=True)
    ctc_tracks, ctc_masks = graph_to_ctc(
        track_graph, masks_tracked, outdir=str(export_dir)
    )

    n_tracks = len(set(masks_tracked[masks_tracked > 0].flatten()))
    return {"n_tracks": n_tracks, "n_frames": len(sorted_names)}
