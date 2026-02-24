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
        _create_simple_overlay,
        _create_outline_overlay,
    )
    from cellquant.core.io.image_loader import load_image, normalize_image
    from cellquant.core.io.mask_io import save_mask
    from cellquant.tiles.thumbnail import imagej_auto_contrast
    from PIL import Image
    import numpy as np

    params = SegmentationParams(**seg_params)
    print(f"[CellQuant] Initializing Cellpose engine: model={params.model_type}, gpu={params.use_gpu}, batch={params.batch_size}")
    engine = CellposeEngine(
        model_type=params.model_type,
        use_gpu=params.use_gpu,
        batch_size=params.batch_size,
    )
    print(f"[CellQuant] Engine created, loading model...")

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

            # Skip existing masks if requested
            if seg_params.get("skip_existing", False):
                mask_path = session.get_mask_path(cond_name, base_name)
                if mask_path.exists():
                    session.masks.setdefault(cond_name, {})[base_name] = np.load(mask_path)
                    n_cells = int(session.masks[cond_name][base_name].max())
                    total_cells += n_cells
                    print(f"[CellQuant] Skipping {cond_name}/{base_name} (mask exists, {n_cells} cells)")
                    processed += 1
                    queue.update_progress(
                        task,
                        current=processed,
                        stage="segmenting",
                        condition=cond_name,
                        image_set=base_name,
                        message=f"Skipped {cond_name}/{base_name} (existing mask)",
                    )
                    continue

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

            print(f"[CellQuant] Segmenting {cond_name}/{base_name} ...")
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
            print(f"[CellQuant]   -> {n_cells} cells detected")

            # Store in session
            if cond_name not in session.masks:
                session.masks[cond_name] = {}
            session.masks[cond_name][base_name] = masks

            # Save to disk
            mask_path = session.get_mask_path(cond_name, base_name)
            np.save(mask_path, masks)

            # Save outputs to per-image-set folder (matching v1 structure)
            try:
                from skimage import io as skio

                seg_type = params.model_type
                image_set_dir = session.directory / cond_name / base_name
                image_set_dir.mkdir(parents=True, exist_ok=True)

                # Mask TIFF (uint16, compatible with ImageJ/FIJI)
                skio.imsave(
                    str(image_set_dir / f"{base_name}_masks_{seg_type}.tif"),
                    masks.astype(np.uint16),
                    check_contrast=False,
                )

                # Build visual composite from all channels (matching v1)
                channel_imgs = [nuc_img.astype(np.float64)]
                for suffix, ch_path in channels.items():
                    if suffix in (nuclear_suffix, nuclear_suffix.upper()):
                        continue  # already loaded
                    if ch_path and Path(ch_path).exists():
                        channel_imgs.append(load_image(ch_path).astype(np.float64))
                if len(channel_imgs) == 1:
                    composite = channel_imgs[0]
                else:
                    composite = np.sum(np.stack(channel_imgs, axis=0), axis=0)
                composite_uint8 = imagej_auto_contrast(composite)

                # Outline overlay (red cell boundaries)
                outline = _create_outline_overlay(composite_uint8, masks)
                Image.fromarray(outline, "RGB").save(
                    image_set_dir / f"{base_name}_outline_overlay_{seg_type}.png"
                )

                # Colorful segmentation overlay
                filled = _create_simple_overlay(composite_uint8, masks, alpha=0.5)
                Image.fromarray(filled, "RGB").save(
                    image_set_dir / f"{base_name}_colorful_segmentation_{seg_type}.png"
                )

                print(f"[CellQuant]   -> Saved masks + overlays to {cond_name}/{base_name}/")
            except Exception as e:
                print(f"[CellQuant] Warning: output save failed for {base_name}: {e}")

            processed += 1

    engine.clear_gpu_memory()
    return {"total_cells": total_cells, "images_processed": processed}


def run_quantification_task(
    task: TaskInfo,
    queue: TaskQueue,
    session,
    quant_params: dict,
):
    """Run the full CTCF quantification pipeline.

    Pipeline stages:
    1. Optional preprocessing (dark subtraction + flat-field)
    2. QC filtering (border removal, morphology gates)
    3. Background estimation (7 methods + auto)
    4. Enhanced CTCF with error propagation, MCF, quality flags
    5. Outlier detection (MAD-based)
    """
    from cellquant.core.io.image_loader import load_image
    from cellquant.core.quantification.ctcf import (
        quantify_multiple_markers,
        results_to_dataframe,
    )
    from cellquant.core.quantification.background import estimate_background
    from cellquant.core.quantification.qc_filters import (
        QCFilterConfig,
        apply_qc_filters,
        compute_cell_morphology,
    )
    from cellquant.core.quantification.outliers import flag_outliers_in_dataframe
    from cellquant.core.preprocessing.correction import (
        PreprocessingConfig,
        correct_image,
    )
    import pandas as pd
    import numpy as np

    bg_method = quant_params.get("background_method", "auto")
    marker_suffixes = quant_params.get("marker_suffixes", [])
    marker_names = quant_params.get("marker_names", [])
    mitochondrial_markers = quant_params.get("mitochondrial_markers", [])
    outlier_threshold = quant_params.get("outlier_threshold", 3.5)

    # ── QC filter config ──────────────────────────────────────────
    # Morphological filters are opt-in: only applied when the frontend
    # sends an explicit non-None value that differs from the "disabled"
    # sentinel.  Border removal is the only default-on filter.
    qc_raw = quant_params.get("qc_filters", {})
    qc_enabled = qc_raw.get("enabled", True) if isinstance(qc_raw, dict) else True

    # Old frontend versions always send the old hardcoded defaults
    # (0.80, 0.90, 0.40, 3.0) even when the user hasn't opted in.
    # Treat those exact old defaults as "not set".
    _OLD_DEFAULTS = {
        "min_solidity": 0.80,
        "max_eccentricity": 0.90,
        "min_circularity": 0.40,
        "max_aspect_ratio": 3.0,
    }
    def _filter_val(key: str):
        v = qc_raw.get(key)
        if v is None:
            return None
        # If it matches the old hardcoded default, treat as disabled
        if key in _OLD_DEFAULTS and v == _OLD_DEFAULTS[key]:
            return None
        return v

    qc_config = QCFilterConfig(
        remove_border_objects=qc_raw.get("remove_border_objects", True),
        min_area=qc_raw.get("min_area"),
        max_area=qc_raw.get("max_area"),
        area_iqr_factor=qc_raw.get("area_iqr_factor", 1.5),
        min_solidity=_filter_val("min_solidity"),
        max_eccentricity=_filter_val("max_eccentricity"),
        min_circularity=_filter_val("min_circularity"),
        max_aspect_ratio=_filter_val("max_aspect_ratio"),
    ) if isinstance(qc_raw, dict) else QCFilterConfig()

    # ── Preprocessing config ──────────────────────────────────────
    pp_config = PreprocessingConfig(
        dark_master=getattr(session, "dark_master", None),
        flat_norm=getattr(session, "flat_norm", None),
    )
    has_preprocessing = (
        pp_config.dark_master is not None or pp_config.flat_norm is not None
    )

    # ── Negative control / manual BG ──────────────────────────────
    neg_control_path = quant_params.get("negative_control_path")
    manual_bg = quant_params.get("manual_background_value")
    negative_control = None
    if neg_control_path:
        try:
            negative_control = load_image(neg_control_path)
            if has_preprocessing:
                negative_control = correct_image(negative_control, pp_config)
        except Exception as e:
            print(f"[CellQuant] Warning: failed to load negative control: {e}")

    # Load masks from disk if not in memory (e.g., after server restart)
    if not session.masks or all(len(v) == 0 for v in session.masks.values()):
        loaded_count = 0
        for cond_name in session.conditions:
            mask_dir = session.directory / "masks" / cond_name
            if mask_dir.exists():
                session.masks.setdefault(cond_name, {})
                for npy_file in sorted(mask_dir.glob("*_masks.npy")):
                    base = npy_file.stem.replace("_masks", "")
                    session.masks[cond_name][base] = np.load(npy_file)
                    loaded_count += 1
        if loaded_count > 0:
            print(f"[CellQuant] Loaded {loaded_count} masks from disk")

    total = sum(len(masks) for masks in session.masks.values())
    task.total = total
    processed = 0
    all_dfs = []
    total_rejected = 0

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

            # ── Stage 1: Compute morphology + QC filter ───────────
            morphology = compute_cell_morphology(masks)
            if qc_enabled:
                filtered_masks, morphology, rej_counts = apply_qc_filters(
                    masks, qc_config, morphology
                )
                total_rejected += rej_counts.get("total_rejected", 0)
                # Detailed per-filter breakdown
                breakdown = []
                for key in ("border", "area_small", "area_large", "solidity",
                            "eccentricity", "circularity", "aspect_ratio"):
                    count = rej_counts.get(key, 0)
                    if count > 0:
                        breakdown.append(f"{key}={count}")
                detail = f" ({', '.join(breakdown)})" if breakdown else ""
                print(
                    f"[CellQuant] QC {cond_name}/{base_name}: "
                    f"kept {rej_counts.get('total_kept', 0)}, "
                    f"rejected {rej_counts.get('total_rejected', 0)}{detail}"
                )
                # Send structured QC data to frontend
                task.progress_data = {
                    "qc": {
                        "condition": cond_name,
                        "image_set": base_name,
                        "total": rej_counts.get("total_kept", 0) + rej_counts.get("total_rejected", 0),
                        "kept": rej_counts.get("total_kept", 0),
                        "rejected": rej_counts.get("total_rejected", 0),
                        "border": rej_counts.get("border", 0),
                        "area_small": rej_counts.get("area_small", 0),
                        "area_large": rej_counts.get("area_large", 0),
                        "solidity": rej_counts.get("solidity", 0),
                        "eccentricity": rej_counts.get("eccentricity", 0),
                        "circularity": rej_counts.get("circularity", 0),
                        "aspect_ratio": rej_counts.get("aspect_ratio", 0),
                    }
                }
            else:
                filtered_masks = masks

            # ── Stage 2: Load marker images + preprocess ──────────
            channels = image_sets.get(base_name, {})
            marker_images = {}
            for suffix, name in zip(marker_suffixes, marker_names):
                path = channels.get(suffix) or channels.get(suffix.upper())
                if path:
                    img = load_image(path)
                    if has_preprocessing:
                        img = correct_image(img, pp_config)
                    marker_images[name] = img

            if not marker_images:
                processed += 1
                continue

            # ── Stage 3: Background estimation ────────────────────
            backgrounds = {}
            per_cell_bgs = {}
            bg_stds = {}
            actual_bg_method = ""

            for name, img in marker_images.items():
                bg_result = estimate_background(
                    img,
                    filtered_masks,
                    method=bg_method,
                    negative_control=negative_control,
                    manual_bg=manual_bg,
                )
                backgrounds[name] = bg_result.global_value
                bg_stds[name] = bg_result.background_std
                if bg_result.per_cell_values is not None:
                    per_cell_bgs[name] = bg_result.per_cell_values
                actual_bg_method = bg_result.method_used
                has_per_cell = bg_result.per_cell_values is not None
                # Diagnostic: compare bg to cell signal
                cell_median = float(np.median(img[filtered_masks > 0].astype(np.float64)))
                bg_vs_cell = ""
                if bg_result.global_value > cell_median:
                    bg_vs_cell = f" | WARNING: bg ({bg_result.global_value:.0f}) > cell median ({cell_median:.0f})"
                else:
                    bg_vs_cell = f" | cell median={cell_median:.0f}"
                print(
                    f"[CellQuant] BG {cond_name}/{base_name} [{name}]: "
                    f"method={bg_result.method_used}, "
                    f"value={bg_result.global_value:.1f}, "
                    f"std={bg_result.background_std:.1f}, "
                    f"per_cell={'yes' if has_per_cell else 'no'}"
                    f"{' | ' + bg_result.method_reason if bg_result.method_reason else ''}"
                    f"{bg_vs_cell}"
                )

            # ── Stage 4: Enhanced CTCF ────────────────────────────
            results = quantify_multiple_markers(
                marker_images=marker_images,
                masks=filtered_masks,
                backgrounds=backgrounds,
                per_cell_backgrounds_map=per_cell_bgs if per_cell_bgs else None,
                background_stds=bg_stds,
                mitochondrial_markers=mitochondrial_markers,
            )

            df = results_to_dataframe(
                results=results,
                condition=cond_name,
                image_set=base_name,
                morphology=morphology,
            )
            if len(df) > 0:
                df["background_method"] = actual_bg_method
                all_dfs.append(df)

            processed += 1

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)

        # ── Stage 5: Outlier detection ────────────────────────────
        combined = flag_outliers_in_dataframe(combined, threshold=outlier_threshold)

        session.results_df = combined
        session.save_results()
        n_outliers = 0
        outlier_cols = [c for c in combined.columns if c.startswith("is_outlier_")]
        if outlier_cols:
            n_outliers = int(combined[outlier_cols].any(axis=1).sum())
        return {
            "total_cells": len(combined),
            "conditions": len(session.masks),
            "qc_rejected": total_rejected,
            "outliers_flagged": n_outliers,
        }

    return {"total_cells": 0, "conditions": 0, "qc_rejected": 0, "outliers_flagged": 0}


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
