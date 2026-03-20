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

    # Filter to only fields that SegmentationParams accepts
    import dataclasses
    _param_fields = {f.name for f in dataclasses.fields(SegmentationParams)}
    params = SegmentationParams(**{k: v for k, v in seg_params.items() if k in _param_fields})
    custom_model_path = seg_params.get("custom_model_path")
    print(f"[CellQuant] Initializing Cellpose engine: model={params.model_type}, gpu={params.use_gpu}, batch={params.batch_size}")
    engine = CellposeEngine(
        model_type=params.model_type,
        use_gpu=params.use_gpu,
        batch_size=params.batch_size,
        custom_model_path=custom_model_path,
    )
    print(f"[CellQuant] Engine created, loading model...")

    # Clear cached renders so previews use fresh channel config
    import shutil
    renders_dir = session.directory / "renders"
    if renders_dir.exists():
        shutil.rmtree(renders_dir, ignore_errors=True)

    total_images = sum(
        len(cond.get("image_sets", {})) for cond in session.conditions.values()
    )
    task.total = total_images
    processed = 0
    total_cells = 0

    # Cache engines by model_type to avoid reloading identical models
    engine_cache: dict = {params.model_type: engine}

    # Nuclear segmentation for preview
    has_any_nuclear = any(
        cond.get("nuclear_suffix") for cond in session.conditions.values()
    )
    nuclear_seg_engine = None
    if has_any_nuclear:
        nuclear_seg_engine = CellposeEngine(
            model_type="nuclei",
            use_gpu=params.use_gpu,
            batch_size=params.batch_size,
        )
        print(f"[CellQuant] Nuclear engine initialized for preview (model=nuclei)")

    for cond_name, cond_data in session.conditions.items():
        if task.status == "cancelled":
            break

        image_sets = cond_data.get("image_sets", {})
        nuclear_suffix = cond_data.get("nuclear_suffix", "C0")
        cyto_suffix = cond_data.get("cyto_suffix", "C1")

        # Apply per-condition overrides if any
        cond_overrides = seg_params.get("condition_overrides", {}).get(cond_name, {})
        cond_diameter = cond_overrides.get("diameter", params.diameter)
        cond_flow = cond_overrides.get("flow_threshold", params.flow_threshold)
        cond_cellprob = cond_overrides.get("cellprob_threshold", params.cellprob_threshold)
        cond_min_size = cond_overrides.get("min_size", params.min_size)
        cond_pre_smooth = cond_overrides.get("pre_smooth_sigma") or 0

        # Per-condition model type (lazy-loads new engine if different from global)
        cond_model = cond_overrides.get("model_type") or params.model_type
        if cond_model not in engine_cache:
            print(f"[CellQuant] Loading per-condition model '{cond_model}' for {cond_name}")
            engine_cache[cond_model] = CellposeEngine(
                model_type=cond_model,
                use_gpu=params.use_gpu,
                batch_size=params.batch_size,
            )
        active_engine = engine_cache[cond_model]

        # User-selected segmentation channels (per-condition or global)
        seg_suffixes = cond_overrides.get("segmentation_suffixes") or seg_params.get("segmentation_suffixes")

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

            # Build segmentation input from user-selected channels (or default nuclear+cyto)
            if seg_suffixes:
                # Use exactly the channels the user selected for segmentation
                loaded_channels = []
                for suffix in seg_suffixes:
                    ch_path = channels.get(suffix) or channels.get(suffix.upper()) or channels.get(suffix.lower())
                    if ch_path:
                        img = load_image(ch_path)
                        loaded_channels.append(normalize_image(img))
                if not loaded_channels:
                    print(f"[CellQuant] Skipping {cond_name}/{base_name} (no matching channels for {seg_suffixes})")
                    processed += 1
                    continue
                if len(loaded_channels) == 1:
                    seg_input = loaded_channels[0]
                else:
                    seg_input = np.stack(loaded_channels, axis=0)
            else:
                # Default: cyto/brightfield channel only
                cyto_path = channels.get(cyto_suffix) or channels.get(cyto_suffix.upper())
                if cyto_path:
                    cyto_img = load_image(cyto_path)
                    seg_input = normalize_image(cyto_img)
                else:
                    # Fallback to nuclear if no cyto available
                    nuc_path = channels.get(nuclear_suffix) or channels.get(nuclear_suffix.upper())
                    if not nuc_path:
                        processed += 1
                        continue
                    nuc_img = load_image(nuc_path)
                    seg_input = normalize_image(nuc_img)

            # Pre-smoothing: Gaussian blur fills in hollow/ring-shaped nuclei
            if cond_pre_smooth and cond_pre_smooth > 0:
                from scipy.ndimage import gaussian_filter
                if seg_input.ndim == 2:
                    seg_input = gaussian_filter(seg_input.astype(np.float32), sigma=cond_pre_smooth)
                else:
                    seg_input = np.stack(
                        [gaussian_filter(seg_input[i].astype(np.float32), sigma=cond_pre_smooth)
                         for i in range(seg_input.shape[0])], axis=0
                    )
                print(f"[CellQuant]   -> pre-smoothed with sigma={cond_pre_smooth}")

            print(f"[CellQuant] Segmenting {cond_name}/{base_name} (model={cond_model})...")
            result = active_engine.segment_single(
                seg_input,
                diameter=cond_diameter,
                flow_threshold=cond_flow,
                cellprob_threshold=cond_cellprob,
                min_size=cond_min_size,
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
            image_set_dir = session.directory / cond_name / base_name
            image_set_dir.mkdir(parents=True, exist_ok=True)
            try:
                from skimage import io as skio

                seg_type = params.model_type

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

            # Nuclear segmentation preview
            if nuclear_seg_engine and nuclear_suffix:
                nuc_path = channels.get(nuclear_suffix) or channels.get(nuclear_suffix.upper())
                if nuc_path:
                    try:
                        nuc_img = load_image(nuc_path)
                        nuc_result = nuclear_seg_engine.segment_single(
                            normalize_image(nuc_img),
                            diameter=0,
                            flow_threshold=params.flow_threshold,
                            cellprob_threshold=params.cellprob_threshold,
                        )
                        nuclear_masks = nuc_result.masks
                        # Save nuclear masks
                        nuc_mask_path = session.get_nuclear_mask_path(cond_name, base_name)
                        np.save(nuc_mask_path, nuclear_masks)
                        # Save nuclear overlays
                        nuc_uint8 = imagej_auto_contrast(nuc_img.astype(np.float64))
                        from PIL import Image as PILImage
                        nuc_outline = _create_outline_overlay(nuc_uint8, nuclear_masks)
                        PILImage.fromarray(nuc_outline, "RGB").save(
                            image_set_dir / f"{base_name}_nuclear_outline.png"
                        )
                        nuc_filled = _create_simple_overlay(nuc_uint8, nuclear_masks, alpha=0.5)
                        PILImage.fromarray(nuc_filled, "RGB").save(
                            image_set_dir / f"{base_name}_nuclear_filled.jpg"
                        )
                        print(f"[CellQuant]   -> Nuclear: {nuc_result.n_cells} nuclei")
                        task.progress_data = {"has_nuclear": True}
                    except Exception as e:
                        print(f"[CellQuant] Warning: nuclear seg failed for {base_name}: {e}")

            processed += 1

    for eng in engine_cache.values():
        eng.clear_gpu_memory()
    if nuclear_seg_engine:
        nuclear_seg_engine.clear_gpu_memory()
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

    # ── Nuclear engine for mito correction ───────────────────────────────
    # If any mitochondrial markers are flagged, run a second Cellpose pass
    # using the 'nuclei' model on the DAPI/nuclear channel per image set,
    # then subtract that nuclear signal from the whole-cell CTCF.
    nuclear_engine = None
    nuclear_suffix_global = (
        (session.channel_config or {}).get("nuclear_suffix")
        or next(
            (cond.get("nuclear_suffix") for cond in session.conditions.values() if cond.get("nuclear_suffix")),
            None,
        )
    )
    if mitochondrial_markers and nuclear_suffix_global:
        from cellquant.core.segmentation.cellpose_engine import CellposeEngine
        use_gpu = quant_params.get("use_gpu", False)
        print(
            f"[CellQuant] Initializing nuclear engine for mito correction "
            f"(model=nuclei, gpu={use_gpu}, nuclear_channel={nuclear_suffix_global})"
        )
        nuclear_engine = CellposeEngine(model_type="nuclei", use_gpu=use_gpu)
    elif mitochondrial_markers:
        print("[CellQuant] Warning: mitochondrial markers flagged but no nuclear channel configured — skipping nuclear subtraction")

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

            # ── Stage 3b: Nuclear segmentation for mito correction ───────
            nuclear_masks_for_quant = None
            if nuclear_engine is not None:
                nuc_path = channels.get(nuclear_suffix_global) or channels.get(nuclear_suffix_global.upper())
                if nuc_path:
                    from cellquant.core.io.image_loader import normalize_image
                    nuc_img = load_image(nuc_path)
                    if has_preprocessing:
                        nuc_img = correct_image(nuc_img, pp_config)
                    nuc_result = nuclear_engine.segment_single(
                        normalize_image(nuc_img),
                        diameter=0,  # auto-detect nuclear diameter
                    )
                    nuclear_masks_for_quant = nuc_result.masks
                    print(
                        f"[CellQuant] Nuclear seg {cond_name}/{base_name}: "
                        f"{nuc_result.n_cells} nuclei detected"
                    )
                else:
                    print(
                        f"[CellQuant] Warning: nuclear channel '{nuclear_suffix_global}' not found "
                        f"in {cond_name}/{base_name} — mito correction skipped for this image"
                    )

            # ── Stage 4: Enhanced CTCF ────────────────────────────
            results = quantify_multiple_markers(
                marker_images=marker_images,
                masks=filtered_masks,
                backgrounds=backgrounds,
                per_cell_backgrounds_map=per_cell_bgs if per_cell_bgs else None,
                background_stds=bg_stds,
                mitochondrial_markers=mitochondrial_markers,
                nuclear_masks=nuclear_masks_for_quant,
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

    if nuclear_engine is not None:
        nuclear_engine.clear_gpu_memory()

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
