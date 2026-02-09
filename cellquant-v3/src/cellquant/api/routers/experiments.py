"""Experiment scanning and condition management."""

from pathlib import Path
from fastapi import APIRouter, HTTPException

from cellquant.api.dependencies import get_session_manager, get_session
from cellquant.api.schemas.experiments import (
    ScanRequest,
    ScanResponse,
    ConditionInfo,
    ImageSetInfo,
    DetectionResult,
)

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("/scan", response_model=ScanResponse)
async def scan_experiment(req: ScanRequest):
    """Scan a folder for experimental conditions and auto-detect channels."""
    folder = Path(req.path)
    if not folder.is_dir():
        raise HTTPException(400, f"Not a directory: {req.path}")

    manager = get_session_manager()
    session = manager.create_session()

    from cellquant.core.io.channel_detect import detect_image_sets

    conditions = []
    all_tiff_files = []

    for subdir in sorted(folder.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue

        tiff_files = [
            f
            for f in subdir.iterdir()
            if f.suffix.lower() in (".tif", ".tiff")
        ]

        if not tiff_files:
            continue

        detection = detect_image_sets(tiff_files)
        all_tiff_files.extend(tiff_files)

        image_set_infos = []
        for base_name, channels in detection.image_sets.items():
            image_set_infos.append(
                ImageSetInfo(
                    base_name=base_name,
                    channels={k: str(v) for k, v in channels.items()},
                )
            )

        cond = ConditionInfo(
            name=subdir.name,
            path=str(subdir),
            n_image_sets=detection.n_image_sets,
            image_sets=image_set_infos,
        )
        conditions.append(cond)

        # Store in session
        session.conditions[subdir.name] = {
            "name": subdir.name,
            "path": str(subdir),
            "n_images": detection.n_image_sets,
            "image_sets": {
                bn: {k: str(v) for k, v in chs.items()}
                for bn, chs in detection.image_sets.items()
            },
            "channel_suffixes": detection.channel_suffixes,
            "nuclear_suffix": detection.suggested_nuclear,
            "cyto_suffix": detection.suggested_cyto,
            "suggested_markers": detection.suggested_markers,
        }

    session.experiment_path = folder
    session.save_state()

    # Build detection summary from first condition with data
    det_response = None
    if conditions:
        first_cond = list(session.conditions.values())[0]
        det_response = DetectionResult(
            channel_suffixes=first_cond.get("channel_suffixes", []),
            n_channels=len(first_cond.get("channel_suffixes", [])),
            n_image_sets=sum(c.n_image_sets for c in conditions),
            n_complete=sum(c.n_image_sets for c in conditions),
            n_incomplete=0,
            confidence=0.95,
            suggested_nuclear=first_cond.get("nuclear_suffix"),
            suggested_cyto=first_cond.get("cyto_suffix"),
            suggested_markers=first_cond.get("suggested_markers", []),
        )

    return ScanResponse(
        session_id=session.id,
        conditions=conditions,
        detection=det_response,
    )


@router.get("/{session_id}", response_model=ScanResponse)
async def get_experiment(session_id: str):
    """Get experiment state for a session."""
    session = get_session(session_id)

    conditions = []
    for name, data in session.conditions.items():
        image_sets = []
        for bn, chs in data.get("image_sets", {}).items():
            image_sets.append(ImageSetInfo(base_name=bn, channels=chs))
        conditions.append(
            ConditionInfo(
                name=name,
                path=data.get("path", ""),
                n_image_sets=data.get("n_images", 0),
                image_sets=image_sets,
            )
        )

    return ScanResponse(session_id=session.id, conditions=conditions)
