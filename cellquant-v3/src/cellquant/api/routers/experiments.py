"""Experiment scanning and condition management."""

import asyncio
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from fastapi import APIRouter, HTTPException

from cellquant.api.dependencies import get_session_manager, get_session
from cellquant.api.schemas.experiments import (
    ScanRequest,
    ScanResponse,
    ConditionInfo,
    ImageSetInfo,
    DetectionResult,
    ChannelConfigSchema,
    SetOutputRequest,
    PreprocessingRequest,
)


def _wavelength_to_color(nm: float) -> str:
    """Map emission wavelength (nm) to a conventional fluorescence false-color."""
    if nm <= 0:
        return "#FFFFFF"       # transmitted light / brightfield → white
    if nm < 450:
        return "#7B2FBE"       # violet (BFP, ~400-450nm)
    if nm < 500:
        return "#4488FF"       # blue (DAPI ~461nm, Hoechst ~470nm, CFP)
    if nm < 560:
        return "#00CC44"       # green (GFP ~509nm, FITC ~525nm, Alexa488)
    if nm < 600:
        return "#FFD700"       # yellow (YFP ~527em, but often exc ~560)
    if nm < 640:
        return "#FF6600"       # orange (TRITC ~573nm, Cy3 ~570nm, Alexa555)
    return "#FF4444"           # red (Cy5 ~670nm, mCherry ~610nm, Alexa647)


_NAME_TO_COLOR: dict[str, str] = {
    # Nuclear / UV
    "dapi": "#4488FF", "hoechst": "#4488FF", "nuclear": "#4488FF", "nuc": "#4488FF",
    # Green
    "gfp": "#00CC44", "egfp": "#00CC44", "fitc": "#00CC44", "alexa488": "#00CC44", "488": "#00CC44",
    # Yellow/orange
    "yfp": "#FFD700", "cfp": "#7B2FBE",
    # Orange/red (Cy3, mRFP, TRITC)
    "cy3": "#FF6600", "tritc": "#FF6600", "555": "#FF6600", "561": "#FF6600",
    "mrfp": "#FF6600", "rfp": "#FF6600", "mcherry": "#FF6600", "cherry": "#FF6600",
    # Far red
    "cy5": "#FF4444", "alexa647": "#FF4444", "647": "#FF4444",
    # Brightfield / transmitted
    "bf": "#FFFFFF", "brightfield": "#FFFFFF", "dic": "#FFFFFF", "phase": "#FFFFFF", "tl": "#FFFFFF",
}

def _color_from_name(suffix: str) -> str | None:
    """Infer fluorescence color from channel name keywords."""
    lower = suffix.lower()
    # Check whole name first
    if lower in _NAME_TO_COLOR:
        return _NAME_TO_COLOR[lower]
    # Check individual words (handles "DAPI Imaging", "GFP Imaging", etc.)
    for word in lower.split():
        if word in _NAME_TO_COLOR:
            return _NAME_TO_COLOR[word]
    return None


def _extract_wavelengths(image_sets: dict, first_n: int = 1) -> dict[str, float]:
    """Extract emission wavelengths from TIFF metadata for each channel suffix.

    Reads MetaMorph/ImageJ XML metadata from the first TIFF of each channel.
    Returns {suffix: wavelength_nm}.
    """
    wavelengths: dict[str, float] = {}
    for base_name, channels in image_sets.items():
        for suffix, filepath in channels.items():
            if suffix in wavelengths:
                continue
            try:
                import tifffile
                with tifffile.TiffFile(str(filepath)) as tif:
                    desc = tif.pages[0].tags.get("ImageDescription")
                    if not desc:
                        continue
                    root = ET.fromstring(desc.value)
                    for prop in root.iter("prop"):
                        if prop.get("id") == "wavelength":
                            wl = float(prop.get("value", "0"))
                            if wl >= 0:
                                wavelengths[suffix] = wl
                            break
            except Exception:
                pass
        if len(wavelengths) >= len(channels):
            break  # got all suffixes
    return wavelengths

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("/browse")
async def browse_folder() -> dict:
    """Open a native OS folder picker dialog and return the selected path.

    Falls back gracefully if no display is available (remote access).
    """
    result: dict = {"path": None}

    def _open_dialog():
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.focus_force()
            path = filedialog.askdirectory(title="Select Experiment Folder")
            root.destroy()
            if path:
                result["path"] = path
        except Exception:
            result["path"] = None

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _open_dialog)
    return {"path": result["path"]}


@router.post("/list-dir")
async def list_directory(req: dict) -> dict:
    """List subdirectories and drives for the web-based folder picker.

    Body: { "path": "C:/some/folder" }  — omit or null for drive roots.
    """
    import platform

    raw_path = req.get("path")

    # No path → list drive roots (Windows) or filesystem root (Unix)
    if not raw_path:
        if platform.system() == "Windows":
            import string
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if Path(drive).exists():
                    drives.append({"name": f"{letter}:", "path": drive, "is_dir": True})
            return {"path": "", "parent": None, "entries": drives}
        else:
            raw_path = "/"

    p = Path(raw_path)
    if not p.exists():
        raise HTTPException(400, f"Path does not exist: {raw_path}")
    if not p.is_dir():
        raise HTTPException(400, f"Not a directory: {raw_path}")

    entries = []
    try:
        for child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                entries.append({
                    "name": child.name,
                    "path": str(child).replace("\\", "/"),
                    "is_dir": True,
                })
    except PermissionError:
        pass  # skip inaccessible dirs

    parent = str(p.parent).replace("\\", "/") if p.parent != p else None

    return {
        "path": str(p).replace("\\", "/"),
        "parent": parent,
        "entries": entries,
    }


@router.post("/scan", response_model=ScanResponse)
async def scan_experiment(req: ScanRequest):
    """Scan a folder for experimental conditions and auto-detect channels."""
    folder = Path(req.path)
    if not folder.is_dir():
        raise HTTPException(400, f"Not a directory: {req.path}")

    # Determine output directory
    if req.output_path:
        output_dir = Path(req.output_path)
    else:
        output_dir = folder.parent / f"{folder.name}_output"

    manager = get_session_manager()
    session = manager.create_session(output_dir=output_dir)

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

    # Pre-render all images in background for instant browsing
    from cellquant.tiles.thumbnail import prerender_all
    prerender_all(session)

    # Build detection summary from first condition with data
    det_response = None
    if conditions:
        first_cond = list(session.conditions.values())[0]
        image_sets_raw = first_cond.get("image_sets", {})

        # Extract wavelengths from TIFF metadata
        wavelengths = _extract_wavelengths(image_sets_raw)
        channel_colors = {s: _wavelength_to_color(w) for s, w in wavelengths.items()}

        # For any suffix without wavelength metadata, infer color from name
        all_suffixes = first_cond.get("channel_suffixes", [])
        for suffix in all_suffixes:
            if suffix not in channel_colors:
                inferred = _color_from_name(suffix)
                if inferred:
                    channel_colors[suffix] = inferred

        det_response = DetectionResult(
            channel_suffixes=first_cond.get("channel_suffixes", []),
            n_channels=len(first_cond.get("channel_suffixes", [])),
            n_image_sets=sum(c.n_image_sets for c in conditions),
            n_complete=sum(c.n_image_sets for c in conditions),
            n_incomplete=0,
            suggested_nuclear=first_cond.get("nuclear_suffix"),
            suggested_cyto=first_cond.get("cyto_suffix"),
            suggested_markers=first_cond.get("suggested_markers", []),
            channel_wavelengths=wavelengths,
            channel_colors=channel_colors,
        )

    return ScanResponse(
        session_id=session.id,
        conditions=conditions,
        detection=det_response,
        output_path=str(session.directory),
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


@router.post("/{session_id}/set-output")
async def set_output_path(session_id: str, req: SetOutputRequest):
    """Change the session output directory, moving existing data."""
    import shutil

    session = get_session(session_id)
    old_dir = session.directory
    new_dir = Path(req.output_path)

    if old_dir == new_dir:
        return {"status": "ok", "output_path": str(new_dir)}

    new_dir.mkdir(parents=True, exist_ok=True)

    # Move existing contents
    if old_dir.exists():
        for item in old_dir.iterdir():
            dest = new_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        # Clean up old directory if empty
        try:
            old_dir.rmdir()
        except OSError:
            pass

    session.directory = new_dir
    session.save_state()
    return {"status": "ok", "output_path": str(new_dir)}


@router.post("/{session_id}/open-folder")
async def open_folder(session_id: str, body: dict):
    """Open a result folder in the OS file explorer."""
    import subprocess, sys

    session = get_session(session_id)
    condition = body.get("condition", "")
    base_name = body.get("base_name", "")

    folder = session.directory / condition / base_name
    if not folder.is_dir():
        # Fall back to condition folder, then session root
        folder = session.directory / condition if (session.directory / condition).is_dir() else session.directory

    if sys.platform == "win32":
        subprocess.Popen(["explorer", str(folder)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])

    return {"status": "ok", "path": str(folder)}


@router.post("/{session_id}/preprocessing")
async def configure_preprocessing(session_id: str, req: PreprocessingRequest):
    """Load dark-frame and flat-field calibration images."""
    from cellquant.core.preprocessing.calibration import (
        load_dark_frames,
        load_flat_field,
        validate_calibration_frame,
    )

    session = get_session(session_id)
    warnings = []

    if req.dark_frame_paths:
        try:
            dark = load_dark_frames([Path(p) for p in req.dark_frame_paths])
            session.dark_master = dark
        except Exception as e:
            warnings.append(f"Dark frame loading failed: {e}")

    if req.flat_field_paths:
        try:
            flat = load_flat_field([Path(p) for p in req.flat_field_paths])
            session.flat_norm = flat
        except Exception as e:
            warnings.append(f"Flat field loading failed: {e}")

    return {
        "status": "ok",
        "has_dark": session.dark_master is not None,
        "has_flat": session.flat_norm is not None,
        "warnings": warnings,
    }


@router.post("/{session_id}/configure")
async def configure_channels(session_id: str, config: ChannelConfigSchema):
    """Update channel role assignments for the session."""
    session = get_session(session_id)
    session.channel_config = {
        "nuclear_suffix": config.nuclear_suffix,
        "cyto_suffix": config.cyto_suffix,
        "marker_suffixes": config.marker_suffixes,
        "marker_names": config.marker_names,
        "mitochondrial_markers": config.mitochondrial_markers,
    }
    session.save_state()
    return {"status": "ok"}
