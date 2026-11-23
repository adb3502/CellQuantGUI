"""Tell a time-lapse from a z-stack from a single image.

Both a video and a z-stack arrive as a 3-D array, so dimensionality alone
cannot separate them - and getting it wrong is costly in both directions:
collapsing a video to one frame throws away ~99% of the data, while treating
a z-stack as a video produces meaningless "traces" down the optical axis.

The axis letters are what distinguish them. SlideBook does NOT tag time: it
writes 'I' (generic sequence) for a 120-frame recording, so a rule keyed on
'T' misses every real video this lab produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class DataType(str, Enum):
    IMAGE = "image"
    TIMELAPSE = "timelapse"
    ZSTACK = "zstack"


# Axes that mean "depth", not "time". A stack along these is a z-stack, where
# taking a middle slice is a legitimate projection choice.
_DEPTH_AXES = frozenset("Z")

# Axes that are neither time nor depth: channels, RGB samples, and the spatial
# axes themselves. A stack along these is a still image.
_NON_SEQUENCE_AXES = frozenset("CSYXA")

# Axes that mean "a sequence of frames". 'I' is tifffile's generic sequence
# axis and is what SlideBook writes for a time-lapse.
_TIME_AXES = frozenset("TI")


def classify_axes(axes: Optional[str], n_pages: int) -> DataType:
    """Classify from tifffile's axis string and page count.

    A single page is always an image whatever the axes claim: one frame is
    not a time series, and one slice is not a stack.
    """
    if n_pages <= 1:
        return DataType.IMAGE

    if not axes:
        # No axis metadata at all. Many pages is far more often a recording
        # than anything else, and being loud about it beats silently
        # discarding frames.
        return DataType.TIMELAPSE

    upper = axes.upper()
    leading = next((a for a in upper if a not in ("Y", "X")), None)

    if leading is None:
        return DataType.IMAGE
    if leading in _DEPTH_AXES:
        return DataType.ZSTACK
    if leading in _TIME_AXES:
        return DataType.TIMELAPSE
    if leading in _NON_SEQUENCE_AXES:
        return DataType.IMAGE

    # Unrecognised leading axis with real depth: treat as frames so the
    # caller warns, rather than collapsing without a word.
    return DataType.TIMELAPSE


@dataclass
class DataTypeInfo:
    """What one file is."""

    data_type: DataType = DataType.IMAGE
    n_frames: int = 1
    axes: Optional[str] = None
    shape: tuple = ()
    channel_name: Optional[str] = None
    objective: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_timelapse(self) -> bool:
        return self.data_type == DataType.TIMELAPSE


@dataclass
class DataTypeSummary:
    """What a whole folder is."""

    data_type: DataType = DataType.IMAGE
    n_frames_typical: int = 1
    mixed: bool = False
    counts: dict = None
    channel_name: Optional[str] = None
    objective: Optional[str] = None

    def __post_init__(self):
        if self.counts is None:
            self.counts = {}


def probe_file(path: Union[str, Path]) -> DataTypeInfo:
    """Read one file's shape and axes WITHOUT decoding pixel data.

    These files are 63 MB each and a scan touches every one of them, so this
    opens the TIFF directory only. Any failure returns IMAGE with the reason
    attached: one unreadable file must never abort a folder scan.
    """
    import tifffile

    path = Path(path)
    try:
        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            axes = getattr(series, "axes", None)
            shape = tuple(getattr(series, "shape", ()) or ())
            n_pages = len(tif.pages)

            channel_name = objective = None
            try:
                from cellquant.core.io.image_loader import _read_slidebook_tags

                channel_name, objective = _read_slidebook_tags(tif)
            except Exception:
                pass

        # Frame count comes from the leading axis when there is one, since a
        # page count can include channel pages.
        n_frames = shape[0] if (shape and len(shape) >= 3) else 1
        n_frames = max(int(n_frames), 1)
        if n_pages > 1 and n_frames == 1:
            n_frames = n_pages

        dt = classify_axes(axes, max(n_pages, n_frames))
        if dt != DataType.TIMELAPSE and dt != DataType.ZSTACK:
            n_frames = 1

        return DataTypeInfo(
            data_type=dt, n_frames=n_frames, axes=axes, shape=shape,
            channel_name=channel_name, objective=objective,
        )
    except Exception as exc:
        return DataTypeInfo(error=f"{type(exc).__name__}: {exc}")


def summarise_data_types(infos) -> DataTypeSummary:
    """Collapse per-file findings into one answer for a folder."""
    from collections import Counter

    infos = list(infos)
    if not infos:
        return DataTypeSummary()

    counts = Counter(i.data_type for i in infos)
    mixed = len(counts) > 1

    # When a folder mixes kinds, report the one that costs most to get wrong:
    # a time-lapse silently collapsed loses almost all of its data.
    if DataType.TIMELAPSE in counts:
        chosen = DataType.TIMELAPSE
    elif DataType.ZSTACK in counts:
        chosen = DataType.ZSTACK
    else:
        chosen = DataType.IMAGE

    of_kind = [i for i in infos if i.data_type == chosen]
    frames = sorted(i.n_frames for i in of_kind) or [1]
    typical = frames[len(frames) // 2]

    first = of_kind[0] if of_kind else infos[0]
    return DataTypeSummary(
        data_type=chosen,
        n_frames_typical=max(int(typical), 1),
        mixed=mixed,
        counts={k.value: v for k, v in counts.items()},
        channel_name=first.channel_name,
        objective=first.objective,
    )
