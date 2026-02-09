"""Track visualization and lineage utilities."""

from typing import Dict, List, Set, Tuple
import numpy as np


def get_track_colors(n_tracks: int, seed: int = 42) -> Dict[int, Tuple[int, int, int]]:
    """Generate deterministic colors for track IDs."""
    rng = np.random.RandomState(seed)
    colors = {}
    for i in range(n_tracks):
        colors[i + 1] = tuple(rng.randint(50, 230, size=3).tolist())
    return colors


def build_lineage_tree(track_graph) -> List[dict]:
    """
    Build a lineage tree from a Trackastra track graph.

    Returns list of nodes with parent/children relationships.
    """
    nodes = []
    try:
        for node in track_graph.nodes():
            parent = None
            children = []
            for pred in track_graph.predecessors(node):
                parent = pred
            for succ in track_graph.successors(node):
                children.append(succ)

            nodes.append({
                "id": node,
                "parent": parent,
                "children": children,
                "is_division": len(children) > 1,
            })
    except Exception:
        pass
    return nodes


def tracked_masks_to_colored(
    masks: np.ndarray, alpha: int = 180
) -> np.ndarray:
    """Convert tracked masks to RGBA with consistent per-track colors."""
    rgba = np.zeros((*masks.shape, 4), dtype=np.uint8)

    for track_id in np.unique(masks):
        if track_id == 0:
            continue
        r = (track_id * 67 + 13) % 256
        g = (track_id * 137 + 43) % 256
        b = (track_id * 209 + 97) % 256
        pixels = masks == track_id
        rgba[pixels] = [r, g, b, alpha]

    return rgba


def compute_track_paths(
    tracked_masks_stack: np.ndarray,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Compute centroid paths for each track across frames.

    Returns dict mapping track_id -> [(frame, centroid_y, centroid_x), ...]
    """
    from scipy.ndimage import center_of_mass

    n_frames = tracked_masks_stack.shape[0]
    paths: Dict[int, List[Tuple[int, float, float]]] = {}

    for t in range(n_frames):
        frame = tracked_masks_stack[t]
        unique_ids = np.unique(frame)
        unique_ids = unique_ids[unique_ids != 0]

        for track_id in unique_ids:
            mask = frame == track_id
            cy, cx = center_of_mass(mask)

            if track_id not in paths:
                paths[track_id] = []
            paths[track_id].append((t, float(cy), float(cx)))

    return paths
