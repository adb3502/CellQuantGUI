"""
Smart channel detection for microscopy TIFF files.

Uses position-based tokenization + frequency analysis to automatically detect:
- Which filename token position represents the channel identifier
- How many channels and image sets exist
- Which files are incomplete or orphaned

Works regardless of naming convention or channel position in the filename.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import re

# Known channel-like names (case-insensitive matching)
KNOWN_CHANNEL_NAMES = {
    "dapi", "hoechst", "nuclear", "nuc",
    "gfp", "egfp", "fitc", "alexa488", "488",
    "rfp", "mcherry", "cherry", "tritc", "cy3", "alexa555", "555", "561",
    "cy5", "alexa647", "647",
    "bf", "brightfield", "dic", "phase", "tl",
    "cfp", "yfp", "bfp",
}


@dataclass
class ImageSetDetection:
    """Result of auto-detecting image sets and channels from TIFF filenames."""
    channel_suffixes: List[str]
    n_channels: int
    n_image_sets: int
    n_complete: int
    n_incomplete: int
    n_orphan_files: int
    image_sets: Dict[str, Dict[str, Path]]      # {base_name: {suffix: filepath}}
    incomplete_info: Dict[str, List[str]]        # {base_name: [missing_suffixes]}
    orphan_files: List[Path]
    confidence: float
    suggested_nuclear: Optional[str] = None
    suggested_cyto: Optional[str] = None
    suggested_markers: List[str] = field(default_factory=list)
    channel_position: Optional[int] = None       # which token position is the channel


def _strip_tiff_ext(name: str) -> str:
    """Remove .tif or .tiff extension (case-insensitive)."""
    lower = name.lower()
    if lower.endswith(".tiff"):
        return name[:-5]
    if lower.endswith(".tif"):
        return name[:-4]
    return name


def _tokenize(stem: str) -> List[str]:
    """Split filename stem by underscores, filtering empty tokens."""
    return [t for t in stem.split("_") if t]


def _is_channel_like(value: str) -> float:
    """
    Score how channel-like a string is. Returns 0.0-1.0.

    High scores: w1, C0, ch2, DAPI, GFP, 488
    Low scores: A04, sample, sx, 1, experiment
    """
    v = value.strip().lower()

    # Known channel names → definitive
    if v in KNOWN_CHANNEL_NAMES:
        return 1.0

    # Common channel prefixes: w, c, ch followed by 1 digit
    # (w1, w2, C0, C1, ch1, ch2)
    if re.match(r'^(w|c|ch)\d$', v):
        return 0.95

    # Common channel prefixes with 2 digits: C02, w01
    if re.match(r'^(w|c|ch)\d{2}$', v):
        return 0.85

    # Generic letter + single digit (less confident): D1, F2
    if re.match(r'^[a-z]\d$', v):
        return 0.6

    # Well-plate pattern (A01-H12): these are NOT channels
    if re.match(r'^[a-h]\d{2,3}$', v):
        return 0.1

    # Wavelength-like: 488, 555, 647
    if re.match(r'^\d{3}$', v):
        try:
            if 350 <= int(v) <= 800:
                return 0.8
        except ValueError:
            pass

    # Pure single digits or very short tokens → likely coordinates, not channels
    if re.match(r'^\d{1,2}$', v):
        return 0.05

    return 0.0


def _channel_score(values: Set[str], cardinality: int, freq: int,
                   total_files: int, pos: int, n_positions: int) -> float:
    """
    Score how likely a token position is the channel identifier.
    Higher score = more likely to be the channel position.
    """
    score = 0.0

    # Must have at least 2 unique values to be a channel (need multiple channels)
    if cardinality < 2:
        return -1.0

    # Must have at most ~10 channels (more than that is unlikely)
    if cardinality > 10:
        score -= 5.0

    # Frequency uniformity: each value should appear ~the same number of times
    # Perfect uniformity: cardinality * freq == total_files
    coverage = (cardinality * freq) / total_files if total_files > 0 else 0
    if coverage > 0.8:
        score += 10.0 * coverage
    elif coverage > 0.5:
        score += 5.0 * coverage
    else:
        score -= 5.0

    # Channel-like pattern scoring (strongest signal)
    channel_scores = [_is_channel_like(v) for v in values]
    avg_channel_score = sum(channel_scores) / len(channel_scores) if channel_scores else 0

    # This is the dominant signal - weight heavily
    score += 20.0 * avg_channel_score

    # Lower cardinality is better (typically 2-6 channels vs many fields of view)
    if cardinality <= 6:
        score += 3.0
    elif cardinality <= 10:
        score += 1.0

    # Positional preference: last position slightly preferred
    if n_positions > 1:
        position_bias = (pos / (n_positions - 1)) * 2.0  # 0 to 2 points
        score += position_bias

    # Penalize if all values are pure digits without channel pattern
    all_pure_digits = all(re.match(r'^\d+$', v) and _is_channel_like(v) < 0.5 for v in values)
    if all_pure_digits:
        score -= 8.0  # Likely coordinates (sx, sy values), not channels

    return score


def detect_image_sets(tiff_files: List[Path]) -> ImageSetDetection:
    """
    Auto-detect image sets and channels from a list of TIFF files.

    Uses position-based tokenization + frequency analysis:
    1. Tokenize each filename by '_'
    2. Analyze each token position for cardinality + frequency
    3. Score positions to find the channel identifier
    4. Group files into image sets
    """
    # Handle trivial cases
    if not tiff_files:
        return ImageSetDetection(
            channel_suffixes=[], n_channels=0, n_image_sets=0,
            n_complete=0, n_incomplete=0, n_orphan_files=0,
            image_sets={}, incomplete_info={}, orphan_files=[],
            confidence=0.0,
        )

    if len(tiff_files) == 1:
        f = tiff_files[0]
        stem = _strip_tiff_ext(f.name)
        return ImageSetDetection(
            channel_suffixes=[stem], n_channels=1, n_image_sets=1,
            n_complete=1, n_incomplete=0, n_orphan_files=0,
            image_sets={stem: {stem: f}}, incomplete_info={},
            orphan_files=[], confidence=1.0,
            suggested_markers=[stem],
        )

    # ─── Phase 1: Tokenize all files ─────────────────────────────
    file_tokens = []  # [(path, tokens)]
    orphans = []

    for f in tiff_files:
        stem = _strip_tiff_ext(f.name)
        tokens = _tokenize(stem)
        if len(tokens) < 2:
            orphans.append(f)  # No underscore = can't parse into base + suffix
        else:
            file_tokens.append((f, tokens))

    if not file_tokens:
        # All files are unparseable
        return ImageSetDetection(
            channel_suffixes=[], n_channels=0, n_image_sets=len(tiff_files),
            n_complete=0, n_incomplete=0, n_orphan_files=len(tiff_files),
            image_sets={}, incomplete_info={},
            orphan_files=list(tiff_files), confidence=0.0,
        )

    # ─── Phase 2: Group by token count ────────────────────────────
    # Files with different token counts likely have different naming schemes.
    # Analyze the largest group (most common token count).
    token_count_groups = {}
    for f, tokens in file_tokens:
        n = len(tokens)
        if n not in token_count_groups:
            token_count_groups[n] = []
        token_count_groups[n].append((f, tokens))

    # Use the group with the most files
    dominant_n_tokens = max(token_count_groups, key=lambda k: len(token_count_groups[k]))
    main_group = token_count_groups[dominant_n_tokens]
    n_positions = dominant_n_tokens

    # Files not in the dominant group are orphans
    for n, group in token_count_groups.items():
        if n != dominant_n_tokens:
            orphans.extend(f for f, _ in group)

    total_in_group = len(main_group)

    # ─── Phase 3: Per-position analysis ───────────────────────────
    position_values = [[] for _ in range(n_positions)]
    for _, tokens in main_group:
        for i, token in enumerate(tokens):
            position_values[i].append(token)

    position_stats = []
    for pos in range(n_positions):
        values = position_values[pos]
        counter = Counter(values)
        unique_values = set(counter.keys())
        cardinality = len(unique_values)

        # Dominant frequency for this position (most common count)
        freq_counter = Counter(counter.values())
        dominant_freq = freq_counter.most_common(1)[0][0]

        score = _channel_score(
            unique_values, cardinality, dominant_freq,
            total_in_group, pos, n_positions
        )
        position_stats.append({
            "pos": pos,
            "values": unique_values,
            "cardinality": cardinality,
            "dominant_freq": dominant_freq,
            "counter": counter,
            "score": score,
        })

    # ─── Phase 4: Select channel position ─────────────────────────
    # Pick the position with the highest score
    best = max(position_stats, key=lambda x: x["score"])
    channel_pos = best["pos"]
    channel_suffixes = sorted(best["values"])

    # If best score is negative, we couldn't find a good channel position
    if best["score"] < 0:
        # Fallback: use last token as suffix (most common convention)
        channel_pos = n_positions - 1
        channel_suffixes = sorted(position_stats[channel_pos]["values"])

    # ─── Phase 5: Build image sets ────────────────────────────────
    image_sets: Dict[str, Dict[str, Path]] = {}

    for f, tokens in main_group:
        channel_value = tokens[channel_pos]
        # Base name = all tokens except the channel token
        base_tokens = tokens[:channel_pos] + tokens[channel_pos + 1:]
        base_name = "_".join(base_tokens)

        if base_name not in image_sets:
            image_sets[base_name] = {}
        image_sets[base_name][channel_value] = f

    # ─── Phase 6: Classify completeness ───────────────────────────
    channel_set = set(channel_suffixes)
    n_complete = 0
    n_incomplete = 0
    incomplete_info = {}

    for base_name, channels in image_sets.items():
        present = set(channels.keys())
        missing = channel_set - present
        if missing:
            n_incomplete += 1
            incomplete_info[base_name] = sorted(missing)
        else:
            n_complete += 1

    # ─── Phase 7: Confidence score ────────────────────────────────
    files_in_sets = sum(len(chs) for chs in image_sets.values())
    confidence = files_in_sets / len(tiff_files) if tiff_files else 0.0

    # ─── Phase 8: Suggest channel roles ───────────────────────────
    roles = suggest_channel_roles(channel_suffixes)

    return ImageSetDetection(
        channel_suffixes=channel_suffixes,
        n_channels=len(channel_suffixes),
        n_image_sets=len(image_sets),
        n_complete=n_complete,
        n_incomplete=n_incomplete,
        n_orphan_files=len(orphans),
        image_sets=image_sets,
        incomplete_info=incomplete_info,
        orphan_files=orphans,
        confidence=confidence,
        suggested_nuclear=roles.get("nuclear"),
        suggested_cyto=roles.get("cyto"),
        suggested_markers=roles.get("markers", []),
        channel_position=channel_pos,
    )


def suggest_channel_roles(suffixes: List[str]) -> Dict[str, object]:
    """
    Suggest which detected channel suffix is nuclear, cyto, or marker.

    Heuristics:
    1. Known names override everything (DAPI → nuclear, GFP → marker)
    2. Otherwise, sort by numeric component: first → nuclear, second → cyto, rest → markers
    """
    if not suffixes:
        return {}

    roles: Dict[str, object] = {}
    remaining = list(suffixes)

    # Pass 1: Check for known channel names
    for suffix in suffixes:
        lower = suffix.lower()
        if lower in ("dapi", "hoechst", "nuclear", "nuc"):
            roles["nuclear"] = suffix
            remaining.remove(suffix)
        elif lower in ("phalloidin", "actin", "cyto", "cytoplasm"):
            roles["cyto"] = suffix
            remaining.remove(suffix)

    # Pass 2: Sort remaining by numeric component
    def sort_key(s):
        nums = re.findall(r'\d+', s)
        return int(nums[0]) if nums else 999

    remaining.sort(key=sort_key)

    # Assign by order: first unassigned → nuclear, second → cyto, rest → markers
    if "nuclear" not in roles and remaining:
        roles["nuclear"] = remaining.pop(0)
    if "cyto" not in roles and remaining:
        roles["cyto"] = remaining.pop(0)
    roles["markers"] = remaining  # Everything else is a marker

    return roles
