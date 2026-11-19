"""Dose parsing from filenames and Hill-equation fitting.

Parsing is deliberately conservative: anything not confidently recognised is
returned unparsed for user confirmation rather than guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

_UNIT_TO_MOLAR = {
    "m": 1e0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9, "pm": 1e-12,
}

# A unit must be followed by a non-letter. Plain \b will not do: real
# filenames read "Ach 1 uM_XY1222807246_Z0_T000_C0", and '_' is a word
# character, so \b does not match after "uM" and the dose is missed.
_UNIT_END = r"(?![A-Za-z])"

# PREFIXED units (mM/uM/nM/pM) are accepted in any case, because real
# filenames write "100nm" meaning nanomolar and "epi1uM" with no separator.
# "nm"/"um" are formally ambiguous with nanometres/micrometres, but a bare
# length in a dose-series filename is far less likely than a concentration,
# and the confirmation gate shows the user what was parsed.
#
# A BARE unit must be an uppercase "M". Lowercase "m" alone is far more often
# minutes, a well id or a size ("plate_5m", "Img_3mm", "well_30m"); reading
# those as 5 M / 3 mM / 30 M would feed junk dose points into the EC50 fit.
#
# The lookbehind stops a match starting mid-number, so "T000"/"Z0" cannot
# produce a dose.
_DOSE_RE = re.compile(
    r"(?<![0-9.])(\d+(?:\.\d+)?)\s*((?:[mM][mM]|[uUµ][mM]|[nN][mM]|[pP][mM]|M))" + _UNIT_END)
_VOLUME_RE = re.compile(
    r"(?<![0-9.])(\d+(?:\.\d+)?)\s*([uUµ][lL])" + _UNIT_END)


@dataclass
class ParsedDose:
    source: str
    parsed: bool = False
    value: Optional[float] = None
    unit: Optional[str] = None
    molar: Optional[float] = None
    volume_ul: Optional[float] = None
    ambiguous: bool = False
    note: str = ""


def parse_dose(name: str) -> ParsedDose:
    """Extract a concentration from a condition or filename."""
    result = ParsedDose(source=name)

    vol = _VOLUME_RE.search(name)
    if vol:
        result.volume_ul = float(vol.group(1))

    text = _VOLUME_RE.sub(" ", name)
    match = _DOSE_RE.search(text)
    if not match:
        result.note = "No concentration recognised"
        return result

    value = float(match.group(1))
    unit = match.group(2).lower().replace("µ", "u")
    factor = _UNIT_TO_MOLAR.get(unit)
    if factor is None:
        result.note = f"Unrecognised unit {match.group(2)!r}"
        return result

    result.parsed = True
    result.value = value
    result.unit = match.group(2)
    result.molar = value * factor

    # Lowercase "mm"/"um"/"nm" are formally ambiguous with millimetres /
    # micrometres / nanometres. We read them as concentrations because that is
    # the usual intent in a dose series, but flag them so the confirmation gate
    # puts them in front of the user instead of quietly fitting them.
    if match.group(2).lower() in ("mm", "um", "nm", "pm"):
        result.ambiguous = True
        result.note = (f"Read {match.group(2)!r} as a concentration; it could "
                       "also be a length - confirm before fitting")
    return result


def parse_dose_table(names: Sequence[str]) -> List[ParsedDose]:
    """Parse many names, flagging equal concentrations that differ in volume."""
    doses = [parse_dose(n) for n in names]
    by_molar: dict = {}
    for d in doses:
        if d.parsed and d.molar is not None:
            by_molar.setdefault(d.molar, []).append(d)
    for group in by_molar.values():
        if len(group) > 1:
            volumes = {d.volume_ul for d in group}
            if len(volumes) > 1:
                for d in group:
                    d.ambiguous = True
                    d.note = "Same concentration, different volume - confirm grouping"
    return doses


@dataclass
class HillFit:
    success: bool = False
    ec50: float = float("nan")
    hill_slope: float = float("nan")
    bottom: float = float("nan")
    top: float = float("nan")
    message: str = ""


def hill_equation(conc, bottom, top, ec50, hill):
    """4-parameter Hill. The base is kept positive so curve_fit cannot wander
    into (negative) ** (fractional), which yields NaN and poisons the fit."""
    conc = np.asarray(conc, dtype=np.float64)
    ratio = np.maximum(ec50, 1e-30) / np.maximum(conc, 1e-30)
    return bottom + (top - bottom) / (1.0 + ratio ** hill)


def fit_hill(concentrations_m: Sequence[float], responses: Sequence[float]) -> HillFit:
    """4-parameter Hill fit. Reports failure rather than an unreliable number."""
    from scipy.optimize import curve_fit

    conc = np.asarray(concentrations_m, dtype=np.float64)
    resp = np.asarray(responses, dtype=np.float64)
    ok = np.isfinite(conc) & np.isfinite(resp) & (conc > 0)
    conc, resp = conc[ok], resp[ok]

    if len(conc) < 4:
        return HillFit(message="Insufficient data for EC50 (need >= 4 dose points)")

    p0 = (float(resp.min()), float(resp.max()), float(np.median(conc)), 1.0)
    try:
        popt, _ = curve_fit(hill_equation, conc, resp, p0=p0, maxfev=20000)
    except Exception as exc:
        return HillFit(message=f"Fit did not converge: {exc}")

    bottom, top, ec50, hill = (float(v) for v in popt)
    if not np.isfinite(ec50) or ec50 <= 0:
        return HillFit(message="Fit produced a non-physical EC50")

    return HillFit(success=True, ec50=ec50, hill_slope=hill,
                   bottom=bottom, top=top, message="OK")
