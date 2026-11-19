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

_DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mM|uM|µM|nM|pM|M|mm|um|µm|nm|pm|m)\b",
                      re.IGNORECASE)
_VOLUME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ul|µl|uL)\b", re.IGNORECASE)


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
    conc = np.asarray(conc, dtype=np.float64)
    return bottom + (top - bottom) / (1.0 + (ec50 / np.maximum(conc, 1e-30)) ** hill)


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
