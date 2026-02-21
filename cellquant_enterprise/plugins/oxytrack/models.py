"""
Data models for OxyTrack experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
import uuid


@dataclass
class OxysterolTreatment:
    """A single oxysterol treatment applied to cells."""
    compound: str  # e.g. "7-ketocholesterol", "25-hydroxycholesterol"
    concentration_um: float  # micromolar
    duration_hours: float
    vehicle: str = "ethanol"
    vehicle_pct: float = 0.1

    def label(self) -> str:
        return f"{self.compound} {self.concentration_um}\u00b5M {self.duration_hours}h"


@dataclass
class SenescenceMarker:
    """A senescence marker measured in the experiment."""
    name: str  # e.g. "SA-\u03b2-gal", "p21", "p16", "\u03b3H2AX"
    channel_suffix: str  # maps to CellQuant channel, e.g. "C2"
    marker_type: str = "fluorescence"  # fluorescence | staining | morphology


@dataclass
class Experiment:
    """A complete oxysterol-senescence experiment."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    description: str = ""
    cell_line: str = ""
    passage: str = ""
    date_created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    date_modified: str = ""

    treatments: List[OxysterolTreatment] = field(default_factory=list)
    markers: List[SenescenceMarker] = field(default_factory=list)

    # Folder that was loaded into CellQuant for this experiment
    data_folder: str = ""
    # CellQuant results CSV path (after quantification)
    results_path: str = ""

    notes: str = ""
    tags: List[str] = field(default_factory=list)

    # Summary stats populated after quantification
    n_conditions: int = 0
    n_cells: int = 0
    status: str = "draft"  # draft | in_progress | analysed | archived

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "cell_line": self.cell_line,
            "passage": self.passage,
            "date_created": self.date_created,
            "date_modified": self.date_modified,
            "treatments": [
                {
                    "compound": t.compound,
                    "concentration_um": t.concentration_um,
                    "duration_hours": t.duration_hours,
                    "vehicle": t.vehicle,
                    "vehicle_pct": t.vehicle_pct,
                }
                for t in self.treatments
            ],
            "markers": [
                {
                    "name": m.name,
                    "channel_suffix": m.channel_suffix,
                    "marker_type": m.marker_type,
                }
                for m in self.markers
            ],
            "data_folder": self.data_folder,
            "results_path": self.results_path,
            "notes": self.notes,
            "tags": self.tags,
            "n_conditions": self.n_conditions,
            "n_cells": self.n_cells,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Experiment":
        treatments = [
            OxysterolTreatment(**t) for t in data.get("treatments", [])
        ]
        markers = [
            SenescenceMarker(**m) for m in data.get("markers", [])
        ]
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            name=data.get("name", ""),
            description=data.get("description", ""),
            cell_line=data.get("cell_line", ""),
            passage=data.get("passage", ""),
            date_created=data.get("date_created", ""),
            date_modified=data.get("date_modified", ""),
            treatments=treatments,
            markers=markers,
            data_folder=data.get("data_folder", ""),
            results_path=data.get("results_path", ""),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
            n_conditions=data.get("n_conditions", 0),
            n_cells=data.get("n_cells", 0),
            status=data.get("status", "draft"),
        )
