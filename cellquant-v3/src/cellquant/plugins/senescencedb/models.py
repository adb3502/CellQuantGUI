"""SenescenceDB domain models – papers, markers, pathways, protocols, notes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class EntryType(str, Enum):
    PAPER = "paper"
    MARKER = "marker"
    PATHWAY = "pathway"
    PROTOCOL = "protocol"
    NOTE = "note"


class SenescenceType(str, Enum):
    REPLICATIVE = "replicative"
    ONCOGENE_INDUCED = "oncogene-induced"
    STRESS_INDUCED = "stress-induced"
    THERAPY_INDUCED = "therapy-induced"
    DEVELOPMENTAL = "developmental"
    PARACRINE = "paracrine"
    OTHER = "other"


# -- Papers ------------------------------------------------------------------

@dataclass
class Paper:
    """A reference to a published paper relevant to senescence research."""

    id: str
    title: str
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    year: int = 0
    doi: str = ""
    pmid: str = ""
    abstract: str = ""
    key_findings: List[str] = field(default_factory=list)
    senescence_types: List[str] = field(default_factory=list)
    markers_mentioned: List[str] = field(default_factory=list)
    oxysterols_mentioned: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# -- Markers -----------------------------------------------------------------

@dataclass
class Marker:
    """A senescence-associated marker with context and references."""

    id: str
    name: str
    gene_symbol: str = ""
    marker_type: str = ""  # e.g. "surface", "secreted", "nuclear", "enzymatic"
    description: str = ""
    detection_methods: List[str] = field(default_factory=list)
    senescence_types: List[str] = field(default_factory=list)
    upregulated: bool = True  # vs downregulated in senescence
    caveats: str = ""
    paper_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# -- Pathways ----------------------------------------------------------------

@dataclass
class Pathway:
    """A signalling or metabolic pathway relevant to senescence."""

    id: str
    name: str
    description: str = ""
    key_genes: List[str] = field(default_factory=list)
    senescence_role: str = ""  # pro-senescence, anti-senescence, context-dependent
    oxysterol_link: str = ""
    paper_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# -- Protocols ---------------------------------------------------------------

@dataclass
class Protocol:
    """A lab protocol relevant to senescence experiments."""

    id: str
    title: str
    objective: str = ""
    cell_lines: List[str] = field(default_factory=list)
    reagents: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)
    paper_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# -- Notes -------------------------------------------------------------------

@dataclass
class Note:
    """Free-form research note with cross-references."""

    id: str
    title: str
    body: str = ""
    linked_paper_ids: List[str] = field(default_factory=list)
    linked_marker_ids: List[str] = field(default_factory=list)
    linked_pathway_ids: List[str] = field(default_factory=list)
    linked_protocol_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
