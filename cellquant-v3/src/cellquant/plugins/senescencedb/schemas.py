"""Pydantic schemas for SenescenceDB API endpoints."""

from __future__ import annotations

from pydantic import BaseModel
from typing import List, Optional


# -- Papers ------------------------------------------------------------------

class PaperCreate(BaseModel):
    title: str
    authors: List[str] = []
    journal: str = ""
    year: int = 0
    doi: str = ""
    pmid: str = ""
    abstract: str = ""
    key_findings: List[str] = []
    senescence_types: List[str] = []
    markers_mentioned: List[str] = []
    oxysterols_mentioned: List[str] = []
    tags: List[str] = []
    notes: str = ""


class PaperUpdate(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    abstract: Optional[str] = None
    key_findings: Optional[List[str]] = None
    senescence_types: Optional[List[str]] = None
    markers_mentioned: Optional[List[str]] = None
    oxysterols_mentioned: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class PaperResponse(PaperCreate):
    id: str
    created_at: str


# -- Markers -----------------------------------------------------------------

class MarkerCreate(BaseModel):
    name: str
    gene_symbol: str = ""
    marker_type: str = ""
    description: str = ""
    detection_methods: List[str] = []
    senescence_types: List[str] = []
    upregulated: bool = True
    caveats: str = ""
    paper_ids: List[str] = []
    tags: List[str] = []


class MarkerUpdate(BaseModel):
    name: Optional[str] = None
    gene_symbol: Optional[str] = None
    marker_type: Optional[str] = None
    description: Optional[str] = None
    detection_methods: Optional[List[str]] = None
    senescence_types: Optional[List[str]] = None
    upregulated: Optional[bool] = None
    caveats: Optional[str] = None
    paper_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class MarkerResponse(MarkerCreate):
    id: str
    created_at: str


# -- Pathways ----------------------------------------------------------------

class PathwayCreate(BaseModel):
    name: str
    description: str = ""
    key_genes: List[str] = []
    senescence_role: str = ""
    oxysterol_link: str = ""
    paper_ids: List[str] = []
    tags: List[str] = []


class PathwayUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    key_genes: Optional[List[str]] = None
    senescence_role: Optional[str] = None
    oxysterol_link: Optional[str] = None
    paper_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class PathwayResponse(PathwayCreate):
    id: str
    created_at: str


# -- Protocols ---------------------------------------------------------------

class ProtocolCreate(BaseModel):
    title: str
    objective: str = ""
    cell_lines: List[str] = []
    reagents: List[str] = []
    steps: List[str] = []
    tips: List[str] = []
    paper_ids: List[str] = []
    tags: List[str] = []


class ProtocolUpdate(BaseModel):
    title: Optional[str] = None
    objective: Optional[str] = None
    cell_lines: Optional[List[str]] = None
    reagents: Optional[List[str]] = None
    steps: Optional[List[str]] = None
    tips: Optional[List[str]] = None
    paper_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class ProtocolResponse(ProtocolCreate):
    id: str
    created_at: str
    updated_at: str


# -- Notes -------------------------------------------------------------------

class NoteCreate(BaseModel):
    title: str
    body: str = ""
    linked_paper_ids: List[str] = []
    linked_marker_ids: List[str] = []
    linked_pathway_ids: List[str] = []
    linked_protocol_ids: List[str] = []
    tags: List[str] = []


class NoteUpdate(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    linked_paper_ids: Optional[List[str]] = None
    linked_marker_ids: Optional[List[str]] = None
    linked_pathway_ids: Optional[List[str]] = None
    linked_protocol_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class NoteResponse(NoteCreate):
    id: str
    created_at: str
    updated_at: str


# -- Search / cross-reference ------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    entry_types: List[str] = []  # filter to specific types
    tags: List[str] = []


class SearchHit(BaseModel):
    entry_type: str
    id: str
    title: str
    snippet: str = ""
    tags: List[str] = []
    score: float = 0.0


class SearchResponse(BaseModel):
    query: str
    total_hits: int
    hits: List[SearchHit]


class KBStatsResponse(BaseModel):
    n_papers: int
    n_markers: int
    n_pathways: int
    n_protocols: int
    n_notes: int
    all_tags: List[str]
