"""FastAPI router for SenescenceDB endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from cellquant.plugins.senescencedb.schemas import (
    KBStatsResponse,
    MarkerCreate,
    MarkerResponse,
    MarkerUpdate,
    NoteCreate,
    NoteResponse,
    NoteUpdate,
    PaperCreate,
    PaperResponse,
    PaperUpdate,
    PathwayCreate,
    PathwayResponse,
    PathwayUpdate,
    ProtocolCreate,
    ProtocolResponse,
    ProtocolUpdate,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from cellquant.plugins.senescencedb.store import SenescenceDBStore

_store: SenescenceDBStore | None = None


def set_store(store: SenescenceDBStore) -> None:
    global _store
    _store = store


def _get_store() -> SenescenceDBStore:
    if _store is None:
        raise RuntimeError("SenescenceDB store not initialised")
    return _store


def build_router() -> APIRouter:
    router = APIRouter()

    # ── Papers ──────────────────────────────────────────────────────────

    @router.get("/papers", response_model=List[PaperResponse])
    async def list_papers():
        return _get_store().list_papers()

    @router.post("/papers", response_model=PaperResponse, status_code=201)
    async def create_paper(req: PaperCreate):
        return _get_store().create_paper(req.model_dump())

    @router.get("/papers/{paper_id}", response_model=PaperResponse)
    async def get_paper(paper_id: str):
        p = _get_store().get_paper(paper_id)
        if p is None:
            raise HTTPException(404, "Paper not found")
        return p

    @router.patch("/papers/{paper_id}", response_model=PaperResponse)
    async def update_paper(paper_id: str, req: PaperUpdate):
        p = _get_store().update_paper(paper_id, req.model_dump(exclude_unset=True))
        if p is None:
            raise HTTPException(404, "Paper not found")
        return p

    @router.delete("/papers/{paper_id}")
    async def delete_paper(paper_id: str):
        if not _get_store().delete_paper(paper_id):
            raise HTTPException(404, "Paper not found")
        return {"status": "deleted"}

    # ── Markers ─────────────────────────────────────────────────────────

    @router.get("/markers", response_model=List[MarkerResponse])
    async def list_markers():
        return _get_store().list_markers()

    @router.post("/markers", response_model=MarkerResponse, status_code=201)
    async def create_marker(req: MarkerCreate):
        return _get_store().create_marker(req.model_dump())

    @router.get("/markers/{marker_id}", response_model=MarkerResponse)
    async def get_marker(marker_id: str):
        m = _get_store().get_marker(marker_id)
        if m is None:
            raise HTTPException(404, "Marker not found")
        return m

    @router.patch("/markers/{marker_id}", response_model=MarkerResponse)
    async def update_marker(marker_id: str, req: MarkerUpdate):
        m = _get_store().update_marker(marker_id, req.model_dump(exclude_unset=True))
        if m is None:
            raise HTTPException(404, "Marker not found")
        return m

    @router.delete("/markers/{marker_id}")
    async def delete_marker(marker_id: str):
        if not _get_store().delete_marker(marker_id):
            raise HTTPException(404, "Marker not found")
        return {"status": "deleted"}

    # ── Pathways ────────────────────────────────────────────────────────

    @router.get("/pathways", response_model=List[PathwayResponse])
    async def list_pathways():
        return _get_store().list_pathways()

    @router.post("/pathways", response_model=PathwayResponse, status_code=201)
    async def create_pathway(req: PathwayCreate):
        return _get_store().create_pathway(req.model_dump())

    @router.get("/pathways/{pathway_id}", response_model=PathwayResponse)
    async def get_pathway(pathway_id: str):
        p = _get_store().get_pathway(pathway_id)
        if p is None:
            raise HTTPException(404, "Pathway not found")
        return p

    @router.patch("/pathways/{pathway_id}", response_model=PathwayResponse)
    async def update_pathway(pathway_id: str, req: PathwayUpdate):
        p = _get_store().update_pathway(pathway_id, req.model_dump(exclude_unset=True))
        if p is None:
            raise HTTPException(404, "Pathway not found")
        return p

    @router.delete("/pathways/{pathway_id}")
    async def delete_pathway(pathway_id: str):
        if not _get_store().delete_pathway(pathway_id):
            raise HTTPException(404, "Pathway not found")
        return {"status": "deleted"}

    # ── Protocols ───────────────────────────────────────────────────────

    @router.get("/protocols", response_model=List[ProtocolResponse])
    async def list_protocols():
        return _get_store().list_protocols()

    @router.post("/protocols", response_model=ProtocolResponse, status_code=201)
    async def create_protocol(req: ProtocolCreate):
        return _get_store().create_protocol(req.model_dump())

    @router.get("/protocols/{protocol_id}", response_model=ProtocolResponse)
    async def get_protocol(protocol_id: str):
        p = _get_store().get_protocol(protocol_id)
        if p is None:
            raise HTTPException(404, "Protocol not found")
        return p

    @router.patch("/protocols/{protocol_id}", response_model=ProtocolResponse)
    async def update_protocol(protocol_id: str, req: ProtocolUpdate):
        p = _get_store().update_protocol(protocol_id, req.model_dump(exclude_unset=True))
        if p is None:
            raise HTTPException(404, "Protocol not found")
        return p

    @router.delete("/protocols/{protocol_id}")
    async def delete_protocol(protocol_id: str):
        if not _get_store().delete_protocol(protocol_id):
            raise HTTPException(404, "Protocol not found")
        return {"status": "deleted"}

    # ── Notes ───────────────────────────────────────────────────────────

    @router.get("/notes", response_model=List[NoteResponse])
    async def list_notes():
        return _get_store().list_notes()

    @router.post("/notes", response_model=NoteResponse, status_code=201)
    async def create_note(req: NoteCreate):
        return _get_store().create_note(req.model_dump())

    @router.get("/notes/{note_id}", response_model=NoteResponse)
    async def get_note(note_id: str):
        n = _get_store().get_note(note_id)
        if n is None:
            raise HTTPException(404, "Note not found")
        return n

    @router.patch("/notes/{note_id}", response_model=NoteResponse)
    async def update_note(note_id: str, req: NoteUpdate):
        n = _get_store().update_note(note_id, req.model_dump(exclude_unset=True))
        if n is None:
            raise HTTPException(404, "Note not found")
        return n

    @router.delete("/notes/{note_id}")
    async def delete_note(note_id: str):
        if not _get_store().delete_note(note_id):
            raise HTTPException(404, "Note not found")
        return {"status": "deleted"}

    # ── Search & Stats ──────────────────────────────────────────────────

    @router.post("/search", response_model=SearchResponse)
    async def search_kb(req: SearchRequest):
        store = _get_store()
        hits = store.search(
            query=req.query,
            entry_types=req.entry_types or None,
            tags=req.tags or None,
        )
        return SearchResponse(
            query=req.query,
            total_hits=len(hits),
            hits=[SearchHit(**h) for h in hits],
        )

    @router.get("/stats", response_model=KBStatsResponse)
    async def kb_stats():
        return _get_store().stats()

    return router
