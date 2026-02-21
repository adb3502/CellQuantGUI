"""JSON-file backed storage for SenescenceDB knowledge base."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar

from cellquant.plugins.senescencedb.models import (
    Marker,
    Note,
    Paper,
    Pathway,
    Protocol,
)

T = TypeVar("T")

# Maps collection name -> dataclass type
_MODEL_MAP: Dict[str, Type] = {
    "papers": Paper,
    "markers": Marker,
    "pathways": Pathway,
    "protocols": Protocol,
    "notes": Note,
}


class SenescenceDBStore:
    """File-based storage with one JSON file per collection."""

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        # In-memory cache: collection -> {id: dict}
        self._cache: Dict[str, Dict[str, dict]] = {}
        for collection in _MODEL_MAP:
            self._cache[collection] = self._load_collection(collection)

    # -- low-level -----------------------------------------------------------

    def _collection_path(self, collection: str) -> Path:
        return self._dir / f"{collection}.json"

    def _load_collection(self, collection: str) -> Dict[str, dict]:
        path = self._collection_path(collection)
        if not path.exists():
            return {}
        with open(path) as f:
            items = json.load(f)
        return {item["id"]: item for item in items}

    def _persist(self, collection: str) -> None:
        path = self._collection_path(collection)
        items = list(self._cache[collection].values())
        with open(path, "w") as f:
            json.dump(items, f, indent=2, default=str)

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:12]

    # -- generic CRUD --------------------------------------------------------

    def _create(self, collection: str, data: dict) -> dict:
        entry_id = self._new_id()
        now = datetime.utcnow().isoformat()
        data["id"] = entry_id
        data["created_at"] = now
        if "updated_at" in _MODEL_MAP[collection].__dataclass_fields__:
            data["updated_at"] = now
        self._cache[collection][entry_id] = data
        self._persist(collection)
        return data

    def _get(self, collection: str, entry_id: str) -> Optional[dict]:
        return self._cache[collection].get(entry_id)

    def _list(self, collection: str) -> List[dict]:
        return list(self._cache[collection].values())

    def _update(self, collection: str, entry_id: str, updates: dict) -> Optional[dict]:
        existing = self._cache[collection].get(entry_id)
        if existing is None:
            return None
        for key, val in updates.items():
            if val is not None:
                existing[key] = val
        if "updated_at" in _MODEL_MAP[collection].__dataclass_fields__:
            existing["updated_at"] = datetime.utcnow().isoformat()
        self._cache[collection][entry_id] = existing
        self._persist(collection)
        return existing

    def _delete(self, collection: str, entry_id: str) -> bool:
        if entry_id not in self._cache[collection]:
            return False
        del self._cache[collection][entry_id]
        self._persist(collection)
        return True

    # -- Papers --------------------------------------------------------------

    def create_paper(self, data: dict) -> dict:
        return self._create("papers", data)

    def get_paper(self, paper_id: str) -> Optional[dict]:
        return self._get("papers", paper_id)

    def list_papers(self) -> List[dict]:
        return self._list("papers")

    def update_paper(self, paper_id: str, updates: dict) -> Optional[dict]:
        return self._update("papers", paper_id, updates)

    def delete_paper(self, paper_id: str) -> bool:
        return self._delete("papers", paper_id)

    # -- Markers -------------------------------------------------------------

    def create_marker(self, data: dict) -> dict:
        return self._create("markers", data)

    def get_marker(self, marker_id: str) -> Optional[dict]:
        return self._get("markers", marker_id)

    def list_markers(self) -> List[dict]:
        return self._list("markers")

    def update_marker(self, marker_id: str, updates: dict) -> Optional[dict]:
        return self._update("markers", marker_id, updates)

    def delete_marker(self, marker_id: str) -> bool:
        return self._delete("markers", marker_id)

    # -- Pathways ------------------------------------------------------------

    def create_pathway(self, data: dict) -> dict:
        return self._create("pathways", data)

    def get_pathway(self, pathway_id: str) -> Optional[dict]:
        return self._get("pathways", pathway_id)

    def list_pathways(self) -> List[dict]:
        return self._list("pathways")

    def update_pathway(self, pathway_id: str, updates: dict) -> Optional[dict]:
        return self._update("pathways", pathway_id, updates)

    def delete_pathway(self, pathway_id: str) -> bool:
        return self._delete("pathways", pathway_id)

    # -- Protocols -----------------------------------------------------------

    def create_protocol(self, data: dict) -> dict:
        return self._create("protocols", data)

    def get_protocol(self, protocol_id: str) -> Optional[dict]:
        return self._get("protocols", protocol_id)

    def list_protocols(self) -> List[dict]:
        return self._list("protocols")

    def update_protocol(self, protocol_id: str, updates: dict) -> Optional[dict]:
        return self._update("protocols", protocol_id, updates)

    def delete_protocol(self, protocol_id: str) -> bool:
        return self._delete("protocols", protocol_id)

    # -- Notes ---------------------------------------------------------------

    def create_note(self, data: dict) -> dict:
        return self._create("notes", data)

    def get_note(self, note_id: str) -> Optional[dict]:
        return self._get("notes", note_id)

    def list_notes(self) -> List[dict]:
        return self._list("notes")

    def update_note(self, note_id: str, updates: dict) -> Optional[dict]:
        return self._update("notes", note_id, updates)

    def delete_note(self, note_id: str) -> bool:
        return self._delete("notes", note_id)

    # -- Search --------------------------------------------------------------

    def search(
        self,
        query: str,
        entry_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[dict]:
        """Simple text search across all collections.

        Returns list of dicts with ``entry_type``, ``id``, ``title``,
        ``snippet``, ``tags``, ``score`` keys.
        """
        query_lower = query.lower()
        results: List[dict] = []

        collections = entry_types if entry_types else list(_MODEL_MAP.keys())
        # Normalise singular forms ("paper" -> "papers")
        collections = [c if c.endswith("s") else c + "s" for c in collections]

        for collection in collections:
            if collection not in self._cache:
                continue
            for entry in self._cache[collection].values():
                score = self._match_score(entry, query_lower, tags)
                if score > 0:
                    title = (
                        entry.get("title")
                        or entry.get("name")
                        or entry.get("id", "")
                    )
                    snippet = self._extract_snippet(entry, query_lower)
                    results.append({
                        "entry_type": collection.rstrip("s"),
                        "id": entry["id"],
                        "title": title,
                        "snippet": snippet,
                        "tags": entry.get("tags", []),
                        "score": score,
                    })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    @staticmethod
    def _match_score(entry: dict, query_lower: str, tags: Optional[List[str]]) -> float:
        """Compute a simple relevance score for an entry."""
        score = 0.0
        searchable = json.dumps(entry, default=str).lower()

        if query_lower in searchable:
            score += 1.0

        # Boost for title/name match
        title = (entry.get("title") or entry.get("name") or "").lower()
        if query_lower in title:
            score += 2.0

        # Tag filter
        if tags:
            entry_tags = {t.lower() for t in entry.get("tags", [])}
            if not entry_tags.intersection(t.lower() for t in tags):
                return 0.0
            score += 0.5

        return score

    @staticmethod
    def _extract_snippet(entry: dict, query_lower: str) -> str:
        """Pull a short text snippet around the first match."""
        for field_name in ("abstract", "description", "body", "notes", "objective"):
            text = entry.get(field_name, "")
            if not text:
                continue
            idx = text.lower().find(query_lower)
            if idx >= 0:
                start = max(0, idx - 60)
                end = min(len(text), idx + len(query_lower) + 60)
                return ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
        return ""

    # -- Stats ---------------------------------------------------------------

    def stats(self) -> dict:
        all_tags: set[str] = set()
        for collection in self._cache.values():
            for entry in collection.values():
                all_tags.update(entry.get("tags", []))
        return {
            "n_papers": len(self._cache["papers"]),
            "n_markers": len(self._cache["markers"]),
            "n_pathways": len(self._cache["pathways"]),
            "n_protocols": len(self._cache["protocols"]),
            "n_notes": len(self._cache["notes"]),
            "all_tags": sorted(all_tags),
        }
