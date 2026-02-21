"""
Data models for the SenescenceDB knowledge base.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import uuid


@dataclass
class Paper:
    """A literature reference."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    authors: str = ""
    journal: str = ""
    year: int = 0
    doi: str = ""
    pmid: str = ""
    abstract: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def citation(self) -> str:
        parts = []
        if self.authors:
            parts.append(self.authors)
        if self.year:
            parts.append(f"({self.year})")
        if self.title:
            parts.append(self.title)
        if self.journal:
            parts.append(self.journal)
        return " ".join(parts) if parts else "(untitled)"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "doi": self.doi,
            "pmid": self.pmid,
            "abstract": self.abstract,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            title=data.get("title", ""),
            authors=data.get("authors", ""),
            journal=data.get("journal", ""),
            year=data.get("year", 0),
            doi=data.get("doi", ""),
            pmid=data.get("pmid", ""),
            abstract=data.get("abstract", ""),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )


@dataclass
class Protocol:
    """A lab protocol or method."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    category: str = ""  # e.g. "staining", "treatment", "imaging", "analysis"
    steps: str = ""  # Markdown-formatted steps
    reagents: str = ""
    tips: str = ""
    tags: List[str] = field(default_factory=list)
    linked_papers: List[str] = field(default_factory=list)  # Paper IDs
    date_created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "steps": self.steps,
            "reagents": self.reagents,
            "tips": self.tips,
            "tags": self.tags,
            "linked_papers": self.linked_papers,
            "date_created": self.date_created,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Protocol":
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            title=data.get("title", ""),
            category=data.get("category", ""),
            steps=data.get("steps", ""),
            reagents=data.get("reagents", ""),
            tips=data.get("tips", ""),
            tags=data.get("tags", []),
            linked_papers=data.get("linked_papers", []),
            date_created=data.get("date_created", ""),
        )


@dataclass
class Finding:
    """A key research finding or observation."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    summary: str = ""
    details: str = ""
    category: str = ""  # e.g. "mechanism", "biomarker", "pathway", "phenotype"
    confidence: str = "medium"  # low | medium | high | confirmed
    linked_papers: List[str] = field(default_factory=list)
    linked_experiments: List[str] = field(default_factory=list)  # OxyTrack experiment IDs
    tags: List[str] = field(default_factory=list)
    date_created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "details": self.details,
            "category": self.category,
            "confidence": self.confidence,
            "linked_papers": self.linked_papers,
            "linked_experiments": self.linked_experiments,
            "tags": self.tags,
            "date_created": self.date_created,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Finding":
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            details=data.get("details", ""),
            category=data.get("category", ""),
            confidence=data.get("confidence", "medium"),
            linked_papers=data.get("linked_papers", []),
            linked_experiments=data.get("linked_experiments", []),
            tags=data.get("tags", []),
            date_created=data.get("date_created", ""),
        )
