"""Session lifecycle management."""

import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from cellquant.sessions.models import Session


class SessionManager:
    """Create, retrieve, and clean up user sessions."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or (Path.home() / ".cellquant" / "sessions")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, Session] = {}
        self._last_access: Dict[str, float] = {}

    def create_session(self) -> Session:
        session_id = uuid.uuid4().hex[:8]
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        session = Session(id=session_id, directory=session_dir)
        self._sessions[session_id] = session
        self._last_access[session_id] = time.time()
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        self._last_access[session_id] = time.time()
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        if session_id and session_id in self._sessions:
            self._last_access[session_id] = time.time()
            return self._sessions[session_id]
        return self.create_session()

    def delete_session(self, session_id: str):
        session = self._sessions.pop(session_id, None)
        self._last_access.pop(session_id, None)
        if session and session.directory.exists():
            shutil.rmtree(session.directory, ignore_errors=True)

    def cleanup_sync(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours. Returns count removed."""
        now = time.time()
        cutoff = max_age_hours * 3600
        stale = [
            sid for sid, t in self._last_access.items() if (now - t) > cutoff
        ]
        for sid in stale:
            self.delete_session(sid)

        # Also clean orphaned directories on disk
        count = len(stale)
        if self.base_dir.exists():
            for d in self.base_dir.iterdir():
                if d.is_dir() and d.name not in self._sessions:
                    # Check age by directory mtime
                    age = now - d.stat().st_mtime
                    if age > cutoff:
                        shutil.rmtree(d, ignore_errors=True)
                        count += 1
        return count

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)
