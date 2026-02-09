"""FastAPI dependency injection."""

from fastapi import Request, HTTPException

from cellquant.sessions.manager import SessionManager
from cellquant.sessions.models import Session
from cellquant.tasks.queue import TaskQueue

# Singletons - initialized by app factory
_session_manager: SessionManager | None = None
_task_queue: TaskQueue | None = None


def init_dependencies():
    global _session_manager, _task_queue
    _session_manager = SessionManager()
    _task_queue = TaskQueue(max_workers=2)


def get_session_manager() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("Dependencies not initialized")
    return _session_manager


def get_task_queue() -> TaskQueue:
    if _task_queue is None:
        raise RuntimeError("Dependencies not initialized")
    return _task_queue


def get_session(session_id: str) -> Session:
    manager = get_session_manager()
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session
