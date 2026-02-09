"""Background task queue for long-running operations."""

import asyncio
import uuid
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TaskInfo:
    id: str
    session_id: str
    task_type: str  # segmentation, quantification, tracking, export
    status: str = "pending"  # pending, running, complete, error, cancelled
    progress: float = 0.0
    current: int = 0
    total: int = 0
    stage: str = ""
    condition: str = ""
    image_set: str = ""
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at


class TaskQueue:
    """In-process background task queue using ThreadPoolExecutor."""

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, TaskInfo] = {}
        self._ws_callbacks: Dict[str, List[Callable]] = {}  # session_id -> callbacks

    def submit(
        self,
        session_id: str,
        task_type: str,
        fn: Callable,
        *args,
        **kwargs,
    ) -> str:
        """Submit a background task. Returns task_id."""
        task_id = uuid.uuid4().hex[:12]
        task = TaskInfo(id=task_id, session_id=session_id, task_type=task_type)
        self._tasks[task_id] = task

        def wrapper():
            task.status = "running"
            task.started_at = time.time()
            self._notify(session_id, task)
            try:
                result = fn(task, *args, **kwargs)
                task.status = "complete"
                task.result = result
                task.progress = 100.0
            except Exception as e:
                task.status = "error"
                task.error = traceback.format_exc()
                task.message = str(e)
            finally:
                task.completed_at = time.time()
                self._notify(session_id, task)

        self._executor.submit(wrapper)
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task and task.status == "running":
            task.status = "cancelled"
            task.completed_at = time.time()
            return True
        return False

    def update_progress(self, task: TaskInfo, **kwargs):
        """Update task progress and notify WebSocket listeners."""
        for k, v in kwargs.items():
            if hasattr(task, k):
                setattr(task, k, v)
        if task.total > 0:
            task.progress = 100.0 * task.current / task.total
        self._notify(task.session_id, task)

    def register_ws_callback(self, session_id: str, callback: Callable):
        if session_id not in self._ws_callbacks:
            self._ws_callbacks[session_id] = []
        self._ws_callbacks[session_id].append(callback)

    def unregister_ws_callback(self, session_id: str, callback: Callable):
        if session_id in self._ws_callbacks:
            self._ws_callbacks[session_id] = [
                cb for cb in self._ws_callbacks[session_id] if cb is not callback
            ]

    def _notify(self, session_id: str, task: TaskInfo):
        """Send task update to all WebSocket listeners for this session."""
        callbacks = self._ws_callbacks.get(session_id, [])
        msg = {
            "type": "progress" if task.status == "running" else "task_complete",
            "task_id": task.id,
            "task_type": task.task_type,
            "status": task.status,
            "progress": task.progress,
            "current": task.current,
            "total": task.total,
            "stage": task.stage,
            "condition": task.condition,
            "image_set": task.image_set,
            "message": task.message,
            "elapsed_seconds": task.elapsed,
        }
        if task.status == "complete" and task.result:
            msg["data"] = task.result
        if task.status == "error":
            msg["error"] = task.error

        for cb in callbacks:
            try:
                cb(msg)
            except Exception:
                pass
