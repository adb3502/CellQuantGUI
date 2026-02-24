"""Background task queue for long-running operations."""

import asyncio
import io
import logging
import sys
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
    logs: List[str] = field(default_factory=list)
    progress_data: Optional[dict] = None  # Transient data sent with progress msgs

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at


class LogCapture:
    """Context manager that captures stdout/stderr and Python logging to a task.

    Cellpose and other libraries use the logging module rather than print,
    so we attach a temporary handler to the root logger to catch those too.
    """

    _LOGGER_NAMES = ["cellpose"]

    def __init__(self, task: TaskInfo, notify: Callable):
        self._task = task
        self._notify = notify
        self._old_stdout: Any = None
        self._old_stderr: Any = None
        self._log_handlers: List[tuple] = []  # (logger, handler)

    def __enter__(self):
        # Redirect stdout/stderr
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = _LineWriter(self._task, self._notify, self._old_stdout)
        sys.stderr = _LineWriter(self._task, self._notify, self._old_stderr)

        # Add logging handler to capture library log messages
        handler = _TaskLogHandler(self._task, self._notify)
        handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        handler.setLevel(logging.DEBUG)
        for name in self._LOGGER_NAMES:
            logger = logging.getLogger(name)
            # Save original level so we can restore it
            self._log_handlers.append((logger, handler, logger.level))
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)

        return self

    def __exit__(self, *exc):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        for logger, handler, original_level in self._log_handlers:
            logger.removeHandler(handler)
            logger.setLevel(original_level)
        self._log_handlers.clear()
        return False


class _LineWriter(io.TextIOBase):
    """Write-only stream that appends each line to task.logs and notifies."""

    def __init__(self, task: TaskInfo, notify: Callable, passthrough):
        self._task = task
        self._notify_fn = notify
        self._passthrough = passthrough
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        # Also write to original stream so terminal still shows output
        if self._passthrough:
            try:
                self._passthrough.write(s)
                self._passthrough.flush()
            except Exception:
                pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._task.logs.append(line)
            self._notify_fn(self._task.session_id, self._task)
        return len(s)

    def flush(self):
        # Flush any partial line
        if self._buf:
            self._task.logs.append(self._buf)
            self._buf = ""
            self._notify_fn(self._task.session_id, self._task)
        if self._passthrough:
            try:
                self._passthrough.flush()
            except Exception:
                pass

    @property
    def encoding(self):
        return getattr(self._passthrough, "encoding", "utf-8")


class _TaskLogHandler(logging.Handler):
    """Logging handler that appends formatted records to task.logs."""

    def __init__(self, task: TaskInfo, notify: Callable):
        super().__init__()
        self._task = task
        self._notify_fn = notify

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self._task.logs.append(msg)
            self._notify_fn(self._task.session_id, self._task)
        except Exception:
            pass


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
                with LogCapture(task, self._notify):
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
            "logs": list(task.logs),
        }
        if task.progress_data:
            msg["data"] = task.progress_data
            task.progress_data = None  # Consume after sending
        if task.status == "complete" and task.result:
            msg["data"] = task.result
        if task.status == "error":
            msg["error"] = task.error

        for cb in callbacks:
            try:
                cb(msg)
            except Exception:
                pass
