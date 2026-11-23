"""A task that fails before any WebSocket client connects must still be
discoverable over HTTP.

Real incident: 183 torch files were corrupted by a transient disk fault, so
the Cellpose import raised immediately. The task errored before the frontend
had registered its WebSocket callback, _notify delivered the terminal event
to an empty callback list, and the UI sat at "Submitting... 0%" forever with
no way to learn what went wrong.
"""
import time

import pytest

from cellquant.tasks.queue import TaskQueue


def _wait(queue, task_id, timeout=5.0):
    """Wait for a task to be fully settled.

    Waiting on ``status`` alone races the worker: the except block assigns
    status first and error/message after, so a reader can observe
    status == "error" with error still None. completed_at is set last, in
    the finally block, so it is the only safe completion signal.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = queue.get_task(task_id)
        if task and task.completed_at is not None:
            return task
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not settle within {timeout}s")


class TestFailureRecordedWithoutListener:

    def test_error_status_recorded_with_no_ws_callback(self):
        """No client is connected: the failure must still be recorded."""
        queue = TaskQueue(max_workers=1)

        def boom(task):
            raise RuntimeError("source code string cannot contain null bytes")

        task_id = queue.submit("sess-none", "segmentation", boom)
        task = _wait(queue, task_id)
        assert task.status == "error"

    def test_traceback_is_retained(self):
        queue = TaskQueue(max_workers=1)

        def boom(task):
            raise RuntimeError("null bytes in torch/version.py")

        task_id = queue.submit("sess-tb", "segmentation", boom)
        task = _wait(queue, task_id)
        assert task.error, "traceback must be retained for the user"
        assert "null bytes" in task.error

    def test_message_carries_the_reason(self):
        queue = TaskQueue(max_workers=1)

        def boom(task):
            raise ImportError("cannot import torch")

        task_id = queue.submit("sess-msg", "segmentation", boom)
        task = _wait(queue, task_id)
        assert "torch" in task.message

    def test_failure_is_not_left_running(self):
        """The precise symptom: status stuck at 'running' forever."""
        queue = TaskQueue(max_workers=1)

        def boom(task):
            raise RuntimeError("boom")

        task_id = queue.submit("sess-stuck", "segmentation", boom)
        task = _wait(queue, task_id)
        assert task.status != "running"
        assert task.completed_at is not None


class TestStatusEndpointExposesError:
    """The HTTP status endpoint is the only way to learn why a task failed
    when the WebSocket event was missed. It must carry the reason."""

    def test_schema_has_an_error_field(self):
        from cellquant.api.schemas.segmentation import SegmentationStatusResponse
        assert "error" in SegmentationStatusResponse.model_fields, (
            "status response must expose the traceback, or a client that "
            "missed the websocket event can never learn why the task failed"
        )

    def test_status_response_carries_error_text(self):
        from cellquant.api.schemas.segmentation import SegmentationStatusResponse
        r = SegmentationStatusResponse(
            task_id="abc123def456", status="error", progress=0.0,
            message="cannot import torch", error="Traceback...null bytes",
        )
        assert r.error and "null bytes" in r.error


class TestTerminalStateIsAtomic:
    """status must not become terminal before the reason is attached.

    A client polling on status alone would otherwise render an error with
    a blank reason - which is what "Submitting... 0%" with no message is.
    """

    def test_error_and_message_land_with_the_status(self):
        queue = TaskQueue(max_workers=1)
        observed = []

        def boom(task):
            raise RuntimeError("null bytes in torch")

        task_id = queue.submit("sess-atomic", "segmentation", boom)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            t = queue.get_task(task_id)
            if t and t.status == "error":
                observed.append((t.error, t.message))
                break
            time.sleep(0.001)

        assert observed, "never observed the error status"
        err, msg = observed[0]
        assert err is not None, (
            "status flipped to 'error' before the traceback was attached"
        )
        assert msg, "status flipped to 'error' before the message was attached"
