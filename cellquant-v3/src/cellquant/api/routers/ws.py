"""WebSocket endpoint for real-time progress updates."""

import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from cellquant.api.dependencies import get_task_queue

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{session_id}")
async def websocket_progress(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for receiving task progress updates."""
    await websocket.accept()

    queue = get_task_queue()
    message_queue: asyncio.Queue = asyncio.Queue()

    def on_message(msg: dict):
        """Callback from task queue - put message in async queue."""
        try:
            message_queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    queue.register_ws_callback(session_id, on_message)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(message_queue.get(), timeout=30.0)
                await websocket.send_text(json.dumps(msg))
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        pass
    finally:
        queue.unregister_ws_callback(session_id, on_message)
