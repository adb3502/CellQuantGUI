"""Activity logging helper — convenience wrapper around database.log_activity."""

from __future__ import annotations

from fastapi import Request

from cellquant.auth.database import log_activity
from cellquant.auth.middleware import CurrentUser


async def log(
    user: CurrentUser,
    action: str,
    resource: str = "",
    detail: dict | None = None,
    request: Request | None = None,
) -> None:
    """Log a user action. Call from router endpoints after performing the action."""
    ip = ""
    if request is not None:
        forwarded = request.headers.get("X-Forwarded-For")
        ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "")
    await log_activity(
        user_id=user.id,
        username=user.username,
        action=action,
        resource=resource,
        detail=detail or {},
        ip_address=ip,
    )
