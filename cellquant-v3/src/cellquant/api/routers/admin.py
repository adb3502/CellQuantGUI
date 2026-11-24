"""Admin endpoints: user management + activity log."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from cellquant.auth import activity as act
from cellquant.auth.database import (
    count_activity,
    delete_user,
    list_activity,
    list_users,
    update_user,
)
from cellquant.auth.middleware import CurrentUser, require_admin

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Schemas ────────────────────────────────────────────────────────


class UserRow(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: str
    last_login: Optional[str]


class PatchUserRequest(BaseModel):
    role: Optional[str] = None
    is_active: Optional[bool] = None


class ActivityRow(BaseModel):
    id: int
    user_id: int
    username: str
    action: str
    resource: str
    detail_json: str
    ip_address: str
    timestamp: str


class ActivityPage(BaseModel):
    total: int
    page: int
    per_page: int
    data: List[ActivityRow]


# ── Routes ─────────────────────────────────────────────────────────


@router.get("/users", response_model=List[UserRow])
async def get_users(admin: CurrentUser = Depends(require_admin)):
    users = await list_users()
    return [
        UserRow(
            id=u.id,
            username=u.username,
            email=u.email,
            role=u.role,
            is_active=u.is_active,
            created_at=u.created_at.isoformat(),
            last_login=u.last_login.isoformat() if u.last_login else None,
        )
        for u in users
    ]


@router.patch("/users/{user_id}", response_model=UserRow)
async def patch_user(
    user_id: int,
    body: PatchUserRequest,
    request: Request,
    admin: CurrentUser = Depends(require_admin),
):
    from cellquant.auth.database import get_user_by_id

    user = await get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    updates = {}
    if body.role is not None:
        updates["role"] = body.role
    if body.is_active is not None:
        updates["is_active"] = int(body.is_active)

    if updates:
        await update_user(user_id, **updates)
        await act.log(
            admin,
            "update_user",
            resource=f"user:{user_id}",
            detail=updates,
            request=request,
        )

    updated = await get_user_by_id(user_id)
    return UserRow(
        id=updated.id,
        username=updated.username,
        email=updated.email,
        role=updated.role,
        is_active=updated.is_active,
        created_at=updated.created_at.isoformat(),
        last_login=updated.last_login.isoformat() if updated.last_login else None,
    )


@router.delete("/users/{user_id}", status_code=204)
async def remove_user(
    user_id: int,
    request: Request,
    admin: CurrentUser = Depends(require_admin),
):
    from cellquant.auth.database import get_user_by_id

    user = await get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    await delete_user(user_id)
    await act.log(
        admin,
        "delete_user",
        resource=f"user:{user_id}",
        detail={"deleted_username": user.username},
        request=request,
    )


@router.get("/activity", response_model=ActivityPage)
async def get_activity(
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    page: int = 1,
    per_page: int = 50,
    admin: CurrentUser = Depends(require_admin),
):
    offset = (page - 1) * per_page
    rows = await list_activity(user_id=user_id, action=action, limit=per_page, offset=offset)
    total = await count_activity(user_id=user_id, action=action)
    return ActivityPage(
        total=total,
        page=page,
        per_page=per_page,
        data=[
            ActivityRow(
                id=r.id,
                user_id=r.user_id,
                username=r.username,
                action=r.action,
                resource=r.resource,
                detail_json=r.detail_json,
                ip_address=r.ip_address,
                timestamp=r.timestamp.isoformat(),
            )
            for r in rows
        ],
    )
