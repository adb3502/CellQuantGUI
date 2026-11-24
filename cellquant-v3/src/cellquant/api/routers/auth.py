"""Authentication endpoints: login, register (admin only), me, logout."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr

from cellquant.auth.auth import create_access_token, hash_password, verify_password
from cellquant.auth.database import (
    create_user,
    get_user_by_username,
    update_user,
)
from cellquant.auth.middleware import CurrentUser, get_current_user, require_admin
from cellquant.auth import activity as act

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Schemas ────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "user"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool


# ── Routes ─────────────────────────────────────────────────────────


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request, response: Response):
    user = await get_user_by_username(body.username)
    if user is None or not user.is_active or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = create_access_token(user.id, user.username, user.role)

    # Update last_login
    await update_user(user.id, last_login=datetime.now(timezone.utc).isoformat())

    # Set httpOnly cookie for browser clients (SvelteKit SSR)
    response.set_cookie(
        key="cellquant_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=8 * 3600,
    )

    ip = request.headers.get("X-Forwarded-For", "")
    if not ip and request.client:
        ip = request.client.host
    from cellquant.auth.database import log_activity
    await log_activity(user.id, user.username, "login", ip_address=ip)

    return TokenResponse(access_token=token, username=user.username, role=user.role)


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    current_user: CurrentUser = Depends(get_current_user),
):
    response.delete_cookie("cellquant_token")
    await act.log(current_user, "logout", request=request)
    return {"detail": "Logged out"}


@router.get("/me", response_model=UserResponse)
async def me(current_user: CurrentUser = Depends(get_current_user)):
    from cellquant.auth.database import get_user_by_id
    user = await get_user_by_id(current_user.id)
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
    )


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    body: RegisterRequest,
    request: Request,
    admin: CurrentUser = Depends(require_admin),
):
    existing = await get_user_by_username(body.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")

    user = await create_user(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
        role=body.role,
    )
    await act.log(
        admin,
        "create_user",
        resource=f"user:{user.id}",
        detail={"new_username": body.username, "role": body.role},
        request=request,
    )
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
    )
