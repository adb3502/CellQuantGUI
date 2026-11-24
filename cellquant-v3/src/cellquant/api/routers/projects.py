"""Project management endpoints."""

from __future__ import annotations

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from cellquant.auth import activity as act
from cellquant.auth.database import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    update_project,
)
from cellquant.auth.middleware import CurrentUser, get_current_user

router = APIRouter(prefix="/projects", tags=["projects"])


# ── Schemas ────────────────────────────────────────────────────────


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: str
    folder_path: str
    settings: dict
    created_at: str
    updated_at: str


class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""
    folder_path: str = ""
    settings: dict = {}


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    folder_path: Optional[str] = None
    settings: Optional[dict] = None


# ── Routes ─────────────────────────────────────────────────────────


@router.get("", response_model=List[ProjectResponse])
async def get_projects(current_user: CurrentUser = Depends(get_current_user)):
    projects = await list_projects(current_user.id)
    return [_to_response(p) for p in projects]


@router.post("", response_model=ProjectResponse, status_code=201)
async def new_project(
    body: CreateProjectRequest,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await create_project(
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        folder_path=body.folder_path,
        settings=body.settings,
    )
    await act.log(
        current_user,
        "create_project",
        resource=f"project:{project.id}",
        detail={"name": body.name},
        request=request,
    )
    return _to_response(project)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_one_project(
    project_id: int,
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await get_project(project_id, current_user.id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _to_response(project)


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_one_project(
    project_id: int,
    body: UpdateProjectRequest,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await get_project(project_id, current_user.id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    updates: dict = {}
    if body.name is not None:
        updates["name"] = body.name
    if body.description is not None:
        updates["description"] = body.description
    if body.folder_path is not None:
        updates["folder_path"] = body.folder_path
    if body.settings is not None:
        updates["settings_json"] = json.dumps(body.settings)

    if updates:
        await update_project(project_id, current_user.id, **updates)
        await act.log(
            current_user,
            "update_project",
            resource=f"project:{project_id}",
            detail={"fields": list(updates.keys())},
            request=request,
        )

    updated = await get_project(project_id, current_user.id)
    return _to_response(updated)


@router.delete("/{project_id}", status_code=204)
async def delete_one_project(
    project_id: int,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    project = await get_project(project_id, current_user.id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    await delete_project(project_id, current_user.id)
    await act.log(
        current_user,
        "delete_project",
        resource=f"project:{project_id}",
        detail={"name": project.name},
        request=request,
    )


def _to_response(p) -> ProjectResponse:
    return ProjectResponse(
        id=p.id,
        name=p.name,
        description=p.description,
        folder_path=p.folder_path,
        settings=json.loads(p.settings_json),
        created_at=p.created_at.isoformat(),
        updated_at=p.updated_at.isoformat(),
    )
