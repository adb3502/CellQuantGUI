"""Async SQLite database layer for auth + project management."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import aiosqlite

from cellquant.auth.models import SCHEMA_SQL, ActivityLog, Project, User

_DB_PATH: Optional[str] = None


def get_db_path() -> str:
    global _DB_PATH
    if _DB_PATH is None:
        _DB_PATH = os.environ.get(
            "CELLQUANT_DB",
            str(Path(__file__).parent.parent.parent.parent / "data" / "cellquant.db"),
        )
    return _DB_PATH


async def init_db() -> None:
    """Create tables and seed the admin user if the DB is fresh."""
    db_path = get_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA_SQL)
        await db.commit()
        # Seed admin from environment if no users exist
        cursor = await db.execute("SELECT COUNT(*) FROM users")
        (count,) = await cursor.fetchone()
        if count == 0:
            from cellquant.auth.auth import hash_password

            admin_user = os.environ.get("CELLQUANT_ADMIN_USER", "admin")
            admin_pass = os.environ.get("CELLQUANT_ADMIN_PASS", "changeme")
            now = _now()
            await db.execute(
                "INSERT INTO users (username, email, hashed_password, role, created_at, is_active) "
                "VALUES (?, ?, ?, 'admin', ?, 1)",
                (admin_user, f"{admin_user}@localhost", hash_password(admin_pass), now),
            )
            await db.commit()
            print(f"[CellQuant] Admin user '{admin_user}' created (change default password!)")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── User CRUD ──────────────────────────────────────────────────────


async def get_user_by_username(username: str) -> Optional[User]:
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = await cursor.fetchone()
        return _row_to_user(row) if row else None


async def get_user_by_id(user_id: int) -> Optional[User]:
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = await cursor.fetchone()
        return _row_to_user(row) if row else None


async def list_users() -> List[User]:
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [_row_to_user(r) for r in rows]


async def create_user(
    username: str, email: str, hashed_password: str, role: str = "user"
) -> User:
    now = _now()
    async with aiosqlite.connect(get_db_path()) as db:
        cursor = await db.execute(
            "INSERT INTO users (username, email, hashed_password, role, created_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (username, email, hashed_password, role, now),
        )
        await db.commit()
        user_id = cursor.lastrowid
    return await get_user_by_id(user_id)


async def update_user(user_id: int, **fields) -> None:
    allowed = {"email", "hashed_password", "role", "is_active", "last_login"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [user_id]
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)
        await db.commit()


async def delete_user(user_id: int) -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await db.commit()


def _row_to_user(row: aiosqlite.Row) -> User:
    return User(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        hashed_password=row["hashed_password"],
        role=row["role"],
        created_at=datetime.fromisoformat(row["created_at"]),
        last_login=datetime.fromisoformat(row["last_login"]) if row["last_login"] else None,
        is_active=bool(row["is_active"]),
    )


# ── Project CRUD ───────────────────────────────────────────────────


async def list_projects(user_id: int) -> List[Project]:
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM projects WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_project(r) for r in rows]


async def get_project(project_id: int, user_id: int) -> Optional[Project]:
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        )
        row = await cursor.fetchone()
        return _row_to_project(row) if row else None


async def create_project(
    user_id: int,
    name: str,
    description: str = "",
    folder_path: str = "",
    settings: dict | None = None,
) -> Project:
    now = _now()
    settings_json = json.dumps(settings or {})
    async with aiosqlite.connect(get_db_path()) as db:
        cursor = await db.execute(
            "INSERT INTO projects (user_id, name, description, folder_path, settings_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, name, description, folder_path, settings_json, now, now),
        )
        await db.commit()
        project_id = cursor.lastrowid
    return await get_project(project_id, user_id)


async def update_project(project_id: int, user_id: int, **fields) -> None:
    allowed = {"name", "description", "folder_path", "settings_json"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    updates["updated_at"] = _now()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [project_id, user_id]
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            f"UPDATE projects SET {set_clause} WHERE id = ? AND user_id = ?", values
        )
        await db.commit()


async def delete_project(project_id: int, user_id: int) -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "DELETE FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)
        )
        await db.commit()


def _row_to_project(row: aiosqlite.Row) -> Project:
    return Project(
        id=row["id"],
        user_id=row["user_id"],
        name=row["name"],
        description=row["description"],
        folder_path=row["folder_path"],
        settings_json=row["settings_json"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# ── Activity log ───────────────────────────────────────────────────


async def log_activity(
    user_id: int,
    username: str,
    action: str,
    resource: str = "",
    detail: dict | None = None,
    ip_address: str = "",
) -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            "INSERT INTO activity_log (user_id, username, action, resource, detail_json, ip_address, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                username,
                action,
                resource,
                json.dumps(detail or {}),
                ip_address,
                _now(),
            ),
        )
        await db.commit()


async def list_activity(
    user_id: int | None = None,
    action: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> List[ActivityLog]:
    conditions = []
    params: list = []
    if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)
    if action is not None:
        conditions.append("action = ?")
        params.append(action)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params += [limit, offset]
    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            f"SELECT * FROM activity_log {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()
        return [_row_to_activity(r) for r in rows]


async def count_activity(user_id: int | None = None, action: str | None = None) -> int:
    conditions = []
    params: list = []
    if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)
    if action is not None:
        conditions.append("action = ?")
        params.append(action)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    async with aiosqlite.connect(get_db_path()) as db:
        cursor = await db.execute(f"SELECT COUNT(*) FROM activity_log {where}", params)
        (count,) = await cursor.fetchone()
        return count


def _row_to_activity(row: aiosqlite.Row) -> ActivityLog:
    return ActivityLog(
        id=row["id"],
        user_id=row["user_id"],
        username=row["username"],
        action=row["action"],
        resource=row["resource"],
        detail_json=row["detail_json"],
        ip_address=row["ip_address"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
    )
