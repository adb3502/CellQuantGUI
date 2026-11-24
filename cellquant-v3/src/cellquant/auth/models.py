"""Data models for users, projects, and activity log."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class User:
    id: int
    username: str
    email: str
    hashed_password: str
    role: str  # "admin" | "user"
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool


@dataclass
class Project:
    id: int
    user_id: int
    name: str
    description: str
    folder_path: str
    settings_json: str  # JSON blob
    created_at: datetime
    updated_at: datetime


@dataclass
class ActivityLog:
    id: int
    user_id: int
    username: str
    action: str       # e.g. "login", "run_quantification", "export"
    resource: str     # e.g. "session:abc123", "project:42"
    detail_json: str  # JSON blob with extra context
    ip_address: str
    timestamp: datetime


# SQLite schema DDL
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    NOT NULL UNIQUE,
    email           TEXT    NOT NULL UNIQUE,
    hashed_password TEXT    NOT NULL,
    role            TEXT    NOT NULL DEFAULT 'user',
    created_at      TEXT    NOT NULL,
    last_login      TEXT,
    is_active       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS projects (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name         TEXT    NOT NULL,
    description  TEXT    NOT NULL DEFAULT '',
    folder_path  TEXT    NOT NULL DEFAULT '',
    settings_json TEXT   NOT NULL DEFAULT '{}',
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS activity_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    username    TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    resource    TEXT    NOT NULL DEFAULT '',
    detail_json TEXT    NOT NULL DEFAULT '{}',
    ip_address  TEXT    NOT NULL DEFAULT '',
    timestamp   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_activity_user   ON activity_log(user_id);
CREATE INDEX IF NOT EXISTS idx_activity_time   ON activity_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_projects_user   ON projects(user_id);
"""
