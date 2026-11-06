"""Keep the unauthenticated legacy UI away from team data by default."""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path


def has_team_accounts(database: str | Path) -> bool:
    path = Path(database).resolve()
    if not path.exists():
        return False
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rm_users'").fetchone()
            return bool(table and conn.execute("SELECT 1 FROM rm_users LIMIT 1").fetchone())
    except sqlite3.DatabaseError as exc:
        raise RuntimeError("无法核验数据库的团队账号边界，旧界面已停止；请先修复或核验数据") from exc


def ensure_legacy_access(database: str | Path, *, server_address: str | None, admin_mode: bool | None = None) -> None:
    if not has_team_accounts(database):
        return
    enabled = os.getenv("REFMIND_LEGACY_ADMIN_MODE", "").strip().casefold() == "true" if admin_mode is None else admin_mode is True
    address = str(server_address or "").strip().casefold()
    if not enabled or address not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("该数据库已有团队账号，旧界面不提供账号隔离。请使用新的科研工作区；管理员本机回退须显式设置REFMIND_LEGACY_ADMIN_MODE=true，并将server.address设为127.0.0.1或localhost")


__all__ = ["has_team_accounts", "ensure_legacy_access"]
