"""One writer process per database, shared by Streamlit and FastAPI."""
from __future__ import annotations
import os
from pathlib import Path
from threading import RLock

_handles: dict[str, tuple[str, object]] = {}
_guard = RLock()


def acquire_runtime_lock(database: Path, mode: str) -> None:
    key = str(database.resolve())
    with _guard:
        if key in _handles:
            if _handles[key][0] != mode:
                raise RuntimeError("同一数据库不能由 Streamlit 与 API 同时运行")
            return
        target = database.with_suffix(database.suffix + ".writer.lock")
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = target.open("a+b")
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("此数据库已有 RefMind 服务运行，请先停止旧服务") from exc
        _handles[key] = (mode, handle)


def release_runtime_lock(database: Path) -> None:
    with _guard:
        entry = _handles.pop(str(database.resolve()), None)
        if entry:
            entry[1].close()
