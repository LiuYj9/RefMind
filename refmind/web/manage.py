"""Offline administration and integrity-checked, original-path disaster recovery."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

from ..config import settings
from ..llm.openai_metering import ledger_path
from ..storage.connection import connect
from ..storage.runtime_lock import acquire_runtime_lock, release_runtime_lock
from . import store


def _locations():
    return {"database": settings.database_path.resolve(), "uploads": settings.upload_dir.resolve(),
            "parsed": settings.parsed_dir.resolve(), "docstore": settings.docstore_dir.resolve(),
            "vectors": settings.chroma_persist_dir.resolve(),
            "research_workspace": (settings.database_path.parent / "research_workspace").resolve(),
            "usage_ledger": ledger_path().resolve()}


def _sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _link(path: Path):
    return path.is_symlink() or (getattr(path, "is_junction", lambda: False)())


def _files(root: Path):
    if _link(root):
        raise ValueError("数据根目录不能是符号链接或目录联接")
    for directory, children, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*children, *names]:
            if _link(base / name):
                raise ValueError("数据目录包含链接，备份已停止")
        for name in names:
            path = base / name
            if path.is_file():
                yield path


def _snapshot(source: Path, target: Path):
    if not source.is_file() or _link(source):
        raise ValueError("SQLite源文件不存在或不是普通文件")
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original, closing(sqlite3.connect(target)) as snapshot:
        original.backup(snapshot)
        snapshot.commit()
        if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite快照校验失败")


def create_backup(destination: str | Path) -> dict:
    """Never initializes/migrates the source database or resets its cost ledger."""
    destination = Path(destination).resolve()
    checksum = destination.with_name(destination.name + ".sha256")
    if destination.exists() or checksum.exists():
        raise ValueError("备份或校验文件已存在，请使用新文件名")
    locations = _locations()
    roots = {key: value for key, value in locations.items() if key not in {"database", "usage_ledger"}}
    if any(destination.is_relative_to(root) for root in roots.values()) or destination in locations.values():
        raise ValueError("备份不能位于被归档的数据目录或覆盖数据文件")
    destination.parent.mkdir(parents=True, exist_ok=True)
    acquire_runtime_lock(settings.database_path, "management")
    created = False
    try:
        manifest = {"version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
                    "restore_strategy": "original_paths_only", "api_credentials_included": False,
                    "contains_user_data": True, "files": {},
                    "storage": {key: {"original_path": str(path), "present": path.exists()} for key, path in locations.items()}}
        excluded = {str(path) + suffix for path in (locations["database"], locations["usage_ledger"])
                    for suffix in ("", "-wal", "-shm", "-journal")}
        with tempfile.TemporaryDirectory(prefix=".refmind-backup-", dir=destination.parent) as tmp:
            database_copy = Path(tmp) / "database.sqlite3"
            _snapshot(locations["database"], database_copy)
            ledger_copy = Path(tmp) / "usage_ledger.sqlite3"
            if locations["usage_ledger"].exists():
                _snapshot(locations["usage_ledger"], ledger_copy)
            def add(archive, source, name):
                before = source.stat()
                digest, size = hashlib.sha256(), 0
                with source.open("rb") as handle, archive.open(name, "w", force_zip64=True) as output:
                    while block := handle.read(1024 * 1024):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                after = source.stat()
                if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                    raise ValueError("备份期间源文件变化，请停止所有写入进程后重试")
                manifest["files"][name] = {"sha256": digest.hexdigest(), "size_bytes": size}
            with ZipFile(destination, "x", ZIP_DEFLATED) as archive:
                created = True
                add(archive, database_copy, "database.sqlite3")
                if ledger_copy.exists():
                    add(archive, ledger_copy, "usage_ledger.sqlite3")
                for name, root in roots.items():
                    if root.exists():
                        if not root.is_dir():
                            raise ValueError("配置的数据目录不是目录")
                        for path in _files(root):
                            if str(path.resolve()) not in excluded:
                                add(archive, path, name + "/" + path.relative_to(root).as_posix())
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        with checksum.open("x", encoding="ascii") as handle:
            handle.write(_sha(destination) + "  " + destination.name + "\n")
        return manifest
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        release_runtime_lock(settings.database_path)


def _archive_manifest(archive: ZipFile):
    entries, seen = {}, set()
    for item in archive.infolist():
        name = item.filename
        pure = PurePosixPath(name)
        mode = (item.external_attr >> 16) & 0o170000
        if not name or "\\" in name or ":" in name or pure.is_absolute() or ".." in pure.parts or mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError("归档包含不安全路径或链接")
        if name.casefold() in seen:
            raise ValueError("归档包含重复或大小写冲突的路径")
        seen.add(name.casefold())
        if not item.is_dir():
            entries[name] = item
    if "manifest.json" not in entries or entries["manifest.json"].file_size > 16 * 1024 * 1024:
        raise ValueError("归档缺少有效清单")
    manifest = json.loads(archive.read("manifest.json"))
    if manifest.get("version") != 2 or manifest.get("restore_strategy") != "original_paths_only":
        raise ValueError("仅支持带完整路径和哈希清单的v2备份；旧备份需要人工迁移")
    inventory, storage = manifest.get("files"), manifest.get("storage")
    if not isinstance(inventory, dict) or not isinstance(storage, dict) or set(storage) != set(_locations()):
        raise ValueError("备份清单结构不完整")
    for value in storage.values():
        if not isinstance(value, dict) or type(value.get("present")) is not bool or not isinstance(value.get("original_path"), str) or not Path(value["original_path"]).is_absolute():
            raise ValueError("备份原路径记录无效")
    if storage["database"]["present"] is not True:
        raise ValueError("备份没有原数据库")
    if set(inventory) != set(entries) - {"manifest.json"} or "database.sqlite3" not in inventory:
        raise ValueError("归档文件与清单不一致")
    if len(entries) > 100000 or sum(v.file_size for v in entries.values()) > 50 * 1024 ** 3:
        raise ValueError("归档超过恢复规模限制")
    for name, info in inventory.items():
        if not isinstance(info, dict) or type(info.get("size_bytes")) is not int or info["size_bytes"] != entries[name].file_size:
            raise ValueError("归档大小与清单不一致")
        digest = info.get("sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("归档哈希无效")
        if name not in {"database.sqlite3", "usage_ledger.sqlite3"}:
            if "/" not in name or name.split("/", 1)[0] not in set(storage) - {"database", "usage_ledger"}:
                raise ValueError("归档包含未登记的目录")
            if storage[name.split("/", 1)[0]]["present"] is not True:
                raise ValueError("归档文件与原目录状态不一致")
    if bool(storage["usage_ledger"].get("present")) != ("usage_ledger.sqlite3" in inventory):
        raise ValueError("费用账本清单不一致")
    return manifest, entries


def _extract_verified(source: Path, destination: Path):
    checksum = source.with_name(source.name + ".sha256")
    if checksum.exists():
        parts = checksum.read_text(encoding="ascii").split()
        if not parts:
            raise ValueError("备份整体SHA256校验文件无效")
        expected = parts[0]
        if expected != _sha(source):
            raise ValueError("备份整体SHA256校验失败")
    with ZipFile(source) as archive:
        manifest, entries = _archive_manifest(archive)
        destination.mkdir(parents=True, exist_ok=False)
        try:
            for name, item in entries.items():
                target = destination.joinpath(*PurePosixPath(name).parts)
                if not target.resolve().is_relative_to(destination):
                    raise ValueError("归档目标越界")
                target.parent.mkdir(parents=True, exist_ok=True)
                digest, size = hashlib.sha256(), 0
                with archive.open(item) as original, target.open("xb") as output:
                    while block := original.read(1024 * 1024):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                if name != "manifest.json" and (size != manifest["files"][name]["size_bytes"] or digest.hexdigest() != manifest["files"][name]["sha256"]):
                    raise ValueError("恢复文件SHA256校验失败")
            for name in ("database.sqlite3", "usage_ledger.sqlite3"):
                path = destination / name
                if path.exists():
                    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
                        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                            raise ValueError("恢复SQLite校验失败")
        except BaseException:
            # We created this previously absent staging directory; never remove
            # an existing user directory or follow an archive-provided link.
            if destination.resolve() == destination and not _link(destination):
                shutil.rmtree(destination)
            raise
    return manifest


def restore_backup(source: str | Path, destination: str | Path, *, original_paths: bool = False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("恢复提取目录必须不存在，不覆盖已有目录")
    manifest = _extract_verified(source, destination)
    instructions = "本目录是已校验的提取副本，不是可直接启动的数据目录。\nSQLite、Chroma和解析元数据可能含原绝对路径；不能仅改.env指向这里。\n停止RefMind服务及所有评测/模型进程，保留密钥配置。\n使用restore ARCHIVE NEW_STAGING --original-paths，当前各配置路径须与manifest原路径一致，所有原数据目标必须不存在。\n费用账本若已存在，安装会停止，绝不以旧快照覆盖或回退费用记录。\n安装后先验证账户、文档、PDF引用和检索，再开放访问。\n"
    (destination / "RESTORE-INSTRUCTIONS.txt").write_text(instructions, encoding="utf-8")
    if not original_paths:
        return {"manifest": manifest, "staging": str(destination), "installed": False}
    locations = _locations()
    for name, current in locations.items():
        recorded = manifest["storage"][name].get("original_path")
        if not isinstance(recorded, str) or Path(recorded).resolve() != current:
            raise ValueError("当前配置与原数据路径不一致；不迁移绝对路径，请使用原路径配置")
    acquire_runtime_lock(settings.database_path, "management")
    created_files, created_dirs = [], []
    try:
        targets = {key: path for key, path in locations.items() if manifest["storage"][key].get("present")}
        for name, path in targets.items():
            if path.exists() or path.is_symlink():
                if name == "usage_ledger":
                    raise ValueError("现有费用账本必须保留；拒绝用旧备份覆盖或回退预算")
                raise ValueError("原数据目标已存在，恢复不会覆盖任何现有数据")
        def mkdir(path):
            missing, cursor = [], path
            while not cursor.exists():
                missing.append(cursor)
                cursor = cursor.parent
            if _link(cursor):
                raise ValueError("恢复路径父目录不能是链接")
            for folder in reversed(missing):
                folder.mkdir()
                created_dirs.append(folder)
        for name, target in targets.items():
            if name not in {"database", "usage_ledger"}:
                mkdir(target)
        for name in manifest["files"]:
            if name == "database.sqlite3":
                target = locations["database"]
            elif name == "usage_ledger.sqlite3":
                target = locations["usage_ledger"]
            else:
                prefix, relative = name.split("/", 1)
                target = locations[prefix].joinpath(*PurePosixPath(relative).parts)
            mkdir(target.parent)
            with (destination / name).open("rb") as original, target.open("xb") as output:
                created_files.append(target)
                shutil.copyfileobj(original, output)
            if _sha(target) != manifest["files"][name]["sha256"]:
                raise ValueError("原路径安装后校验失败")
        return {"manifest": manifest, "staging": str(destination), "installed": True}
    except BaseException:
        # Only paths created by this invocation are eligible for rollback.
        for path in reversed(created_files):
            path.unlink(missing_ok=True)
        for path in reversed(created_dirs):
            try:
                path.rmdir()
            except OSError:
                pass
        raise
    finally:
        release_runtime_lock(settings.database_path)


def main():
    parser = argparse.ArgumentParser(description="RefMind离线管理：先停止服务、评测及模型写入进程")
    commands = parser.add_subparsers(dest="command", required=True)
    user = commands.add_parser("create-user")
    user.add_argument("username")
    user.add_argument("--admin", action="store_true")
    claim = commands.add_parser("claim-library")
    claim.add_argument("group_id", type=int)
    claim.add_argument("username")
    commands.add_parser("unclaimed-libraries")
    backup = commands.add_parser("backup")
    backup.add_argument("destination", type=Path)
    restore = commands.add_parser("restore")
    restore.add_argument("archive", type=Path)
    restore.add_argument("destination", type=Path)
    restore.add_argument("--original-paths", action="store_true", help="仅安装到manifest记录的原路径，目标须不存在")
    reindex = commands.add_parser("reindex-legacy", help="从已核对原文重建历史索引，保留文献编号与引用")
    reindex.add_argument("doc_id", type=int)
    reindex.add_argument("--expected-sha256", required=True, help="已从独立入库记录核对的文件SHA256")
    maintenance = commands.add_parser("run-reindex-request", help="停止写入服务后执行一项维护申请")
    maintenance.add_argument("task_id")
    maintenance.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if args.command == "run-reindex-request":
        from .maintenance import execute_request
        result = execute_request(args.task_id, expected_sha256=args.expected_sha256)
        print(f"维护申请 {result['id']}：{result['stage']}")
        return
    if args.command == "reindex-legacy":
        from ..services.reindex import rebuild_legacy_index
        row = rebuild_legacy_index(args.doc_id, expected_sha256=args.expected_sha256)
        print(f"文献 {row.id} 已就绪，索引版本 {row.index_version}，原文及历史引用保留")
        return
    if args.command == "backup":
        create_backup(args.destination)
        print(f"已生成备份及SHA256校验文件：{args.destination}")
        return
    if args.command == "restore":
        result = restore_backup(args.archive, args.destination, original_paths=args.original_paths)
        print("已验证并安装到原数据路径；密钥配置须自行保留，开放服务前核验引用与检索" if result["installed"] else "已验证并提取；尚未安装为运行数据，请阅读RESTORE-INSTRUCTIONS.txt")
        return
    acquire_runtime_lock(settings.database_path, "management")
    try:
        store.initialize()
        if args.command == "create-user":
            password = getpass.getpass("新账号密码（至少12字符）: ")
            if password != getpass.getpass("再次输入密码: "):
                raise SystemExit("两次密码不一致")
            created = store.create_user(args.username, password, admin=args.admin)
            print(f"已创建账号 {created['username']}（ID {created['id']}）")
        elif args.command == "unclaimed-libraries":
            with connect() as conn:
                for row in conn.execute("SELECT g.id,g.name FROM groups g LEFT JOIN rm_libraries l ON l.group_id=g.id WHERE l.group_id IS NULL"):
                    print(f"{row['id']}\t{row['name']}")
        elif args.command == "claim-library":
            with connect() as conn:
                owner = conn.execute("SELECT id FROM rm_users WHERE username=?", (args.username,)).fetchone()
                group = conn.execute("SELECT id,name FROM groups WHERE id=?", (args.group_id,)).fetchone()
                if not owner or not group:
                    raise SystemExit("账号或文献库不存在")
                conn.execute("INSERT INTO rm_libraries VALUES(?,?,?,'private')", (args.group_id, owner["id"], group["name"]))
            print("已登记为该账号的私有库，未修改文档或向量")
    finally:
        release_runtime_lock(settings.database_path)


if __name__ == "__main__":
    main()
