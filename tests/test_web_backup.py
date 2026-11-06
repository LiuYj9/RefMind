"""Offline archive, ledger and original-path recovery invariants."""
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack, closing
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from unittest.mock import patch

from refmind import storage
from refmind.config import settings
from refmind.llm import openai_metering
from refmind.storage.connection import connect
from refmind.storage.runtime_lock import acquire_runtime_lock, release_runtime_lock
from refmind.web import manage, store, jobs


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        self.stack = ExitStack()
        for key, path in {"database_path": self.base / "data" / "refmind.db", "upload_dir": self.base / "data" / "uploads",
                          "parsed_dir": self.base / "data" / "parsed", "docstore_dir": self.base / "data" / "images",
                          "chroma_persist_dir": self.base / "data" / "vectors"}.items():
            self.stack.enter_context(patch.object(settings, key, path))
        self.ledger = self.base / "billing" / "usage.sqlite3"
        self.stack.enter_context(patch.dict(os.environ, {"OPENAI_USAGE_DB": str(self.ledger)}))
        store.initialize()
        with connect() as conn:
            conn.execute("INSERT INTO rm_users(id,username,password_hash,created_at) VALUES(1,'test','unused',0)")
        self.group = storage.create_group("legacy")
        self.source = settings.upload_dir / "paper.pdf"
        self.source.write_bytes(b"offline source fixture")
        self.image = settings.docstore_dir / "figure.png"
        self.image.write_bytes(b"offline image fixture")
        self.parsed = settings.parsed_dir / "paper.json"
        self.parsed.write_text(json.dumps({"source": str(self.source), "image_path": str(self.image)}), encoding="utf-8")
        self.doc_id = storage.create_document(self.group.id, "paper.pdf", original_path=str(self.source), status="parsing")
        storage.update_document(self.doc_id, parsed_json_path=str(self.parsed), status="ready")
        with openai_metering.connect() as conn:
            conn.execute("INSERT INTO calls VALUES('paid',0,'2026-11-05','model','chat',2,1.25,'ok',10,5,1)")
            conn.execute("INSERT INTO calls VALUES('unknown',0,'2026-11-05','model','chat',2.5,NULL,'unknown',NULL,NULL,1)")
        self.archive = self.base / "backups" / "backup.zip"

    def tearDown(self):
        self.stack.close()
        self.tmp.cleanup()

    def remove_originals(self, *, keep_ledger=False):
        locations = manage._locations()
        # Every recursive deletion is confined to this test's own temp workspace.
        for key, path in locations.items():
            if key == "usage_ledger" and keep_ledger:
                continue
            self.assertTrue(path.resolve().is_relative_to(self.base))
            self.assertNotEqual(path.resolve(), self.base)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def rewrite_archive(self, mutation):
        target = self.base / "modified.zip"
        with ZipFile(self.archive) as original, ZipFile(target, "w", ZIP_DEFLATED) as output:
            values = {name: original.read(name) for name in original.namelist()}
            mutation(values)
            for name, data in values.items():
                output.writestr(name, data)
        return target

    def test_backup_includes_cost_ledger_inventory_and_preserves_source_state(self):
        task = store.create_task(1, "search", {"query": "offline"})
        before = manage._sha(settings.database_path)
        manifest = manage.create_backup(self.archive)
        self.assertEqual(store.task(task["id"])["status"], "queued")
        self.assertEqual(manage._sha(settings.database_path), before)
        self.assertEqual(manifest["version"], 2)
        self.assertIn("usage_ledger.sqlite3", manifest["files"])
        with ZipFile(self.archive) as archive:
            self.assertEqual(set(archive.namelist()) - {"manifest.json"}, set(manifest["files"]))
            for name, info in manifest["files"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), info["sha256"])
        self.assertTrue(self.archive.with_name(self.archive.name + ".sha256").is_file())

    def test_extract_is_explicitly_not_a_relocated_running_database(self):
        manage.create_backup(self.archive)
        result = manage.restore_backup(self.archive, self.base / "inspection")
        self.assertFalse(result["installed"])
        with closing(sqlite3.connect(self.base / "inspection" / "database.sqlite3")) as conn:
            self.assertEqual(conn.execute("SELECT original_path FROM documents WHERE id=?", (self.doc_id,)).fetchone()[0], str(self.source))
        self.assertIn("不能仅改.env", (self.base / "inspection" / "RESTORE-INSTRUCTIONS.txt").read_text(encoding="utf-8"))

    def test_original_path_restore_preserves_absolute_references_and_budget(self):
        manage.create_backup(self.archive)
        self.remove_originals()
        result = manage.restore_backup(self.archive, self.base / "restore-stage", original_paths=True)
        self.assertTrue(result["installed"])
        row = storage.get_document(self.doc_id)
        self.assertEqual(row.original_path, str(self.source))
        self.assertTrue(Path(row.original_path).is_file())
        self.assertEqual(json.loads(self.parsed.read_text(encoding="utf-8"))["image_path"], str(self.image))
        with closing(sqlite3.connect(self.ledger)) as conn:
            self.assertEqual(conn.execute("SELECT SUM(COALESCE(cost,reserved)) FROM calls").fetchone()[0], 3.75)
            self.assertEqual(conn.execute("SELECT status FROM calls WHERE id='unknown'").fetchone()[0], "unknown")

    def test_existing_ledger_is_never_overwritten_by_older_backup(self):
        manage.create_backup(self.archive)
        with openai_metering.connect() as conn:
            conn.execute("INSERT INTO calls VALUES('newer',1,'2026-11-05','model','chat',1,1,'ok',1,1,1)")
        self.remove_originals(keep_ledger=True)
        before = manage._sha(self.ledger)
        with self.assertRaisesRegex(ValueError, "费用账本"):
            manage.restore_backup(self.archive, self.base / "refused", original_paths=True)
        self.assertEqual(manage._sha(self.ledger), before)
        self.assertFalse(settings.database_path.exists())

    def test_existing_original_data_and_existing_staging_are_not_overwritten(self):
        manage.create_backup(self.archive)
        before = manage._sha(self.source)
        with self.assertRaises(ValueError):
            manage.restore_backup(self.archive, self.base / "stage", original_paths=True)
        self.assertEqual(manage._sha(self.source), before)
        existing = self.base / "existing"
        existing.mkdir()
        (existing / "keep").write_text("keep")
        with self.assertRaises(ValueError):
            manage.restore_backup(self.archive, existing)
        self.assertEqual((existing / "keep").read_text(), "keep")

    def test_changed_storage_paths_require_explicit_migration(self):
        manage.create_backup(self.archive)
        with patch.object(settings, "upload_dir", self.base / "elsewhere"):
            with self.assertRaisesRegex(ValueError, "原数据路径"):
                manage.restore_backup(self.archive, self.base / "stage", original_paths=True)

    def test_corrupt_file_and_zip_traversal_are_rejected_before_install(self):
        manage.create_backup(self.archive)
        modified = self.rewrite_archive(lambda values: values.__setitem__("uploads/paper.pdf", b"tampered"))
        with self.assertRaises(ValueError):
            manage.restore_backup(modified, self.base / "invalid-stage")
        self.assertFalse((self.base / "invalid-stage").exists())
        modified.unlink()
        modified = self.rewrite_archive(lambda values: values.__setitem__("../escape", b"unsafe"))
        with self.assertRaises(ValueError):
            manage.restore_backup(modified, self.base / "unsafe-stage")
        self.assertFalse((self.base / "escape").exists())

    def test_whole_archive_checksum_is_checked_when_available(self):
        manage.create_backup(self.archive)
        with self.archive.open("ab") as handle:
            handle.write(b"modified")
        with self.assertRaisesRegex(ValueError, "整体SHA256"):
            manage.restore_backup(self.archive, self.base / "stage")

    def test_backup_does_not_write_into_a_data_root_or_replace_archive(self):
        with self.assertRaises(ValueError):
            manage.create_backup(settings.upload_dir / "backup.zip")
        manage.create_backup(self.archive)
        before = manage._sha(self.archive)
        with self.assertRaises(ValueError):
            manage.create_backup(self.archive)
        self.assertEqual(manage._sha(self.archive), before)

    def test_backup_refuses_a_running_api_writer(self):
        acquire_runtime_lock(settings.database_path, "api")
        try:
            with self.assertRaises(RuntimeError):
                manage.create_backup(self.archive)
        finally:
            release_runtime_lock(settings.database_path)
        self.assertFalse(self.archive.exists())

    def test_actual_chroma_metadata_survives_original_path_restore(self):
        vector_dir = settings.chroma_persist_dir / "group_1"
        create = "import sys,chromadb;from chromadb.config import Settings;c=chromadb.PersistentClient(path=sys.argv[1],settings=Settings(anonymized_telemetry=False));v=c.get_or_create_collection('fixture');v.add(ids=['c1'],documents=['evidence'],embeddings=[[0.,1.]],metadatas=[{'image_path':sys.argv[2]}])"
        subprocess.run([sys.executable, "-B", "-c", create, str(vector_dir), str(self.image)], check=True, capture_output=True, timeout=45)
        manage.create_backup(self.archive)
        self.remove_originals()
        manage.restore_backup(self.archive, self.base / "stage", original_paths=True)
        query = "import sys,chromadb;from chromadb.config import Settings;c=chromadb.PersistentClient(path=sys.argv[1],settings=Settings(anonymized_telemetry=False));r=c.get_collection('fixture').query(query_embeddings=[[0.,1.]],n_results=1);assert r['metadatas'][0][0]['image_path']==sys.argv[2];print('verified')"
        result = subprocess.run([sys.executable, "-B", "-c", query, str(vector_dir), str(self.image)], check=True, capture_output=True, timeout=45)
        self.assertIn(b"verified", result.stdout)

    def test_upload_staging_ttl_skips_active_and_outside_files(self):
        root = jobs.workspace() / "uploads"
        root.mkdir(exist_ok=True)
        old, active = root / "old.pdf", root / "active.pdf"
        old.write_bytes(b"old")
        active.write_bytes(b"active")
        def task(path, status):
            record = store.create_task(1, "upload", {"path": str(path), "sha256": "a" * 64, "group_id": self.group.id, "filename": "paper.pdf"})
            with connect() as conn:
                conn.execute("UPDATE rm_tasks SET status=?,updated_at=? WHERE id=?", (status, time.time() - 8 * 86400, record["id"]))
            return record
        expired = task(old, "cancelled")
        queued = task(active, "queued")
        outside = task(self.image, "failed")
        with patch.object(jobs.LOG, "exception"):
            result = jobs.expire_upload_staging()
        self.assertEqual(result["uploads_expired"], [expired["id"]])
        self.assertEqual(result["upload_cleanup_failed"], [outside["id"]])
        self.assertFalse(old.exists())
        self.assertTrue(active.exists())
        self.assertTrue(self.image.exists())
        self.assertFalse(store.task(queued["id"])["payload"].get("staging_expired", False))
        self.assertTrue(store.task(expired["id"])["payload"]["staging_expired"])
        with connect() as conn:
            conn.execute("UPDATE rm_tasks SET status='queued',cancel_requested=0 WHERE id=?", (expired["id"],))
        runner = jobs.JobRunner(workers=1)
        try:
            with patch("refmind.services.ingestion.ingest_pdf") as ingest, patch.object(jobs.LOG, "exception"):
                runner._run(expired["id"])
            ingest.assert_not_called()
        finally:
            runner.close()
        self.assertIn("重新上传", store.task(expired["id"])["error"])


if __name__ == "__main__":
    unittest.main()
