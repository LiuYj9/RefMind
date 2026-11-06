import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from refmind import storage
from refmind.config import settings
from refmind.services import ingestion
from refmind.storage.connection import connect
from refmind.storage.runtime_lock import acquire_runtime_lock, release_runtime_lock
from refmind.web import store
from refmind.web.api import create_app


class StartupRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.stack = ExitStack()
        for key, path in {"database_path": self.base / "test.db", "upload_dir": self.base / "uploads",
                          "parsed_dir": self.base / "parsed", "docstore_dir": self.base / "images",
                          "chroma_persist_dir": self.base / "vectors"}.items():
            self.stack.enter_context(patch.object(settings, key, path))
        store.initialize()
        with connect() as conn:
            conn.execute("INSERT INTO rm_users(id,username,password_hash,created_at) VALUES(1,'test','unused',0)")

    def tearDown(self):
        release_runtime_lock(settings.database_path)
        self.stack.close()
        self.tmp.cleanup()

    def test_recovery_holds_writer_lock_and_precedes_workers_preserving_ready_marker(self):
        library = store.create_library(1, "Private")
        incomplete = storage.create_document(library["id"], "incomplete.pdf", status="indexing")
        ready = storage.create_document(library["id"], "refmind-ready-marker.pdf", status="ready")
        task = store.create_task(1, "upload", {"group_id": library["id"], "path": "fixture", "filename": "paper.pdf", "sha256": "a" * 64})
        events = []
        original = ingestion.recover_incomplete_ingestions
        def recover():
            with self.assertRaises(RuntimeError):
                acquire_runtime_lock(settings.database_path, "streamlit")
            self.assertEqual(store.task(task["id"])["status"], "interrupted")
            events.append("recover")
            return original()
        runner = Mock()
        def start_runner():
            events.append("workers")
            self.assertIsNone(storage.get_document(incomplete))
            self.assertEqual(storage.get_document(ready).filename, "refmind-ready-marker.pdf")
            return runner
        with patch.object(ingestion, "recover_incomplete_ingestions", side_effect=recover), \
             patch.object(ingestion, "remove_document", side_effect=storage.delete_document), \
             patch("refmind.web.api.JobRunner", side_effect=start_runner):
            with TestClient(create_app(start_workers=False)) as client:
                self.assertEqual(client.app.state.startup_recovery, {"recovered": [incomplete], "failed": []})
                self.assertEqual(client.get("/api/health").status_code, 200)
        self.assertEqual(events, ["recover", "workers"])
        runner.close.assert_called_once()
        acquire_runtime_lock(settings.database_path, "streamlit")

    def test_recovery_exception_starts_no_workers_and_releases_lock_and_provider_flag(self):
        before = settings.allow_openai_library
        with patch.object(ingestion, "recover_incomplete_ingestions", side_effect=RuntimeError("simulated recovery failure")), \
             patch("refmind.web.api.JobRunner") as runner:
            with self.assertRaises(RuntimeError):
                with TestClient(create_app(start_workers=False)):
                    pass
        runner.assert_not_called()
        self.assertEqual(settings.allow_openai_library, before)
        acquire_runtime_lock(settings.database_path, "streamlit")


if __name__ == "__main__":
    unittest.main()
