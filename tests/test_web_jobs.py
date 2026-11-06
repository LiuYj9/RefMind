"""Real SQLite workflow integration with deterministic offline ingestion/fetch fixtures."""
import hashlib
import json
import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock

import fitz

from refmind import storage
from refmind.config import settings
from refmind.storage.connection import connect
from refmind.services.fulltext import FetchedPaper
from refmind.services.paper_discovery import PaperCandidate, PaperLocation
from refmind.web import jobs, store


def pdf_bytes(pages=1):
    document = fitz.open()
    for _ in range(pages):
        document.new_page().insert_text((40, 60), "A deterministic test paper\ndoi:10.1234/test")
    data = document.tobytes()
    document.close()
    return data


class WebJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stack = ExitStack()
        base = Path(self.tmp.name)
        for key, path in {"database_path": base / "test.db", "upload_dir": base / "uploads",
                          "parsed_dir": base / "parsed", "docstore_dir": base / "images",
                          "chroma_persist_dir": base / "chroma"}.items():
            self.stack.enter_context(patch.object(settings, key, path))
        store.initialize()
        with connect() as conn:
            conn.execute("INSERT INTO rm_users(id,username,password_hash,created_at) VALUES(1,'alice','unused',?)", (time.time(),))
            conn.execute("INSERT INTO rm_users(id,username,password_hash,created_at) VALUES(2,'bob','unused',?)", (time.time(),))
        self.uid = 1
        self.library = store.create_library(self.uid, "Research", "shared")
        self.session = store.create_session(self.uid, "Read")
        self.runner = jobs.JobRunner(workers=2)
        self.stack.callback(self.runner.close)
        self.ingest_count = 0
        self.parsed_pages = [1]
        self.after_ingest = None
        self.stack.enter_context(patch("refmind.services.ingestion.ingest_pdf", side_effect=self.ingest))
        self.stack.enter_context(patch("refmind.services.ingestion.remove_document", side_effect=self.remove_document))
        self.stack.enter_context(patch("refmind.services.ingestion.remove_group", side_effect=self.remove_group))

    def tearDown(self):
        self.stack.close()
        self.tmp.cleanup()

    def ingest(self, gid, source, filename, **kwargs):
        self.ingest_count += 1
        kwargs.get("progress", lambda *args: None)(.1, "parse")
        doc_id = storage.create_document(gid, filename, status="parsing")
        destination = settings.upload_dir / f"fixture-{doc_id}.pdf"
        shutil.copyfile(source, destination)
        parsed = settings.parsed_dir / f"fixture-{doc_id}.json"
        parsed.write_text(json.dumps({"pages": [{"page": p, "text": "source evidence"} for p in self.parsed_pages]}), encoding="utf-8")
        storage.update_document(doc_id, original_path=str(destination), parsed_json_path=str(parsed), status="ready", num_chunks=1)
        if self.after_ingest:
            self.after_ingest()
        return storage.get_document(doc_id)

    def remove_document(self, doc_id):
        row = storage.get_document(doc_id)
        if row:
            for path in (row.original_path, row.parsed_json_path):
                if path:
                    Path(path).unlink(missing_ok=True)
            storage.delete_document(doc_id)

    def remove_group(self, gid):
        for row in storage.list_documents(gid):
            self.remove_document(row.id)
        storage.delete_group(gid)

    def upload_task(self, body=None, user_id=1):
        body = body or pdf_bytes()
        folder = jobs.workspace() / "uploads"
        folder.mkdir(exist_ok=True)
        path = folder / f"{time.time_ns()}.pdf"
        path.write_bytes(body)
        return store.create_task(user_id, "upload", {"group_id": self.library["id"], "path": str(path),
                    "filename": "paper.pdf", "sha256": hashlib.sha256(body).hexdigest()})

    def retry(self, tid):
        with connect() as conn:
            conn.execute("UPDATE rm_tasks SET status='queued',cancel_requested=0 WHERE id=?", (tid,))
        self.runner._run(tid)
        return store.task(tid)

    def cancel(self, tid):
        with connect() as conn:
            conn.execute("UPDATE rm_tasks SET cancel_requested=1 WHERE id=?", (tid,))

    def reading_task(self, identity="matched", pages=1):
        paper = PaperCandidate("doi:10.1234/test", "A deterministic test paper", doi="10.1234/test",
                    locations=(PaperLocation("https://public.example/a.pdf", is_oa=True),))
        search = store.create_task(self.uid, "search", {})
        store.update_task(search["id"], status="completed", result={"papers": [paper.to_dict()]})
        task = store.create_task(self.uid, "reading", {"session_id": self.session["id"], "search_id": search["id"], "paper_id": paper.paper_id})
        body = pdf_bytes(pages)
        def fetch(candidate, folder, **kwargs):
            folder.mkdir(parents=True, exist_ok=True)
            sha = hashlib.sha256(body).hexdigest()
            path = folder / (sha + ".pdf")
            path.write_bytes(body)
            return FetchedPaper(path, sha, len(body), candidate.locations[0].url, candidate.locations[0].url,
                                "acceptedVersion", "cc-by", identity)
        self.stack.enter_context(patch("refmind.services.fulltext.fetch_public_pdf", side_effect=fetch))
        return task

    def test_upload_commits_source_and_completion_once(self):
        task = self.upload_task()
        self.runner._run(task["id"])
        self.runner._run(task["id"])
        result = store.task(task["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.ingest_count, 1)
        self.assertFalse(Path(task["payload"]["path"]).exists())
        self.assertEqual(storage.list_documents(self.library["id"])[0].filename, "paper.pdf")
        with connect() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM rm_document_sources").fetchone()[0], 1)

    def test_failure_after_ingest_before_source_registration_recovers_without_reingest(self):
        task = self.upload_task()
        with patch.object(self.runner, "_register", side_effect=RuntimeError("simulated registration interruption")):
            self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "failed")
        self.assertEqual(len(storage.list_documents(self.library["id"])), 1)
        result = self.retry(task["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.ingest_count, 1)

    def test_concurrent_same_bytes_imports_are_deduplicated(self):
        body = pdf_bytes()
        tasks = [self.upload_task(body) for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda t: self.runner._run(t["id"]), tasks))
        self.assertEqual(self.ingest_count, 1)
        self.assertEqual(len(storage.list_documents(self.library["id"])), 1)
        self.assertTrue(all(store.task(t["id"])["status"] == "completed" for t in tasks))

    def test_cancellation_after_ingestion_removes_uncommitted_document(self):
        task = self.upload_task()
        self.after_ingest = lambda: self.cancel(task["id"])
        self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "cancelled")
        self.assertEqual(storage.list_documents(self.library["id"]), [])
        # The owned upload remains available for an explicit retry.
        self.assertTrue(Path(task["payload"]["path"]).exists())
        self.after_ingest = None
        self.assertEqual(self.retry(task["id"])["status"], "completed")

    def test_cancelled_before_start_does_not_ingest(self):
        task = self.upload_task()
        self.cancel(task["id"])
        self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "cancelled")
        self.assertEqual(self.ingest_count, 0)

    def test_tampered_upload_fails_before_ingestion(self):
        task = self.upload_task()
        Path(task["payload"]["path"]).write_bytes(b"tampered")
        self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "failed")
        self.assertEqual(self.ingest_count, 0)

    def test_viewer_cannot_import(self):
        store.grant_member(self.uid, self.library["id"], 2, "viewer")
        task = self.upload_task(user_id=2)
        self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "failed")
        self.assertEqual(self.ingest_count, 0)

    def test_revoked_permission_after_ingestion_removes_uncommitted_document(self):
        store.grant_member(self.uid, self.library["id"], 2, "editor")
        task = self.upload_task(user_id=2)
        def revoke():
            with connect() as conn:
                conn.execute("DELETE FROM rm_members WHERE user_id=2")
        self.after_ingest = revoke
        self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "failed")
        self.assertEqual(storage.list_documents(self.library["id"]), [])

    def test_unknown_fulltext_waits_for_manual_identity_confirmation(self):
        task = self.reading_task(identity="unknown")
        self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "needs_review")
        self.assertEqual(self.ingest_count, 0)
        asset = store.asset(task["id"], self.uid)
        self.assertIsNone(asset["group_id"])
        info = asset["file_info"]
        info["identity_status"] = "user_confirmed"
        with connect() as conn:
            conn.execute("UPDATE rm_assets SET file_info=?,status='downloaded' WHERE id=?", (json.dumps(info), task["id"]))
        self.assertEqual(self.retry(task["id"])["status"], "completed")
        self.assertEqual(self.ingest_count, 1)

    def test_reading_interruption_recovers_ready_document_without_duplicate(self):
        task = self.reading_task()
        with patch.object(self.runner, "_register", side_effect=RuntimeError("interruption")):
            self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "failed")
        self.assertEqual(self.retry(task["id"])["status"], "completed")
        self.assertEqual(self.ingest_count, 1)
        self.assertEqual(store.asset(task["id"], self.uid)["status"], "ready")

    def test_partial_reading_is_not_silently_reparsed_or_ready(self):
        task = self.reading_task(pages=2)
        self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "needs_review")
        self.assertEqual(store.asset(task["id"], self.uid)["status"], "partial")
        self.retry(task["id"])
        self.assertEqual(self.ingest_count, 1)
        self.assertEqual(store.task(task["id"])["status"], "needs_review")

    def test_cancelled_reading_removes_private_copy_and_can_refetch(self):
        task = self.reading_task()
        self.after_ingest = lambda: self.cancel(task["id"])
        self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "cancelled")
        asset = store.asset(task["id"], self.uid)
        self.assertEqual(asset["status"], "cancelled")
        self.assertIsNone(asset["group_id"])
        self.assertFalse(Path(asset["file_info"]["path"]).exists())
        self.after_ingest = None
        self.assertEqual(self.retry(task["id"])["status"], "completed")

    def test_import_copies_are_independent_of_expiring_temporary_assets(self):
        reading = self.reading_task()
        self.runner._run(reading["id"])
        task = store.create_task(self.uid, "import", {"asset_id": reading["id"], "group_id": self.library["id"]})
        self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "completed")
        imported_id = store.task(task["id"])["result"]["document_id"]
        with connect() as conn:
            conn.execute("UPDATE rm_assets SET expires_at=? WHERE id=?", (time.time() - 1, reading["id"]))
        self.assertIn(reading["id"], jobs.expire_assets()["expired"])
        self.assertIsNotNone(storage.get_document(imported_id))
        self.assertTrue(Path(storage.get_document(imported_id).original_path).is_file())

    def test_expired_asset_before_import_does_not_ingest(self):
        reading = self.reading_task()
        self.runner._run(reading["id"])
        with connect() as conn:
            conn.execute("UPDATE rm_assets SET expires_at=? WHERE id=?", (time.time() - 1, reading["id"]))
        task = store.create_task(self.uid, "import", {"asset_id": reading["id"], "group_id": self.library["id"]})
        count = self.ingest_count
        self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "failed")
        self.assertEqual(self.ingest_count, count)

    def test_ambiguous_question_is_persisted_without_retrieval_or_model(self):
        task = store.create_task(self.uid, "answer", {"session_id": self.session["id"], "question": "它的方法是什么？", "mode": "library", "group_id": self.library["id"]})
        with patch("refmind.rag.graph.answer_from_evidence") as answer, patch("refmind.rag.research_pipeline.retrieve_research_candidates") as retrieve:
            self.runner._run(task["id"])
        result = store.task(task["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["status"], "clarification")
        self.assertTrue(result["result"]["task_description"]["needs_clarification"])
        self.assertEqual(len(storage.list_messages(self.session["id"])), 2)
        answer.assert_not_called()
        retrieve.assert_not_called()

    def test_answer_uses_resolved_query_keeps_candidates_and_sanitizes_snapshot(self):
        from langchain_core.documents import Document
        uploaded = self.upload_task()
        self.runner._run(uploaded["id"])
        doc_id = store.task(uploaded["id"])["result"]["document_id"]
        session = store.create_session(self.uid, "Scoped", self.library["id"])
        storage.add_message(session["id"], "user", "first question")
        storage.add_message(session["id"], "assistant", "first answer")
        storage.add_message(session["id"], "user", "dangling turn")
        task = store.create_task(self.uid, "answer", {"session_id": session["id"], "question": "它的方法是什么？", "mode": "library", "selected_doc_ids": [doc_id]})
        docs = [Document(page_content=f"evidence {i}", metadata={"doc_id": doc_id, "group_id": self.library["id"], "chunk_id": str(i), "page": 1,
                    "image_path": r"C:\private\secret.png", "bbox_coordinate_system": "pdf_points"}) for i in range(8)]
        def generate(question, evidence, **kwargs):
            self.assertEqual(question, task["payload"]["question"])
            self.assertTrue(kwargs["apply_research_enhancements"])
            self.assertEqual(len(kwargs["history"]), 2)
            self.assertIn("明确文献", kwargs["task_description"]["resolved_query"])
            return {"answer": "supported answer", "status": "scoped", "documents": evidence, "review_status": "passed",
                    "evidence_review": {"coverage": "scoped"}, "task_description": kwargs["task_description"],
                    "trace": {"stages": {"selected": {"documents": evidence}}, "path": r"C:\private\file"}}
        # The library path binds these functions in the shared research seam.
        # Keep this compatibility fixture offline and independent of feature flags.
        with patch.object(settings, "research_multi_query_enabled", False), \
             patch("refmind.rag.research_pipeline.retrieve_for_task", return_value=docs) as retrieve, \
             patch("refmind.rag.research_pipeline.rerank", side_effect=lambda query, evidence, top_n: evidence) as rank, \
             patch("refmind.rag.graph.answer_from_evidence", side_effect=generate) as answer:
            self.runner._run(task["id"])
            self.runner._run(task["id"])
        self.assertIn("明确文献", retrieve.call_args.args[1])
        self.assertEqual(rank.call_args.kwargs["top_n"], len(docs))
        self.assertEqual(answer.call_count, 1)
        output = store.task(task["id"])["result"]
        self.assertEqual(output["status"], "scoped")
        self.assertEqual(output["review_status"], "passed")
        self.assertNotIn("private", json.dumps(output))
        self.assertEqual(output["evidence"][0]["metadata"]["bbox_coordinate_system"], "pdf_points")

    def selected_metadata_choice(self):
        search = store.create_task(self.uid, "search", {})
        store.update_task(search["id"], status="completed", result={"papers": [PaperCandidate("metadata-only", "Metadata only fixture").to_dict()]})
        return {"search_id": search["id"], "paper_id": "metadata-only"}

    def test_answer_messages_and_snapshot_roll_back_together_on_commit_failure(self):
        task = store.create_task(self.uid, "answer", {"session_id": self.session["id"], "question": "Explain the evidence", "mode": "external", "abstracts": [self.selected_metadata_choice()]})
        with patch("refmind.rag.graph.answer_from_evidence", return_value={"answer": "answer", "status": "refused", "documents": []}), \
             patch.object(self.runner, "_complete", side_effect=RuntimeError("commit failure")):
            self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "failed")
        self.assertEqual(storage.list_messages(self.session["id"]), [])
        with connect() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM rm_answer_evidence").fetchone()[0], 0)
        with patch("refmind.rag.graph.answer_from_evidence", return_value={"answer": "answer", "status": "refused", "documents": []}):
            self.assertEqual(self.retry(task["id"])["status"], "completed")
        self.assertEqual(len(storage.list_messages(self.session["id"])), 2)

    def test_cancelled_generation_does_not_append_partial_history(self):
        task = store.create_task(self.uid, "answer", {"session_id": self.session["id"], "question": "Explain", "mode": "external", "abstracts": [self.selected_metadata_choice()]})
        def generate(*args, **kwargs):
            self.cancel(task["id"])
            return {"answer": "draft", "documents": []}
        with patch("refmind.rag.graph.answer_from_evidence", side_effect=generate):
            self.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "cancelled")
        self.assertEqual(storage.list_messages(self.session["id"]), [])


if __name__ == "__main__":
    unittest.main()
