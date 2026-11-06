"""Reproductions for reviewed queue, retry, scope and citation snapshot defects."""
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import fitz
from langchain_core.documents import Document
from pydantic import ValidationError

from refmind import storage
from refmind.config import settings
from refmind.contracts import ResearchScope
from refmind.storage.connection import connect
from refmind.web import jobs, store
import tests.test_web_api as web_fixture


class WebSecurityTests(unittest.TestCase):
    setUp = web_fixture.WebApiTest.setUp
    tearDown = web_fixture.WebApiTest.tearDown
    login = web_fixture.WebApiTest.login
    post = web_fixture.WebApiTest.post

    def pdf(self):
        document = fitz.open()
        document.new_page().insert_text((40, 60), "Offline upload fixture")
        data = document.tobytes()
        document.close()
        return data

    def scope(self, session_id):
        search = store.create_task(self.alice["id"], "search", {})
        store.update_task(search["id"], status="completed", result={"papers": [{"paper_id": "fixture", "title": "Metadata", "abstract": ""}]})
        return {"session_id": session_id, "question": "Explain", "mode": "external", "abstracts": [{"search_id": search["id"], "paper_id": "fixture"}]}

    def test_queue_rejection_does_not_orphan_a_successfully_uploaded_pdf(self):
        library = store.create_library(self.alice["id"], "Private")
        for i in range(10):
            store.create_task(self.alice["id"], "search", {"query": str(i)})
        response = self.client.post(f"/api/libraries/{library['id']}/documents", files={"file": ("paper.pdf", self.pdf(), "application/pdf")}, headers=self.csrf)
        self.assertEqual(response.status_code, 429)
        folder = jobs.workspace() / "uploads"
        self.assertEqual(list(folder.glob("*.pdf")), [])
        with connect() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM rm_tasks WHERE kind='upload'").fetchone()[0], 0)

    def test_persisted_upload_source_survives_worker_submission_failure(self):
        library = store.create_library(self.alice["id"], "Private")
        self.client.app.state.submit = Mock(side_effect=RuntimeError("worker is closing"))
        with self.assertRaises(RuntimeError):
            self.client.post(f"/api/libraries/{library['id']}/documents", files={"file": ("paper.pdf", self.pdf(), "application/pdf")}, headers=self.csrf)
        with connect() as conn:
            row = conn.execute("SELECT payload FROM rm_tasks WHERE kind='upload'").fetchone()
        from pathlib import Path
        self.assertTrue(Path(json.loads(row["payload"])["path"]).is_file())

    def test_concurrent_retry_claims_a_failed_task_only_once(self):
        task = store.create_task(self.alice["id"], "search", {"query": "offline"})
        store.update_task(task["id"], status="failed")
        original = store.task
        barrier, local = threading.Barrier(2), threading.local()
        def stale_read(task_id, user_id=None):
            row = original(task_id, user_id)
            if task_id == task["id"] and row["status"] == "failed" and not getattr(local, "seen", False):
                local.seen = True
                barrier.wait(timeout=5)
            return row
        submit = Mock()
        self.client.app.state.submit = submit
        def retry(_):
            return self.client.post(f"/api/tasks/{task['id']}/retry", json={}, headers=self.csrf)
        with patch.object(store, "task", side_effect=stale_read), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(retry, range(2)))
        self.assertEqual([r.status_code for r in results], [200, 200])
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(original(task["id"])["status"], "queued")

    def test_one_active_answer_per_session_but_other_sessions_can_run(self):
        first = store.create_session(self.alice["id"], "First")
        second = store.create_session(self.alice["id"], "Second")
        payload = self.scope(first["id"])
        self.assertEqual(self.post("/api/answers", payload).status_code, 200)
        duplicate = self.post("/api/answers", {**payload, "question": "Follow-up"})
        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("等待该轮完成", duplicate.json()["detail"])
        self.assertEqual(self.post("/api/answers", {**payload, "session_id": second["id"]}).status_code, 200)

    def test_retry_cannot_bypass_another_active_answer_in_the_same_session(self):
        session = store.create_session(self.alice["id"], "Read")
        payload = self.scope(session["id"])
        old = store.create_task(self.alice["id"], "answer", payload)
        store.update_task(old["id"], status="failed")
        new = store.create_task(self.alice["id"], "answer", {**payload, "question": "New question"})
        response = self.post(f"/api/tasks/{old['id']}/retry", {})
        self.assertEqual(response.status_code, 400)
        self.assertIn("等待该轮完成", response.json()["detail"])
        self.assertEqual(store.task(old["id"])["status"], "failed")
        self.assertEqual(store.task(new["id"])["status"], "queued")

    def test_research_scope_rejects_boolean_float_zero_and_negative_ids(self):
        for value in (True, 1.0, "1", 0, -1):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ResearchScope(group_id=value)
            with self.subTest(selected=value), self.assertRaises(ValidationError):
                ResearchScope(selected_doc_ids=[value])
        self.assertEqual(ResearchScope(group_id=1, selected_doc_ids=[]).selected_doc_ids, [])

    def test_invalid_scope_is_rejected_before_retrieval_and_model_calls(self):
        session = store.create_session(self.alice["id"], "Read")
        task = store.create_task(self.alice["id"], "answer", {"session_id": session["id"], "question": "Explain", "mode": "external"})
        with patch("refmind.rag.scientific_retrieval.retrieve_for_task") as retrieve, \
             patch("refmind.rag.graph.answer_from_evidence") as generate, patch.object(jobs.LOG, "exception"):
            self.client.app.state.runner._run(task["id"])
        self.assertEqual(store.task(task["id"])["status"], "failed")
        retrieve.assert_not_called()
        generate.assert_not_called()
        self.assertEqual(storage.list_messages(session["id"]), [])

    def test_persistent_citation_snapshot_keeps_paragraph_and_version(self):
        library = store.create_library(self.alice["id"], "Private")
        session = store.create_session(self.alice["id"], "Read", library["id"])
        doc_id = storage.create_document(library["id"], "paper.pdf", status="ready")
        evidence = Document(page_content="Second paragraph", metadata={"doc_id": doc_id, "group_id": library["id"], "page": 3,
                    "paragraph_index": 2, "version": "v3", "library_index": 7, "section_path": "Methods"})
        task = store.create_task(self.alice["id"], "answer", {"session_id": session["id"], "question": "Explain", "mode": "library", "group_id": library["id"], "selected_doc_ids": [doc_id]})
        # Patch the library path's shared seam; no live retrieval/model is used.
        with patch.object(settings, "research_multi_query_enabled", False), \
             patch("refmind.rag.research_pipeline.retrieve_for_task", return_value=[evidence]), \
             patch("refmind.rag.research_pipeline.rerank", return_value=[evidence]), \
             patch("refmind.rag.graph.answer_from_evidence", return_value={"answer": "Answer", "status": "answered", "documents": [evidence]}):
            self.client.app.state.runner._run(task["id"])
        result = store.task(task["id"])
        self.assertEqual(result["status"], "completed")
        stored = result["result"]["evidence"][0]["metadata"]
        self.assertEqual((stored["paragraph_index"], stored["version"], stored["library_index"]), (2, "v3", 7))
        messages = self.client.get(f"/api/sessions/{session['id']}/messages").json()
        self.assertEqual(messages[-1]["result"]["evidence"][0]["metadata"], stored)

    def test_concurrent_submissions_cannot_exceed_existing_ten_task_limit(self):
        barrier = threading.Barrier(12)
        def submit(i):
            barrier.wait(timeout=5)
            try:
                store.create_task(self.alice["id"], "search", {"query": str(i)})
                return True
            except store.TaskQueueFull:
                return False
        with ThreadPoolExecutor(max_workers=12) as pool:
            accepted = list(pool.map(submit, range(12)))
        self.assertEqual(sum(accepted), 10)
        with connect() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM rm_tasks WHERE owner_id=? AND status='queued'", (self.alice["id"],)).fetchone()[0], 10)

    def test_atomic_capacity_failure_keeps_http_429(self):
        with patch.object(store, "create_task", side_effect=store.TaskQueueFull("待处理任务较多")):
            response = self.post("/api/academic-searches", {"query": "offline"})
        self.assertEqual(response.status_code, 429)


if __name__ == "__main__":
    unittest.main()
