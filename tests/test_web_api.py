import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from refmind.config import settings
from refmind.web import store
from refmind.web.api import create_app, _attempts
from refmind.storage.connection import connect


class WebApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.stack = ExitStack()
        base = Path(self.temp.name)
        for key, path in {"database_path": base / "test.db", "upload_dir": base / "uploads", "parsed_dir": base / "parsed", "docstore_dir": base / "images", "chroma_persist_dir": base / "chroma"}.items():
            self.stack.enter_context(patch.object(settings, key, path))
        _attempts.clear()
        self.client = self.stack.enter_context(TestClient(create_app(start_workers=False)))
        self.alice = store.create_user("alice", "test-password-alice", admin=True)
        self.bob = store.create_user("bob", "test-password-bob")
        self.login("alice")

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def login(self, username):
        response = self.client.post("/api/auth/login", json={"username": username, "password": "test-password-" + username})
        self.assertEqual(response.status_code, 200, response.text)
        self.csrf = {"X-CSRF-Token": response.json()["csrf"]}

    def post(self, url, payload):
        return self.client.post(url, json=payload, headers=self.csrf)

    def test_private_library_and_session_do_not_leak_to_member(self):
        library = self.post("/api/libraries", {"title": "私人论文"}).json()
        shared = self.post("/api/libraries", {"title": "课题组", "shared": True}).json()
        session = self.post("/api/sessions", {"group_id": shared["id"]}).json()
        self.assertEqual(self.post(f"/api/libraries/{shared['id']}/members", {"username": "bob", "role": "viewer"}).status_code, 200)
        self.login("bob")
        self.assertEqual([x["id"] for x in self.client.get("/api/libraries").json()], [shared["id"]])
        self.assertEqual(self.client.get(f"/api/libraries/{library['id']}/documents").status_code, 403)
        self.assertEqual(self.client.get(f"/api/sessions/{session['id']}/messages").status_code, 403)
        self.assertEqual(self.post(f"/api/libraries/{shared['id']}/members", {"username": "alice", "role": "editor"}).status_code, 403)

    def test_csrf_and_login_boundary(self):
        self.assertEqual(self.client.post("/api/libraries", json={"title": "bad"}).status_code, 403)
        self.assertEqual(self.client.post("/api/libraries", json={"title": "bad"}, headers={**self.csrf, "Origin": "https://untrusted.invalid"}).status_code, 403)
        self.assertEqual(self.client.post("/api/auth/logout", headers=self.csrf).status_code, 200)
        self.assertEqual(self.client.get("/api/libraries").status_code, 401)

    def test_no_implicit_search_and_owner_scoped_tasks(self):
        with patch("refmind.web.jobs.JobRunner.submit") as submit:
            session = self.post("/api/sessions", {}).json()
            result = self.post("/api/academic-searches", {"query": "thermal conductivity"})
            self.assertEqual(result.status_code, 200)
            tid = result.json()["id"]
            self.assertNotIn("payload", result.json())
            submit.assert_not_called()
        self.login("bob")
        self.assertEqual(self.client.get(f"/api/tasks/{tid}").status_code, 403)
        self.assertEqual(self.post("/api/answers", {"session_id": session["id"], "question": "what?"}).status_code, 403)

    def test_reading_only_accepts_owned_search_candidate(self):
        session = self.post("/api/sessions", {}).json()
        task = store.create_task(self.alice["id"], "search", {})
        store.update_task(task["id"], status="completed", result={"papers": [{"paper_id": "doi:test", "title": "A", "abstract": ""}]})
        response = self.post("/api/reading-tasks", {"search_id": task["id"], "paper_id": "doi:other", "session_id": session["id"]})
        self.assertEqual(response.status_code, 400)
        response = self.post("/api/reading-tasks", {"search_id": task["id"], "paper_id": "doi:test", "session_id": session["id"]})
        self.assertEqual(response.status_code, 200)

    def test_expired_asset_and_snapshot_remain_distinct(self):
        import time
        session = self.post("/api/sessions", {}).json()
        with connect() as conn:
            conn.execute("INSERT INTO rm_assets(id,owner_id,session_id,paper,file_info,status,expires_at,created_at) VALUES(?,?,?,?,?,'ready',?,?)", ("expired", self.alice["id"], session["id"], json.dumps({"title": "old"}), json.dumps({"path": "not-opened.pdf", "sha256": "abc"}), time.time()-1, time.time()-10))
        self.assertEqual(self.client.get("/api/assets/expired/pdf").status_code, 410)
        self.assertEqual(self.client.get(f"/api/sessions/{session['id']}/assets").json()[0]["status"], "expired")


if __name__ == "__main__":
    unittest.main()
