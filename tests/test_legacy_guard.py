import ast
import hashlib
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from refmind.storage.legacy_guard import ensure_legacy_access, has_team_accounts
from refmind.storage.runtime_lock import acquire_runtime_lock, release_runtime_lock


class LegacyGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "data.db"

    def tearDown(self):
        release_runtime_lock(self.db)
        self.tmp.cleanup()

    def team_database(self, *, account=True):
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("CREATE TABLE rm_users(id INTEGER PRIMARY KEY, disabled INTEGER)")
            if account:
                conn.execute("INSERT INTO rm_users VALUES(1,1)")
            conn.commit()

    def initializer(self, address):
        source = Path(__file__).resolve().parents[1] / "app.py"
        function = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body if isinstance(node, ast.FunctionDef) and node.name == "_initialize_runtime")
        function.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        storage = Mock()
        namespace = {"settings": SimpleNamespace(database_path=self.db), "storage": storage,
                     "st": SimpleNamespace(get_option=lambda key: address),
                     "recover_incomplete_ingestions": Mock(return_value={"recovered": [], "failed": []}),
                     "backfill_document_titles": Mock(return_value={"updated": [], "failed": []})}
        exec(compile(module, str(source), "exec"), namespace)
        return namespace["_initialize_runtime"], namespace

    def test_missing_or_legacy_only_database_keeps_single_user_compatibility(self):
        self.assertFalse(has_team_accounts(self.db))
        ensure_legacy_access(self.db, server_address=None, admin_mode=False)
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("CREATE TABLE groups(id INTEGER)")
            conn.commit()
        self.assertFalse(has_team_accounts(self.db))
        ensure_legacy_access(self.db, server_address="0.0.0.0", admin_mode=False)

    def test_empty_team_schema_without_accounts_is_not_a_team_database(self):
        self.team_database(account=False)
        self.assertFalse(has_team_accounts(self.db))

    def test_team_database_is_blocked_by_default_even_on_loopback(self):
        self.team_database()
        with self.assertRaises(RuntimeError):
            ensure_legacy_access(self.db, server_address="127.0.0.1", admin_mode=False)

    def test_admin_override_still_requires_explicit_local_bind(self):
        self.team_database()
        for address in (None, "", "0.0.0.0", "::", "192.168.1.2"):
            with self.subTest(address=address), self.assertRaises(RuntimeError):
                ensure_legacy_access(self.db, server_address=address, admin_mode=True)
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        for address in ("127.0.0.1", "localhost"):
            ensure_legacy_access(self.db, server_address=address, admin_mode=True)
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), before)

    def test_failed_guard_runs_before_recovery_and_releases_writer_lock(self):
        self.team_database()
        initialize, namespace = self.initializer("0.0.0.0")
        with patch.dict(os.environ, {"REFMIND_LEGACY_ADMIN_MODE": "false"}), self.assertRaises(RuntimeError):
            initialize()
        namespace["storage"].init_db.assert_not_called()
        namespace["recover_incomplete_ingestions"].assert_not_called()
        acquire_runtime_lock(self.db, "api")

    def test_allowed_local_admin_fallback_keeps_the_writer_lock(self):
        self.team_database()
        initialize, namespace = self.initializer("127.0.0.1")
        with patch.dict(os.environ, {"REFMIND_LEGACY_ADMIN_MODE": "true"}):
            initialize()
        namespace["storage"].init_db.assert_called_once()
        with self.assertRaises(RuntimeError):
            acquire_runtime_lock(self.db, "api")


if __name__ == "__main__":
    unittest.main()
