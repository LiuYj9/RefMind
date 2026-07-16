"""入库失败补偿与文件原子性的离线回归测试。"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch


def _load_ingestion_without_optional_dependencies():
    """用轻量替身加载编排模块，测试不依赖 LangChain/Chroma。"""
    llm_stub = types.ModuleType("refmind.llm")
    llm_stub.generate_summary = lambda _documents: "summary"

    rag_stub = types.ModuleType("refmind.rag")
    rag_stub.get_vectorstore = Mock()
    rag_stub.ingest_documents = Mock()
    rag_stub.invalidate_retriever = Mock()
    rag_stub.parsed_to_documents = Mock()
    rag_stub.prepare_ingestion_batch = Mock()

    module_name = "refmind.services._ingestion_atomicity_test"
    path = Path(__file__).parents[1] / "refmind" / "services" / "ingestion.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"refmind.llm": llm_stub, "refmind.rag": rag_stub, module_name: module},
    ):
        spec.loader.exec_module(module)
    return module


ingestion = _load_ingestion_without_optional_dependencies()


class _Settings:
    def __init__(self, root: Path) -> None:
        self.upload_dir = root / "uploads"
        self.parsed_dir = root / "parsed"
        self.pdf_max_parallel_documents = 2
        self.document_summary_max_workers = 2

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)


class IngestionAtomicityTests(unittest.TestCase):
    def test_final_progress_failure_does_not_rollback_committed_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pdf"
            source.write_bytes(b"new pdf")
            row = object()
            storage = types.SimpleNamespace(
                create_document=Mock(return_value=6),
                update_document=Mock(),
                get_document=Mock(return_value=row),
                delete_document=Mock(),
            )
            parsed = {
                "source": "paper.pdf",
                "parser": "test",
                "markdown": "body",
                "pages": [{"page": 1, "text": "body"}],
                "tables": [],
            }

            def progress(value: float, _message: str) -> None:
                if value == 1.0:
                    raise RuntimeError("UI disappeared")

            with (
                patch.object(ingestion, "settings", _Settings(root)),
                patch.object(ingestion, "storage", storage),
                patch.object(ingestion, "parse_pdf", return_value=parsed),
                patch.object(
                    ingestion,
                    "parsed_to_documents",
                    return_value=[ingestion.Document(page_content="body")],
                ),
                patch.object(ingestion, "ingest_documents", return_value=1),
                patch.object(ingestion, "invalidate_retriever"),
            ):
                result = ingestion.ingest_pdf(
                    1,
                    source,
                    "paper.pdf",
                    progress=progress,
                    make_summary=False,
                )

            self.assertIs(result, row)
            storage.delete_document.assert_not_called()
            self.assertTrue((root / "uploads" / "1_6_paper.pdf").exists())
            self.assertTrue((root / "parsed" / "1_6_paper.json").exists())
            self.assertEqual(
                storage.update_document.call_args_list[-1].kwargs["status"], "ready"
            )

    def test_parse_failure_removes_new_file_and_database_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pdf"
            source.write_bytes(b"new pdf")
            storage = types.SimpleNamespace(
                create_document=Mock(return_value=7),
                update_document=Mock(),
                delete_document=Mock(),
            )
            with (
                patch.object(ingestion, "settings", _Settings(root)),
                patch.object(ingestion, "storage", storage),
                patch.object(ingestion, "parse_pdf", side_effect=RuntimeError("bad pdf")),
                patch.object(ingestion, "invalidate_retriever"),
            ):
                with self.assertRaisesRegex(RuntimeError, "bad pdf"):
                    ingestion.ingest_pdf(1, source, "paper.pdf", make_summary=False)

            self.assertFalse((root / "uploads" / "1_7_paper.pdf").exists())
            storage.delete_document.assert_called_once_with(7)

    def test_partial_vector_failure_is_deleted_by_doc_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pdf"
            source.write_bytes(b"new pdf")
            vectorstore = types.SimpleNamespace(delete=Mock())
            storage = types.SimpleNamespace(
                create_document=Mock(return_value=8),
                update_document=Mock(),
                delete_document=Mock(),
            )
            parsed = {
                "source": "paper.pdf",
                "parser": "test",
                "markdown": "body",
                "pages": [{"page": 1, "text": "body"}],
                "tables": [],
            }
            with (
                patch.object(ingestion, "settings", _Settings(root)),
                patch.object(ingestion, "storage", storage),
                patch.object(ingestion, "parse_pdf", return_value=parsed),
                patch.object(
                    ingestion,
                    "parsed_to_documents",
                    return_value=[ingestion.Document(page_content="body")],
                ),
                patch.object(
                    ingestion,
                    "ingest_documents",
                    side_effect=RuntimeError("embedding failed"),
                ),
                patch.object(ingestion, "get_vectorstore", return_value=vectorstore),
                patch.object(ingestion, "invalidate_retriever"),
            ):
                with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                    ingestion.ingest_pdf(2, source, "paper.pdf", make_summary=False)

            vectorstore.delete.assert_called_once_with(where={"doc_id": 8})
            storage.delete_document.assert_called_once_with(8)
            self.assertFalse((root / "uploads" / "2_8_paper.pdf").exists())
            self.assertEqual(list((root / "parsed").glob("*.json")), [])

    def test_existing_destination_is_restored_when_reingestion_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _Settings(root)
            settings.ensure_dirs()
            destination = settings.upload_dir / "3_9_paper.pdf"
            destination.write_bytes(b"old pdf")
            source = root / "source.pdf"
            source.write_bytes(b"new pdf")
            storage = types.SimpleNamespace(
                create_document=Mock(return_value=9),
                update_document=Mock(),
                delete_document=Mock(),
            )
            with (
                patch.object(ingestion, "settings", settings),
                patch.object(ingestion, "storage", storage),
                patch.object(ingestion, "parse_pdf", side_effect=RuntimeError("bad pdf")),
                patch.object(ingestion, "invalidate_retriever"),
            ):
                with self.assertRaisesRegex(RuntimeError, "bad pdf"):
                    ingestion.ingest_pdf(3, source, "paper.pdf", make_summary=False)

            self.assertEqual(destination.read_bytes(), b"old pdf")
            self.assertEqual(list(settings.upload_dir.glob("*.backup")), [])

    def test_ingest_hook_cannot_remove_governance_metadata(self) -> None:
        stripped = ingestion.Document(page_content="plugin content", metadata={})
        manager = types.SimpleNamespace(
            run_hook=Mock(return_value=types.SimpleNamespace(value=[stripped]))
        )
        with patch.object(ingestion, "get_plugin_manager", return_value=manager):
            documents = ingestion._run_before_ingest_hook(
                [stripped],
                group_id=4,
                doc_id=12,
                filename="paper.pdf",
                version="v1",
            )

        metadata = documents[0].metadata
        self.assertEqual(metadata["group_id"], 4)
        self.assertEqual(metadata["doc_id"], 12)
        self.assertEqual(metadata["permission"], "group:4")
        self.assertEqual(metadata["filename"], "paper.pdf")
        self.assertTrue(metadata["chunk_id"])

    def test_remove_preserves_record_when_vector_cleanup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "paper.pdf"
            parsed = Path(tmp) / "paper.json"
            original.write_bytes(b"pdf")
            parsed.write_text("{}", encoding="utf-8")
            row = types.SimpleNamespace(
                id=5,
                group_id=2,
                original_path=str(original),
                parsed_json_path=str(parsed),
            )
            storage = types.SimpleNamespace(
                get_document=Mock(return_value=row),
                update_document=Mock(),
                delete_document=Mock(),
            )
            with (
                patch.object(ingestion, "storage", storage),
                patch.object(
                    ingestion,
                    "_delete_chunks_by_doc_id",
                    side_effect=RuntimeError("chroma busy"),
                ),
                patch.object(ingestion, "invalidate_retriever"),
            ):
                with self.assertRaisesRegex(RuntimeError, "记录已保留"):
                    ingestion.remove_document(5)

            self.assertEqual(
                storage.update_document.call_args_list,
                [
                    call(5, status="deleting"),
                    call(5, status="cleanup_failed"),
                ],
            )
            storage.delete_document.assert_not_called()
            self.assertTrue(original.exists())
            self.assertTrue(parsed.exists())

    def test_startup_recovery_retries_every_non_ready_document(self) -> None:
        group = types.SimpleNamespace(id=3)
        ready = types.SimpleNamespace(id=1, status="ready")
        parsing = types.SimpleNamespace(id=2, status="parsing")
        cleanup_failed = types.SimpleNamespace(id=3, status="cleanup_failed")
        storage = types.SimpleNamespace(
            list_groups=Mock(return_value=[group]),
            list_documents=Mock(return_value=[ready, parsing, cleanup_failed]),
        )
        with (
            patch.object(ingestion, "storage", storage),
            patch.object(ingestion, "remove_document") as remove,
        ):
            report = ingestion.recover_incomplete_ingestions()

        self.assertEqual(remove.call_args_list, [call(2), call(3)])
        self.assertEqual(report, {"recovered": [2, 3], "failed": []})


if __name__ == "__main__":
    unittest.main()
