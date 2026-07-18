"""PDF 入库阶段流水线、共享并发闸门与预计算向量提交测试。"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from langchain_core.documents import Document

from refmind.rag import document_processor
from refmind.services import ingestion


class EmbeddingStageTests(unittest.TestCase):
    def test_batches_embed_concurrently_but_keep_document_order(self) -> None:
        lock = threading.Lock()
        first_batches = threading.Barrier(2)
        active = 0
        max_active = 0

        class _EmbeddingModel:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    if texts[0] in {"text-0", "text-2"}:
                        first_batches.wait(timeout=2)
                    time.sleep(0.01)
                    return [[float(text.removeprefix("text-"))] for text in texts]
                finally:
                    with lock:
                        active -= 1

        documents = [
            Document(page_content=f"text-{index}", metadata={"chunk_id": "same"})
            for index in range(5)
        ]
        with (
            patch.object(document_processor.settings, "embedding_batch_size", 2),
            patch.object(
                document_processor.settings, "embedding_max_parallel_batches", 2
            ),
            patch.object(
                document_processor, "get_embedding_model", return_value=_EmbeddingModel()
            ),
        ):
            prepared = document_processor.prepare_ingestion_batch(documents)

        self.assertEqual(max_active, 2)
        self.assertEqual(prepared.embeddings, [[0.0], [1.0], [2.0], [3.0], [4.0]])
        self.assertEqual(len(set(prepared.ids)), len(documents))
        self.assertNotIn("same", prepared.ids)

    def test_embedding_limit_is_global_across_document_calls(self) -> None:
        lock = threading.Lock()
        callers_ready = threading.Barrier(2)
        active = 0
        max_active = 0
        failures: list[Exception] = []

        class _EmbeddingModel:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    time.sleep(0.02)
                    return [[float(len(text))] for text in texts]
                finally:
                    with lock:
                        active -= 1

        def prepare(prefix: str) -> None:
            try:
                callers_ready.wait(timeout=2)
                document_processor.prepare_ingestion_batch(
                    [Document(page_content=f"{prefix}-{index}") for index in range(3)]
                )
            except Exception as exc:  # pragma: no cover - 由主线程统一报告
                failures.append(exc)

        with (
            patch.object(document_processor.settings, "embedding_batch_size", 1),
            patch.object(
                document_processor.settings, "embedding_max_parallel_batches", 2
            ),
            patch.object(
                document_processor, "get_embedding_model", return_value=_EmbeddingModel()
            ),
        ):
            callers = [
                threading.Thread(target=prepare, args=(prefix,))
                for prefix in ("a", "b")
            ]
            for caller in callers:
                caller.start()
            for caller in callers:
                caller.join(timeout=3)

        self.assertTrue(all(not caller.is_alive() for caller in callers))
        self.assertEqual(failures, [])
        self.assertEqual(max_active, 2)

    def test_precomputed_vectors_use_public_batched_upsert_without_reembedding(self) -> None:
        documents = [
            Document(page_content="a", metadata={"doc_id": 1}),
            Document(page_content="b", metadata={"doc_id": 1}),
        ]
        prepared = document_processor.PreparedVectorBatch(
            ids=["id-a", "id-b"],
            texts=["a", "b"],
            metadatas=[{"doc_id": 1}, {"doc_id": 1}],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        collection = Mock()
        client = Mock()
        client.get_or_create_collection.return_value = collection
        client.get_max_batch_size.return_value = 1

        with (
            patch("chromadb.PersistentClient", return_value=client) as client_factory,
            patch.object(
                document_processor.settings,
                "group_chroma_dir",
                return_value=Path("D:/vectors/group_3"),
            ),
            patch.object(document_processor, "prepare_ingestion_batch") as prepare,
        ):
            count = document_processor.ingest_documents(
                3, documents, prepared=prepared
            )

        self.assertEqual(count, 2)
        prepare.assert_not_called()
        client_factory.assert_called_once_with(path="D:\\vectors\\group_3")
        client.get_or_create_collection.assert_called_once_with(
            name="group_3", embedding_function=None
        )
        self.assertEqual(
            collection.upsert.call_args_list,
            [
                call(
                    ids=["id-a"],
                    documents=["a"],
                    metadatas=[{"doc_id": 1}],
                    embeddings=[[1.0, 0.0]],
                ),
                call(
                    ids=["id-b"],
                    documents=["b"],
                    metadatas=[{"doc_id": 1}],
                    embeddings=[[0.0, 1.0]],
                ),
            ],
        )

    def test_precomputed_vector_shape_is_validated_before_chroma(self) -> None:
        documents = [Document(page_content="a"), Document(page_content="b")]
        malformed = document_processor.PreparedVectorBatch(
            ids=["only-one"],
            texts=["a", "b"],
            metadatas=[{}, {}],
            embeddings=[[1.0], [2.0]],
        )
        with (
            patch("chromadb.PersistentClient") as client,
            self.assertRaisesRegex(ValueError, "数量不一致"),
        ):
            document_processor.ingest_documents(1, documents, prepared=malformed)
        client.assert_not_called()


class ImageSummaryStageTests(unittest.TestCase):
    def test_images_run_concurrently_deduplicate_and_isolate_one_failure(self) -> None:
        parsed = {
            "blocks": [
                {"block_id": f"f{index}", "type": "figure", "text": f"图注{index}"}
                for index in range(1, 5)
            ]
        }
        image_paths = ["a.png", "a.png", "b.png", "c.png"]
        assets = [
            {
                "block_id": f"f{index}",
                "image_id": f"i{index}",
                "image_path": image_path,
                "image_mime_type": "image/png",
                "page": 1,
                "page_end": 1,
                "bbox": [],
            }
            for index, image_path in enumerate(image_paths, start=1)
        ]
        lock = threading.Lock()
        first_requests = threading.Barrier(2)
        active = 0
        max_active = 0
        calls: list[str] = []

        def fake_summary(path: str, *, caption: str = "") -> str:
            nonlocal active, max_active
            with lock:
                calls.append(path)
                active += 1
                max_active = max(max_active, active)
            try:
                if path in {"a.png", "b.png"}:
                    first_requests.wait(timeout=2)
                time.sleep(0.01)
                if path == "b.png":
                    raise RuntimeError("one bad image")
                return f"summary-{Path(path).stem}"
            finally:
                with lock:
                    active -= 1

        with (
            patch.object(ingestion, "extract_pdf_figures", return_value=assets),
            patch.object(ingestion.settings, "image_summary_enabled", True),
            patch.object(ingestion.settings, "dashscope_api_key", "test-key"),
            patch.object(ingestion.settings, "image_summary_max_workers", 2),
            patch.object(ingestion, "summarize_image", side_effect=fake_summary),
        ):
            count = ingestion._enrich_figure_blocks(
                parsed, Path("paper.pdf"), doc_id=1
            )

        self.assertEqual(count, 4)
        self.assertEqual(max_active, 2)
        self.assertEqual(Counter(calls), Counter({"a.png": 1, "b.png": 1, "c.png": 1}))
        blocks = parsed["blocks"]
        self.assertIn("summary-a", blocks[0]["text"])
        self.assertIn("图注1", blocks[0]["text"])
        self.assertIn("summary-a", blocks[1]["text"])
        self.assertIn("图注2", blocks[1]["text"])
        self.assertNotIn("Visual summary:", blocks[2]["text"])
        self.assertEqual(blocks[2]["image_path"], "b.png")
        self.assertIn("summary-c", blocks[3]["text"])


class PipelineStageTests(unittest.TestCase):
    def test_document_summary_overlaps_embedding_and_failure_is_optional(self) -> None:
        class _Settings:
            def __init__(self, root: Path) -> None:
                self.upload_dir = root / "uploads"
                self.parsed_dir = root / "parsed"
                self.docstore_dir = root / "docstore"
                self.pdf_max_parallel_documents = 1
                self.document_summary_max_workers = 2

            def ensure_dirs(self) -> None:
                self.upload_dir.mkdir(parents=True, exist_ok=True)
                self.parsed_dir.mkdir(parents=True, exist_ok=True)
                self.docstore_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pdf"
            source.write_bytes(b"pdf")
            settings = _Settings(root)
            row = SimpleNamespace(id=41, group_id=7)
            storage = SimpleNamespace(
                create_document=Mock(return_value=41),
                update_document=Mock(),
                get_document=Mock(return_value=row),
                delete_document=Mock(),
            )
            documents = [Document(page_content="body", metadata={"doc_id": 41})]
            prepared = object()
            overlap = threading.Barrier(2)

            def prepare_vectors(_documents):
                overlap.wait(timeout=2)
                return prepared

            def fail_summary(_documents):
                overlap.wait(timeout=2)
                raise RuntimeError("summary unavailable")

            manager = SimpleNamespace(
                run_hook=lambda _hook, value, **_kwargs: SimpleNamespace(value=value)
            )
            timings: dict[str, float] = {}
            with (
                patch.object(ingestion, "settings", settings),
                patch.object(ingestion, "storage", storage),
                patch.object(ingestion, "_run_parse_hooks", return_value={"blocks": []}),
                patch.object(ingestion, "_enrich_figure_blocks", return_value=0),
                patch.object(ingestion, "save_parsed"),
                patch.object(ingestion, "parsed_to_documents", return_value=documents),
                patch.object(
                    ingestion, "_run_before_ingest_hook", return_value=documents
                ),
                patch.object(
                    ingestion,
                    "prepare_ingestion_batch",
                    side_effect=prepare_vectors,
                ),
                patch.object(ingestion, "generate_summary", side_effect=fail_summary),
                patch.object(ingestion, "ingest_documents", return_value=1) as commit,
                patch.object(ingestion, "invalidate_retriever"),
                patch.object(ingestion, "get_plugin_manager", return_value=manager),
            ):
                result = ingestion.ingest_pdf(
                    7,
                    source,
                    "paper.pdf",
                    make_summary=True,
                    stage_timings=timings,
                )

        self.assertIs(result, row)
        self.assertIs(commit.call_args.kwargs["prepared"], prepared)
        ready_updates = [
            item.kwargs
            for item in storage.update_document.call_args_list
            if item.kwargs.get("status") == "ready"
        ]
        self.assertEqual(len(ready_updates), 1)
        self.assertIn("summary unavailable", ready_updates[0]["summary"])
        expected_stages = {
            "prepare",
            "parse_wait",
            "parse",
            "image_enrichment",
            "save_parsed",
            "chunking",
            "embedding",
            "document_summary_wait",
            "document_summary",
            "vector_commit_wait",
            "vector_commit",
            "total",
        }
        self.assertTrue(expected_stages.issubset(timings))
        self.assertGreaterEqual(timings["total"], max(timings.values()))

    def test_group_deletion_waits_for_active_document_operation(self) -> None:
        operation_started = threading.Event()
        release_operation = threading.Event()
        group_deleted = threading.Event()

        def active_operation() -> None:
            with ingestion._group_operation_slot(9):
                operation_started.set()
                release_operation.wait(timeout=2)

        with tempfile.TemporaryDirectory() as tmp:
            storage = SimpleNamespace(
                get_group=Mock(return_value=SimpleNamespace(id=9)),
                list_documents=Mock(return_value=[]),
                delete_group=Mock(side_effect=lambda _group_id: group_deleted.set()),
            )
            settings = SimpleNamespace(chroma_persist_dir=Path(tmp) / "chroma")
            with (
                patch.object(ingestion, "storage", storage),
                patch.object(ingestion, "settings", settings),
                patch.object(ingestion, "invalidate_retriever"),
            ):
                worker = threading.Thread(target=active_operation)
                worker.start()
                self.assertTrue(operation_started.wait(timeout=1))

                deleter = threading.Thread(target=ingestion.remove_group, args=(9,))
                deleter.start()
                self.assertFalse(group_deleted.wait(timeout=0.05))
                release_operation.set()
                worker.join(timeout=2)
                deleter.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertFalse(deleter.is_alive())
        self.assertTrue(group_deleted.is_set())


if __name__ == "__main__":
    unittest.main()
