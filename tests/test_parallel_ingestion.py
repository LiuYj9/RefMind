"""多 PDF 有界并行、进度线程边界与失败隔离测试。"""

from __future__ import annotations

import threading
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from refmind.services import ingestion


class ParallelIngestionTests(unittest.TestCase):
    def test_batch_respects_worker_limit_and_preserves_input_order(self) -> None:
        lock = threading.Lock()
        active = 0
        max_active = 0
        caller_thread = threading.get_ident()
        progress_threads: list[int] = []

        def fake_ingest(
            _group_id,
            _path,
            filename,
            progress,
            make_summary=True,
            stage_timings=None,
            _parse_semaphore=None,
        ):
            nonlocal active, max_active
            with _parse_semaphore or nullcontext():
                with ingestion._pdf_task_slot():
                    with lock:
                        active += 1
                        max_active = max(max_active, active)
                    progress(0.25, "解析")
                    time.sleep(0.04)
                    progress(0.75, "入库")
                    with lock:
                        active -= 1
            return SimpleNamespace(id=int(filename.removeprefix("p").removesuffix(".pdf")))

        tasks = [
            ingestion.PdfIngestionTask(Path(f"p{index}.pdf"), f"p{index}.pdf")
            for index in range(5)
        ]
        with (
            patch.object(ingestion.settings, "pdf_max_parallel_documents", 2),
            patch.object(ingestion, "ingest_pdf", side_effect=fake_ingest),
        ):
            results = ingestion.ingest_pdfs_parallel(
                7,
                tasks,
                max_workers=2,
                progress=lambda *_args: progress_threads.append(threading.get_ident()),
            )

        self.assertEqual(max_active, 2)
        self.assertEqual([result.filename for result in results], [task.filename for task in tasks])
        self.assertTrue(all(result.succeeded for result in results))
        self.assertEqual(set(progress_threads), {caller_thread})

    def test_one_failure_does_not_cancel_others_or_delete_its_old_version(self) -> None:
        tasks = [
            ingestion.PdfIngestionTask(Path("good.pdf"), "good.pdf", 10),
            ingestion.PdfIngestionTask(Path("bad.pdf"), "bad.pdf", 11),
        ]

        def fake_ingest(
            _group_id,
            _path,
            filename,
            progress,
            make_summary=True,
            stage_timings=None,
            _parse_semaphore=None,
        ):
            if filename == "bad.pdf":
                raise RuntimeError("broken input")
            return SimpleNamespace(id=20)

        with (
            patch.object(ingestion, "ingest_pdf", side_effect=fake_ingest),
            patch.object(ingestion, "remove_document") as remove,
        ):
            results = ingestion.ingest_pdfs_parallel(3, tasks, max_workers=2)

        self.assertTrue(results[0].succeeded)
        self.assertFalse(results[1].succeeded)
        self.assertIn("broken input", results[1].error)
        self.assertEqual(remove.call_args_list, [call(10)])

    def test_same_group_vector_mutations_are_serialized(self) -> None:
        lock = threading.Lock()
        active = 0
        max_active = 0

        class _VectorStore:
            def delete(self, **_kwargs):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.03)
                with lock:
                    active -= 1

        store = _VectorStore()
        with patch.object(ingestion, "get_vectorstore", return_value=store):
            threads = [
                threading.Thread(
                    target=ingestion._delete_chunks_by_doc_id,
                    args=(9, index),
                )
                for index in range(4)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(max_active, 1)

    def test_parallel_limit_is_global_across_two_batch_callers(self) -> None:
        lock = threading.Lock()
        start = threading.Barrier(2)
        active = 0
        max_active = 0

        def fake_ingest(
            _group_id,
            _path,
            filename,
            progress,
            make_summary=True,
            stage_timings=None,
            _parse_semaphore=None,
        ):
            nonlocal active, max_active
            with _parse_semaphore or nullcontext():
                with ingestion._pdf_task_slot():
                    with lock:
                        active += 1
                        max_active = max(max_active, active)
                    time.sleep(0.04)
                    with lock:
                        active -= 1
            return SimpleNamespace(id=filename)

        def run_batch(prefix: str) -> None:
            tasks = [
                ingestion.PdfIngestionTask(Path(f"{prefix}{index}.pdf"), f"{prefix}{index}.pdf")
                for index in range(2)
            ]
            start.wait(timeout=2)
            ingestion.ingest_pdfs_parallel(1, tasks, max_workers=2)

        with (
            patch.object(ingestion.settings, "pdf_max_parallel_documents", 2),
            patch.object(ingestion, "ingest_pdf", side_effect=fake_ingest),
        ):
            callers = [
                threading.Thread(target=run_batch, args=(prefix,))
                for prefix in ("a", "b")
            ]
            for caller in callers:
                caller.start()
            for caller in callers:
                caller.join(timeout=3)

        self.assertTrue(all(not caller.is_alive() for caller in callers))
        self.assertEqual(max_active, 2)

    def test_duplicate_filenames_are_rejected_before_workers_start(self) -> None:
        tasks = [
            ingestion.PdfIngestionTask(Path("a.pdf"), "same.pdf"),
            ingestion.PdfIngestionTask(Path("b.pdf"), "same.pdf"),
        ]
        with (
            patch.object(ingestion, "ingest_pdf") as ingest,
            self.assertRaisesRegex(ValueError, "重名 PDF"),
        ):
            ingestion.ingest_pdfs_parallel(1, tasks, max_workers=2)
        ingest.assert_not_called()

    def test_pipeline_keeps_parse_fed_and_honors_batch_local_limit(self) -> None:
        """后处理中的文档不应占满 worker，导致后续 PDF 无法进入解析槽。"""
        lock = threading.Lock()
        fifth_parse_started = threading.Event()
        parse_order: list[str] = []
        postprocess_observations: list[bool] = []
        active = 0
        max_active = 0

        def fake_ingest(
            _group_id,
            _path,
            filename,
            progress,
            make_summary=True,
            stage_timings=None,
            _parse_semaphore=None,
        ):
            nonlocal active, max_active
            with _parse_semaphore or nullcontext():
                with ingestion._pdf_task_slot():
                    with lock:
                        active += 1
                        max_active = max(max_active, active)
                        parse_order.append(filename)
                        ordinal = len(parse_order)
                    time.sleep(0.01)
                    with lock:
                        active -= 1
            if ordinal == 5:
                fifth_parse_started.set()
            else:
                postprocess_observations.append(
                    fifth_parse_started.wait(timeout=1)
                )
            if stage_timings is not None:
                stage_timings.update({"parse": 0.01, "total": 0.02})
            return SimpleNamespace(id=filename)

        tasks = [
            ingestion.PdfIngestionTask(Path(f"p{index}.pdf"), f"p{index}.pdf")
            for index in range(5)
        ]
        with (
            patch.object(ingestion.settings, "pdf_max_parallel_documents", 8),
            patch.object(ingestion, "ingest_pdf", side_effect=fake_ingest),
        ):
            results = ingestion.ingest_pdfs_parallel(1, tasks, max_workers=2)

        self.assertEqual(max_active, 2)
        self.assertEqual(len(parse_order), 5)
        self.assertEqual(postprocess_observations, [True] * 4)
        self.assertTrue(all(result.succeeded for result in results))
        self.assertTrue(all(result.stage_seconds["parse"] == 0.01 for result in results))


if __name__ == "__main__":
    unittest.main()
