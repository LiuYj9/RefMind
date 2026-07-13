"""PDF 解析可靠性的离线回归测试。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from refmind.parsing import pdf_parser


class MinerUParsingTests(unittest.TestCase):
    def test_content_list_is_normalized_to_layout_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = [
                {"type": "title", "text": "1 Introduction", "page_idx": 0, "bbox": [1, 2, 3, 4]},
                {"type": "text", "text": "Body", "page_idx": 0},
                {"type": "interline_equation", "text": "E=mc^2", "page_idx": 1},
            ]
            (root / "paper_content_list.json").write_text(json.dumps(payload), encoding="utf-8")
            blocks = pdf_parser._collect_mineru_layout_blocks(root, "paper")

        self.assertEqual([b["type"] for b in blocks], ["heading", "paragraph", "equation"])
        self.assertEqual(blocks[0]["bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(blocks[2]["page"], 2)

    def test_collect_output_prefers_current_pdf_over_larger_stale_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "paper" / "auto"
            target_dir.mkdir(parents=True)
            (target_dir / "paper.md").write_text("当前论文正文", encoding="utf-8")
            (target_dir / "paper_content_list.json").write_text(
                json.dumps({"items": [{"page_no": 2, "text": "第二页"}]}),
                encoding="utf-8",
            )

            stale_dir = root / "stale"
            stale_dir.mkdir()
            (stale_dir / "old.md").write_text("旧文档" * 1000, encoding="utf-8")
            (stale_dir / "old_content_list.json").write_text(
                json.dumps([{"page_idx": 0, "text": "旧页面"}]),
                encoding="utf-8",
            )

            markdown, pages, _tables = pdf_parser._collect_mineru_output(root, "paper")

        self.assertEqual(markdown, "当前论文正文")
        self.assertEqual(pages, [{"page": 2, "text": "第二页"}])

    def test_internal_temp_dir_is_removed_and_subprocess_is_bounded(self) -> None:
        seen_output: list[Path] = []

        def fake_run(cmd, **kwargs):
            output = Path(cmd[cmd.index("-o") + 1])
            seen_output.append(output)
            (output / "paper.md").write_text("有效正文", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "paper.pdf"
            pdf.write_bytes(b"%PDF-test")
            with (
                patch.object(pdf_parser, "_resolve_mineru_binary", return_value="mineru"),
                patch.object(pdf_parser.subprocess, "run", side_effect=fake_run) as run,
            ):
                parsed = pdf_parser._parse_with_mineru(pdf, None)

        self.assertEqual(parsed["parser"], "mineru")
        self.assertEqual(len(seen_output), 1)
        self.assertFalse(seen_output[0].exists())
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["timeout"], pdf_parser._MINERU_TIMEOUT_SECONDS)
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")

    def test_timeout_is_reported_as_parse_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"%PDF-test")
            with (
                patch.object(pdf_parser, "_resolve_mineru_binary", return_value="mineru"),
                patch.object(
                    pdf_parser.subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired("mineru", 1),
                ),
            ):
                with self.assertRaisesRegex(pdf_parser.ParseError, "超过"):
                    pdf_parser._parse_with_mineru(pdf, root / "output")


class FallbackParsingTests(unittest.TestCase):
    def test_scanned_pdf_with_only_image_markup_is_rejected(self) -> None:
        class Page:
            def get_text(self, _mode: str) -> str:
                return "  ![](scan.png)  "

        class Document:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def __iter__(self):
                return iter([Page()])

        fake_fitz = types.SimpleNamespace(open=lambda _path: Document())
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "scan.pdf"
            pdf.write_bytes(b"%PDF-test")
            with patch.dict(sys.modules, {"fitz": fake_fitz}):
                with self.assertRaisesRegex(pdf_parser.ParseError, "扫描件"):
                    pdf_parser._parse_with_pymupdf(pdf)

    def test_mineru_failure_keeps_pymupdf_fallback(self) -> None:
        fallback = {
            "source": "paper.pdf",
            "parser": "pymupdf",
            "markdown": "text",
            "pages": [{"page": 1, "text": "text"}],
            "tables": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "paper.pdf"
            pdf.write_bytes(b"%PDF-test")
            with (
                patch.object(pdf_parser.settings, "use_fallback_parser", False),
                patch.object(pdf_parser, "_mineru_available", return_value=True),
                patch.object(
                    pdf_parser,
                    "_parse_with_mineru",
                    side_effect=pdf_parser.ParseError("mineru failed"),
                ),
                patch.object(pdf_parser, "_parse_with_pymupdf", return_value=fallback),
            ):
                self.assertIs(pdf_parser.parse_pdf(pdf), fallback)

    def test_atomic_json_write_preserves_previous_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "parsed.json"
            destination.write_text("old", encoding="utf-8")
            with patch.object(pdf_parser.json, "dump", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    pdf_parser.save_parsed({"markdown": "new"}, destination)

            self.assertEqual(destination.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
