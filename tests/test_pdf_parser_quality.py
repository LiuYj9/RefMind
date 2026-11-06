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
            (output / "paper_content_list.json").write_text(
                json.dumps([{"type": "text", "text": "有效正文", "page_idx": 0}]),
                encoding="utf-8",
            )
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


class MinerUStructurePreservationTests(unittest.TestCase):
    @staticmethod
    def table(caption, page, body="| A | B |\n| 1 | 2 |", **extra):
        return {"type": "table", "table_caption": [caption], "page_idx": page,
                "table_body": body, **extra}

    def parse_items(self, items):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "paper_content_list.json").write_text(json.dumps(items), encoding="utf-8")
            _, _, tables = pdf_parser._collect_mineru_output(root, "paper")
            blocks = pdf_parser._collect_mineru_layout_blocks(root, "paper")
        return tables, blocks

    def test_native_html_blank_cells_spans_and_footnotes_survive_indexing(self):
        from refmind.rag.document_processor import parsed_to_documents

        html = '<table><tr><td rowspan="2">A</td><td>10</td></tr><tr><td></td></tr></table>'
        item = self.table("Table 1", 0, html, table_footnote=["Blank means not measured"])
        tables, blocks = self.parse_items([item])
        self.assertEqual(tables[0]["raw_text"], html)
        self.assertEqual(tables[0]["source_items"], [item])
        self.assertEqual(tables[0]["merge_method"], "mineru")
        docs = parsed_to_documents({"blocks": blocks}, 1, "paper.pdf")
        self.assertEqual(len(docs), 1)
        self.assertIn(html, docs[0].page_content)
        self.assertIn("Blank means not measured", docs[0].page_content)
        self.assertNotIn("Columns:", docs[0].page_content)

    def test_same_header_and_empty_caption_are_not_enough_to_merge(self):
        tables, _ = self.parse_items([self.table("Table 1", 0), self.table("", 1)])
        self.assertEqual(len(tables), 2)

    def test_same_number_without_continued_marker_is_not_merged(self):
        tables, _ = self.parse_items([self.table("Table 1", 0), self.table("Table 1", 1)])
        self.assertEqual(len(tables), 2)

    def test_explicit_continuation_retains_all_original_fragments(self):
        items = [self.table("Table 1", 0), self.table("Table 1 continued", 1),
                 self.table("Table 1 continued", 2)]
        tables, blocks = self.parse_items(items)
        self.assertEqual(len(tables), 1)
        self.assertEqual((blocks[0]["page"], blocks[0]["page_end"]), (1, 3))
        self.assertEqual(tables[0]["source_items"], items)
        self.assertEqual(tables[0]["merge_method"], "caption_continuation")
        self.assertEqual(tables[0]["raw_text"].count("| A | B |"), 3)

    def test_conflicting_number_nonadjacent_page_and_intervening_text_block_merge(self):
        for items in (
            [self.table("Table 1", 0), self.table("Table 2 continued", 1)],
            [self.table("Table 1", 0), self.table("Table 1 continued", 2)],
            [self.table("Table 1", 0), {"type": "text", "text": "Other section", "page_idx": 1},
             self.table("Table 1 continued", 1)],
        ):
            with self.subTest(items=items):
                tables, _ = self.parse_items(items)
                self.assertEqual(len(tables), 2)

    def test_caption_only_block_is_not_a_second_table(self):
        tables, blocks = self.parse_items([
            self.table("Table 1", 0),
            {"type": "table_caption", "text": "Table 1", "page_idx": 0},
        ])
        self.assertEqual(len(tables), 1)
        self.assertEqual(sum(b["type"] == "table" for b in blocks), 1)

    def test_native_heading_list_and_image_associations_are_used(self):
        items = [
            {"type": "text", "text": "Methods", "text_level": 1, "page_idx": 0},
            {"type": "list", "list_items": ["First", "Second"], "page_idx": 0},
            {"type": "image", "image_caption": ["Figure 1"],
             "image_footnote": ["Scale in mm"], "page_idx": 0},
        ]
        _, blocks = self.parse_items(items)
        self.assertEqual([b["type"] for b in blocks], ["heading", "list_item", "figure"])
        self.assertEqual(blocks[1]["text"], "First\nSecond")
        self.assertEqual(blocks[2]["text"], "Figure 1\nScale in mm")
        self.assertEqual(blocks[2]["mineru_source"], items[2])

    def test_missing_or_invalid_page_is_rejected_instead_of_becoming_page_one(self):
        for fields in ({}, {"page_idx": -1}, {"page_no": 0}, {"page_idx": True}):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(pdf_parser.ParseError, "有效页码"):
                    self.parse_items([{"type": "text", "text": "Evidence", **fields}])

    def test_markdown_only_output_has_no_invented_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "paper.md").write_text("Entire paper text", encoding="utf-8")
            _, pages, _ = pdf_parser._collect_mineru_output(root, "paper")
        self.assertEqual(pages, [])

    def test_markdown_only_parse_triggers_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"%PDF-test")
            def fake_run(cmd, **kwargs):
                out = Path(cmd[cmd.index("-o") + 1])
                (out / "paper.md").write_text("Entire paper text", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            with (patch.object(pdf_parser.settings, "use_fallback_parser", False),
                  patch.object(pdf_parser, "_mineru_available", return_value=True),
                  patch.object(pdf_parser.subprocess, "run", side_effect=fake_run),
                  patch.object(pdf_parser, "_parse_with_pymupdf", return_value={"parser": "pymupdf"})):
                self.assertEqual(pdf_parser.parse_pdf(pdf)["parser"], "pymupdf")


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
