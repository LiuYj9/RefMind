"""PDF 引用页渲染与证据高亮回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz

from refmind.pdf_locator import render_pdf_location


class PdfLocatorTests(unittest.TestCase):
    def test_target_page_is_rendered_and_matching_text_is_highlighted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((72, 100), "Superconducting rotor evidence paragraph")
            pdf.save(path)
            pdf.close()

            preview = render_pdf_location(
                path,
                1,
                evidence_text="Superconducting rotor evidence paragraph",
            )

        self.assertEqual(preview.page_number, 1)
        self.assertTrue(preview.highlighted)
        self.assertTrue(preview.image.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_out_of_range_page_fails_with_actionable_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            pdf = fitz.open()
            pdf.new_page()
            pdf.save(path)
            pdf.close()

            with self.assertRaisesRegex(ValueError, "超出 PDF 范围"):
                render_pdf_location(path, 2)


if __name__ == "__main__":
    unittest.main()
