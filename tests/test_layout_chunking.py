"""Layout-aware 论文切分的离线回归测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from refmind.rag.document_processor import parsed_to_documents


class LayoutChunkingTests(unittest.TestCase):
    def test_heading_sets_section_and_blocks_follow_reading_order(self) -> None:
        parsed = {
            "parser": "mineru",
            "layout_confidence": "high",
            "pages": [{"page": 1, "text": "不得重复进入索引"}],
            "blocks": [
                {"block_id": "b2", "type": "text", "text": "第二段", "page": 1, "reading_order": 2},
                {"block_id": "h1", "type": "title", "text": "2 Methods", "page": 1, "reading_order": 0},
                {"block_id": "b1", "type": "text", "text": "第一段", "page": 1, "reading_order": 1},
            ],
        }
        with patch("refmind.rag.document_processor.settings.chunk_size", 1000):
            documents = parsed_to_documents(parsed, 7, "paper.pdf", doc_id=9)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "第一段\n\n第二段")
        self.assertEqual(documents[0].metadata["section"], "2 Methods")
        self.assertNotIn("不得重复", documents[0].page_content)
        self.assertIsInstance(documents[0].metadata["block_ids"], str)
        self.assertTrue(all(not isinstance(v, (list, dict)) for v in documents[0].metadata.values()))

    def test_atomic_table_is_not_merged_with_paragraphs(self) -> None:
        parsed = {
            "parser": "mineru",
            "blocks": [
                {"block_id": "p", "type": "text", "text": "实验设置", "page": 2, "reading_order": 0},
                {"block_id": "t", "type": "table", "text": "Table: Results\nA=1", "page": 2, "page_end": 3, "reading_order": 1},
            ],
        }
        documents = parsed_to_documents(parsed, 1, "paper.pdf")

        self.assertEqual([d.metadata["content_type"] for d in documents], ["paragraph", "table"])
        self.assertEqual(documents[1].metadata["page_end"], 3)

    def test_pages_only_payload_keeps_legacy_fallback(self) -> None:
        parsed = {"parser": "legacy", "pages": [{"page": 4, "text": "Legacy paragraph."}], "tables": []}
        documents = parsed_to_documents(parsed, 1, "old.pdf")
        self.assertEqual(documents[0].metadata["page"], 4)
        self.assertEqual(documents[0].page_content, "Legacy paragraph.")


if __name__ == "__main__":
    unittest.main()
