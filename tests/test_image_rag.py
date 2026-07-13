"""图片摘要索引与携图回答的离线回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from refmind.config import settings
from refmind.llm.image_understanding import build_visual_content
from refmind.rag import graph
from refmind.rag.document_processor import parsed_to_documents
from refmind.services import ingestion


class ImageIndexTests(unittest.TestCase):
    def test_figure_asset_metadata_reaches_text_index(self) -> None:
        parsed = {
            "parser": "mineru",
            "blocks": [{
                "block_id": "fig_1", "type": "figure", "text": "Figure\nVisual summary: flowchart",
                "page": 2, "reading_order": 1, "image_id": "image_001",
                "image_path": "C:/safe/docstore/doc_7/images/figure.png", "image_mime_type": "image/png",
            }],
        }
        document = parsed_to_documents(parsed, 1, "paper.pdf", doc_id=7)[0]
        self.assertEqual(document.metadata["content_type"], "figure")
        self.assertEqual(document.metadata["image_id"], "image_001")
        self.assertTrue(document.metadata["image_path"].endswith("figure.png"))

    def test_only_docstore_images_are_encoded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "docstore"
            image = root / "doc_1" / "images" / "figure.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"png-bytes")
            outside = Path(tmp) / "outside.png"
            outside.write_bytes(b"private")
            docs = [
                Document(page_content="figure", metadata={"image_path": str(outside)}),
                Document(page_content="figure", metadata={"image_path": str(image)}),
            ]
            with patch.object(settings, "docstore_dir", root), patch.object(settings, "image_max_bytes", 1024):
                content = build_visual_content(docs)
        self.assertEqual(len(content), 1)
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_ingestion_enriches_figure_with_summary_without_indexing_bytes(self) -> None:
        parsed = {"blocks": [{"block_id": "f1", "type": "figure", "text": "图注", "page": 1}]}
        assets = [{"block_id": "f1", "image_id": "i1", "image_path": "C:/docstore/doc_1/images/a.png", "image_mime_type": "image/png", "page": 1, "page_end": 1, "bbox": []}]
        with (
            patch.object(ingestion, "extract_pdf_figures", return_value=assets),
            patch.object(ingestion.settings, "image_summary_enabled", True),
            patch.object(ingestion.settings, "dashscope_api_key", "test-key"),
            patch.object(ingestion, "summarize_image", return_value="图片类型：流程图；关系：A 到 B"),
        ):
            ingestion._enrich_figure_blocks(parsed, Path("paper.pdf"), doc_id=1)
        block = parsed["blocks"][0]
        self.assertIn("Visual summary", block["text"])
        self.assertNotIn("base64", block["text"])


class ImageAnswerTests(unittest.TestCase):
    def test_retrieved_figure_uses_multimodal_model(self) -> None:
        document = Document(page_content="Visual summary: flowchart", metadata={"image_path": "x", "filename": "paper.pdf", "page": 1})
        model = SimpleNamespace(invoke=lambda _messages: SimpleNamespace(content="携图答案"))
        with (
            patch.object(graph, "build_visual_content", return_value=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]),
            patch.object(graph, "get_multimodal_llm", return_value=model) as factory,
        ):
            answer = graph._generate_answer("图中是什么？", [document])
        self.assertEqual(answer, "携图答案")
        factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
