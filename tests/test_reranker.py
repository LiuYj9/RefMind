"""重排后端选择与安全降级测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from refmind.config import settings
from refmind.rag import reranker


class RerankerFallbackTests(unittest.TestCase):
    def test_missing_dashscope_sdk_skips_direct_backend_without_import_error(self) -> None:
        document = Document(page_content="evidence")
        with (
            patch.object(settings, "rerank_enabled", True),
            patch.object(settings, "dashscope_api_key", "configured"),
            patch.object(settings, "rerank_model", "gte-rerank-v2"),
            patch.object(reranker, "dashscope_sdk_available", return_value=False),
            patch.object(reranker, "_dashscope_rerank") as direct,
            patch.object(
                reranker,
                "_embedding_rerank",
                return_value=[(0.75, document)],
            ) as embedding,
        ):
            result = reranker.rerank("query", [document], top_n=1)

        direct.assert_not_called()
        embedding.assert_called_once()
        self.assertEqual(result[0].metadata["rerank_score"], 0.75)


if __name__ == "__main__":
    unittest.main()
