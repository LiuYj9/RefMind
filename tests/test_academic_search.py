"""开放学术索引适配与摘要证据边界测试。"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from refmind.config import settings
from refmind.services import academic_search


class AcademicSearchTests(unittest.TestCase):
    def test_auto_mode_keeps_crossref_results_when_semantic_scholar_is_limited(self) -> None:
        document = Document(
            page_content="论文题名：Paper\n摘要：Evidence",
            metadata={
                "evidence_origin": "academic_search",
                "provider_key": "crossref",
                "evidence_id": "academic:crossref:demo",
                "paper_title": "Paper",
            },
        )
        semantic = Mock(side_effect=academic_search.AcademicSearchError("HTTP 429"))
        crossref = Mock(return_value=[document])
        with (
            patch.object(settings, "academic_search_enabled", True),
            patch.object(settings, "academic_search_provider", "auto"),
            patch.object(settings, "openalex_api_key", ""),
            patch.dict(
                academic_search._PROVIDERS,
                {
                    "semantic_scholar": semantic,
                    "crossref": crossref,
                },
            ),
        ):
            result = academic_search.search_academic_papers("query", limit=5)

        self.assertFalse(result.failed)
        self.assertEqual(result.documents, (document,))
        self.assertEqual(result.providers, ("crossref",))
        self.assertIn("HTTP 429", result.warnings[0])

    def test_semantic_scholar_normalises_abstract_evidence_and_attribution(self) -> None:
        payload = {
            "data": [
                {
                    "paperId": "paper-1",
                    "title": "HTS motor loss reduction",
                    "abstract": "A shielding method reduces AC loss.",
                    "year": 2025,
                    "authors": [{"name": "Ada Chen"}],
                    "venue": "IEEE TAS",
                    "url": "https://www.semanticscholar.org/paper/paper-1",
                    "externalIds": {"DOI": "10.1000/example"},
                    "citationCount": 12,
                    "openAccessPdf": {"url": "https://example.org/paper.pdf"},
                },
                {
                    "paperId": "metadata-only",
                    "title": "No abstract",
                    "abstract": None,
                    "year": 2024,
                },
            ]
        }
        with (
            patch.object(settings, "academic_search_enabled", True),
            patch.object(settings, "academic_search_provider", "semantic_scholar"),
            patch.object(settings, "semantic_scholar_api_key", "secret"),
            patch.object(academic_search, "_request_json", return_value=payload) as request,
        ):
            result = academic_search.search_academic_papers(
                "high-temperature superconducting motor",
                limit=10,
            )

        self.assertFalse(result.failed)
        self.assertEqual(len(result.documents), 1)
        document = result.documents[0]
        self.assertEqual(document.metadata["evidence_origin"], "academic_search")
        self.assertEqual(document.metadata["evidence_level"], "abstract")
        self.assertEqual(document.metadata["doi"], "10.1000/example")
        self.assertIn("utm_source=api", document.metadata["external_url"])
        self.assertIn("摘要：A shielding method", document.page_content)
        kwargs = request.call_args.kwargs
        self.assertEqual(kwargs["headers"], {"x-api-key": "secret"})
        self.assertNotIn("-", kwargs["params"]["query"])

    def test_crossref_strips_jats_and_keeps_only_results_with_abstracts(self) -> None:
        payload = {
            "message": {
                "items": [
                    {
                        "DOI": "10.2/demo",
                        "title": ["A paper"],
                        "abstract": "<jats:p>Direct <b>evidence</b>.</jats:p>",
                        "author": [{"given": "Lin", "family": "Wu"}],
                        "published": {"date-parts": [[2023, 1, 2]]},
                        "container-title": ["Journal"],
                        "URL": "https://doi.org/10.2/demo",
                    }
                ]
            }
        }
        with (
            patch.object(settings, "academic_search_enabled", True),
            patch.object(settings, "academic_search_provider", "crossref"),
            patch.object(settings, "crossref_mailto", "dev@example.com"),
            patch.object(academic_search, "_request_json", return_value=payload) as request,
        ):
            result = academic_search.search_academic_papers("query")

        self.assertEqual(len(result.documents), 1)
        self.assertIn("摘要：Direct evidence .", result.documents[0].page_content)
        self.assertEqual(result.documents[0].metadata["publication_year"], 2023)
        self.assertEqual(request.call_args.kwargs["params"]["mailto"], "dev@example.com")

    def test_openalex_requires_api_key_and_failure_is_observable(self) -> None:
        with (
            patch.object(settings, "academic_search_enabled", True),
            patch.object(settings, "academic_search_provider", "openalex"),
            patch.object(settings, "openalex_api_key", ""),
        ):
            result = academic_search.search_academic_papers("query")

        self.assertTrue(result.failed)
        self.assertEqual(result.documents, ())
        self.assertIn("OPENALEX_API_KEY", result.warnings[0])

    def test_external_url_rejects_non_http_and_credentials(self) -> None:
        self.assertEqual(academic_search.sanitize_external_url("javascript:alert(1)"), "")
        self.assertEqual(academic_search.sanitize_external_url("file:///tmp/paper.pdf"), "")
        self.assertEqual(
            academic_search.sanitize_external_url("https://user:pass@example.org/paper"),
            "",
        )
        self.assertEqual(
            academic_search.sanitize_external_url("https://example.org/paper"),
            "https://example.org/paper",
        )


if __name__ == "__main__":
    unittest.main()
