import time
import unittest
from unittest.mock import patch

from refmind.config import settings
from refmind.services import paper_discovery as discovery
from refmind.services.academic_search import AcademicSearchError


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.enabled = patch.object(settings, "academic_search_enabled", True)
        self.enabled.start()
        self.addCleanup(self.enabled.stop)

    def test_metadata_only_paper_is_retained_and_round_trips(self):
        payload = {"data": [{"paperId": "s1", "title": "A paper without an abstract", "abstract": None, "year": 2025}], "next": 1}
        with patch.object(discovery, "_request_json", return_value=payload):
            result = discovery.discover_papers("a paper", provider="semantic_scholar", limit=1)
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.papers[0].abstract, "")
        self.assertIsNotNone(result.next_cursor)
        restored = discovery.PaperCandidate.from_dict(result.papers[0].to_dict())
        self.assertEqual(restored, result.papers[0])

    def test_same_doi_merges_missing_fields_and_preserves_versions(self):
        a = discovery.PaperCandidate("doi:10.1234/paper", "Paper", doi="10.1234/paper", providers=("crossref",))
        b = discovery.PaperCandidate("doi:10.1234/paper", "Paper", "Abstract", doi="10.1234/paper", providers=("openalex",), abstract_source="openalex",
            locations=(discovery.PaperLocation("https://repository.example/preprint.pdf", is_oa=True, version="submittedVersion"),
                       discovery.PaperLocation("https://publisher.example/article.pdf", is_oa=True, version="publishedVersion")))
        merged = discovery.merge_candidates([a, b])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].abstract_source, "openalex")
        self.assertEqual(len(merged[0].locations), 2)
        self.assertEqual(merged[0].providers, ("crossref", "openalex"))

    def test_doi_exact_match_rejects_wrong_work(self):
        payload = {"message": {"DOI": "10.1234/other", "title": ["Another paper"]}}
        with patch.object(discovery, "_request_json", return_value=payload) as request:
            result = discovery.discover_papers(doi="https://doi.org/10.1234/Wanted", provider="crossref")
        self.assertEqual(result.papers, ())
        self.assertIn("10.1234%2Fwanted", request.call_args.args[0])

    def test_doi_query_uses_exact_lookup_for_existing_query_only_clients(self):
        with patch.object(discovery, "_request_json", return_value={"message": {"DOI": "10.1234/wanted", "title": ["Wanted paper"]}}) as request:
            result = discovery.discover_papers("https://doi.org/10.1234/Wanted", provider="crossref")
        self.assertEqual(result.papers[0].doi, "10.1234/wanted")
        self.assertIn("/works/10.1234%2Fwanted", request.call_args.args[0])

    def test_openalex_without_key_and_oa_filters(self):
        payload = {"results": [{"id": "W1", "display_name": "Open paper", "publication_year": 2024,
                    "doi": "https://doi.org/10.1234/open", "best_oa_location": {"pdf_url": "https://repo.example/a.pdf", "is_oa": True, "version": "acceptedVersion"}},
                    {"id": "W2", "display_name": "Old closed paper", "publication_year": 2010}], "meta": {"next_cursor": "next"}}
        with patch.object(settings, "openalex_api_key", ""), patch.object(discovery, "_request_json", return_value=payload) as request:
            result = discovery.discover_papers("paper", provider="openalex", filters=discovery.SearchFilters(2020, 2025, True))
        self.assertEqual(len(result.papers), 1)
        self.assertNotIn("api_key", request.call_args.kwargs["params"])
        self.assertIn("open_access.is_oa:true", request.call_args.kwargs["params"]["filter"])

    def test_pagination_is_bound_to_query_and_filters(self):
        with patch.object(discovery, "_request_json", return_value={"data": [{"paperId": "s1", "title": "Paper"}], "next": 7}) as request:
            first = discovery.discover_papers("query", provider="semantic_scholar")
            discovery.discover_papers("query", provider="semantic_scholar", cursor=first.next_cursor)
            self.assertEqual(request.call_args.kwargs["params"]["offset"], 7)
            with self.assertRaises(ValueError):
                discovery.discover_papers("changed", provider="semantic_scholar", cursor=first.next_cursor)

    def test_partial_provider_failure_does_not_drop_metadata(self):
        def fetch(provider, *args):
            if provider == "semantic_scholar":
                raise AcademicSearchError("429")
            return [discovery.PaperCandidate("crossref:x", "Metadata only")], None
        with patch.object(settings, "openalex_api_key", ""), patch.object(discovery, "_fetch_provider", side_effect=fetch):
            result = discovery.discover_papers("query")
        self.assertFalse(result.failed)
        self.assertTrue(result.partial)
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.provider_status["semantic_scholar"], "failed")

    def test_total_deadline_returns_without_waiting_for_slow_source(self):
        def fetch(*args):
            time.sleep(.15)
            return [], None
        with patch.object(discovery, "_fetch_provider", side_effect=fetch):
            started = time.monotonic()
            result = discovery.discover_papers("query", provider="crossref", timeout_seconds=.02)
            self.assertLess(time.monotonic() - started, .12)
        self.assertTrue(result.failed)
        self.assertEqual(result.provider_status["crossref"], "timeout")

    def test_invalid_filters_and_cursor_are_rejected_without_network(self):
        with patch.object(discovery, "_request_json") as request:
            with self.assertRaises(ValueError):
                discovery.SearchFilters(2025, 2020)
            with self.assertRaises(ValueError):
                discovery.discover_papers("query", cursor="bad!", provider="crossref")
            with self.assertRaises(ValueError):
                discovery.discover_papers(doi="not-a-doi")
            request.assert_not_called()

    def test_missing_title_never_becomes_a_none_string(self):
        with patch.object(discovery, "_request_json", return_value={"data": [{"paperId": "s1"}]}):
            result = discovery.discover_papers("query", provider="semantic_scholar")
        self.assertEqual(result.papers, ())

    def test_one_hop_reference_lookup_retains_metadata_only_papers(self):
        paper = discovery.PaperCandidate("doi:10.1234/root", "Root", doi="10.1234/root")
        payload = {"data": [{"citedPaper": {"paperId": "reference", "title": "A cited work", "abstract": None}}], "next": 10}
        with patch.object(discovery, "_request_json", return_value=payload) as request:
            result = discovery.related_papers(paper)
        self.assertEqual(result.papers[0].abstract, "")
        self.assertTrue(request.call_args.args[0].endswith("/references"))
        self.assertIsNotNone(result.next_cursor)
        with self.assertRaises(ValueError):
            discovery.related_papers(paper, kind="citations", cursor=result.next_cursor)

    def test_result_replay_preserves_partial_and_version_metadata(self):
        result = discovery.DiscoveryResult((discovery.PaperCandidate("x", "Paper"),), partial=True, provider_status={"openalex": "timeout"})
        self.assertEqual(discovery.DiscoveryResult.from_dict(result.to_dict()), result)


if __name__ == "__main__":
    unittest.main()
