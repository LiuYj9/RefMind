import threading
import unittest
from unittest.mock import patch
from refmind.config import settings
from refmind.services import paper_discovery as discovery


class DiscoveryProgressTests(unittest.TestCase):
    def test_fast_source_is_delivered_before_slow_source_completes(self):
        release = threading.Event()
        events = []
        def fetch(provider, *args):
            if provider == 'openalex':
                if not release.wait(.5):
                    raise RuntimeError('progress never reached consumer')
            return [discovery.PaperCandidate(provider + ':1', provider)], None
        def progress(snapshot):
            events.append(snapshot)
            if snapshot.provider_status.get('openalex') == 'running' and snapshot.papers:
                release.set()
        with patch.object(settings, 'academic_search_enabled', True), patch.object(settings, 'openalex_api_key', ''), patch.object(discovery, '_fetch_provider', side_effect=fetch):
            result = discovery.discover_papers('query', on_progress=progress)
        self.assertTrue(release.is_set())
        self.assertEqual(result.provider_status['openalex'], 'completed')
        self.assertEqual(len(result.papers), 3)
        self.assertTrue(any(p.provider_status['openalex'] == 'running' and p.papers for p in events))

    def test_merge_retains_conflicting_source_values_without_title_similarity_merge(self):
        a = discovery._candidate('crossref', {'DOI': '10.1234/work', 'title': ['Published title'], 'published': {'date-parts': [[2025]]}})
        b = discovery._candidate('openalex', {'id': 'W1', 'doi': 'https://doi.org/10.1234/work', 'display_name': 'Repository title', 'publication_year': 2024, 'abstract_inverted_index': {'Evidence': [0]}})
        result = discovery.merge_candidates([a, b])[0]
        self.assertEqual(result.title, 'Published title')
        self.assertEqual({entry['value'] for entry in result.field_sources['title']}, {'Published title', 'Repository title'})
        self.assertEqual(result.field_sources['abstract'][0]['provider'], 'openalex')
        different = discovery.PaperCandidate('different', result.title)
        self.assertEqual(len(discovery.merge_candidates([result, different])), 2)


if __name__ == '__main__':
    unittest.main()
