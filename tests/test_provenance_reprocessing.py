"""Offline cached-parse identity versus current rechunk/embed configuration."""
import copy
import json
import os
import unittest
from unittest.mock import patch

from refmind.config import settings
from refmind.rag import provenance
from refmind.rag.document_processor import parsed_to_documents


def cached_parse():
    return {'parser': 'mineru', 'paper_title': 'Cached Paper', 'file_sha256': 'f' * 64,
            'pages': [{'page': 1, 'text': 'measured result. ' * 25}],
            'processing_versions': {'schema': 1, 'parser': 'mineru',
                'packages': {'mineru': 'old-mineru', 'PyMuPDF': 'old-fitz', 'langchain-text-splitters': 'old-splitter'},
                'chunker': 'old-chunker', 'chunk_size': 9999, 'chunk_overlap': 99,
                'layout_chunk_max_chars': 9999, 'embedding_provider': 'dashscope',
                'embedding_model': 'old-embedding', 'configuration_sha256': 'old-config'}}


class ProvenanceReprocessingTests(unittest.TestCase):
    def make(self, parsed, chunk_size=64, provider='openai', model='current-embedding'):
        with (patch.dict(os.environ, {'EMBEDDING_PROVIDER': provider}),
              patch.object(settings, 'chunk_size', chunk_size),
              patch.object(settings, 'chunk_overlap', 0),
              patch.object(settings, 'layout_chunk_max_chars', 160),
              patch.object(settings, 'embedding_model', model),
              patch.object(provenance, 'version', side_effect=lambda package: 'now-' + package)):
            return parsed_to_documents(parsed, 7, 'paper.pdf', doc_id=3)

    def test_cached_parse_uses_current_chunks_and_embedding_but_original_parser_packages(self):
        parsed = cached_parse()
        original = copy.deepcopy(parsed)
        docs = self.make(parsed)
        self.assertGreater(len(docs), 1)
        self.assertTrue(all(len(doc.page_content) <= 64 for doc in docs))
        metadata = docs[0].metadata
        self.assertEqual(metadata['chunk_size_config'], 64)
        self.assertEqual(metadata['chunk_overlap_config'], 0)
        self.assertEqual(metadata['layout_chunk_max_chars_config'], 160)
        self.assertEqual(metadata['embedding_provider'], 'openai')
        self.assertEqual(metadata['embedding_model'], 'text-embedding-3-small')
        self.assertNotEqual(metadata['processing_config_sha256'], 'old-config')
        parser = json.loads(metadata['parsing_provenance'])
        self.assertEqual(parser['packages'], {'mineru': 'old-mineru', 'PyMuPDF': 'old-fitz'})
        self.assertTrue(parser['recorded'])
        self.assertNotIn('chunk_size', parser)
        current = json.loads(metadata['processing_provenance'])
        self.assertEqual(current['chunk_size'], 64)
        self.assertEqual(current['packages']['langchain-text-splitters'], 'now-langchain-text-splitters')
        self.assertEqual(current['parsing_provenance'], parser)
        self.assertEqual(json.loads(metadata['parser_versions']), parser['packages'])
        self.assertEqual(parsed, original)

    def test_rechunk_fingerprint_changes_while_parsing_fingerprint_stays_stable(self):
        first = self.make(cached_parse(), chunk_size=64)[0].metadata
        second = self.make(cached_parse(), chunk_size=80)[0].metadata
        self.assertNotEqual(first['processing_config_sha256'], second['processing_config_sha256'])
        self.assertEqual(first['parsing_provenance_sha256'], second['parsing_provenance_sha256'])

    def test_reembedding_fingerprint_uses_actual_current_provider_and_model(self):
        parsed = cached_parse()
        first = self.make(parsed, provider='dashscope', model='current-A')[0].metadata
        second = self.make(parsed, provider='dashscope', model='current-B')[0].metadata
        third = self.make(parsed, provider='openai')[0].metadata
        self.assertEqual(first['embedding_model'], 'current-A')
        self.assertEqual(second['embedding_model'], 'current-B')
        self.assertNotEqual(first['processing_config_sha256'], second['processing_config_sha256'])
        self.assertNotEqual(second['processing_config_sha256'], third['processing_config_sha256'])
        self.assertEqual(first['parsing_provenance_sha256'], third['parsing_provenance_sha256'])

    def test_old_payload_without_provenance_does_not_invent_current_parser_versions(self):
        parsed = {'parser': 'legacy-parser', 'pages': [{'page': 1, 'text': 'Legacy result.'}]}
        metadata = self.make(parsed)[0].metadata
        parser = json.loads(metadata['parsing_provenance'])
        self.assertEqual(parser['parser'], 'legacy-parser')
        self.assertEqual(parser['packages'], {})
        self.assertFalse(parser['recorded'])
        self.assertEqual(json.loads(metadata['parser_versions']), {})
        self.assertNotIn('now-mineru', metadata['processing_provenance'])

    def test_explicit_parse_snapshot_takes_precedence_and_keeps_parser_version(self):
        parsed = cached_parse()
        parsed['parsing_provenance'] = {'parser': 'specific-parser', 'parser_version': '4.2',
            'packages': {'parser-package': '4.2'}, 'options': {'ocr': False}}
        original = copy.deepcopy(parsed)
        metadata = self.make(parsed)[0].metadata
        parser = json.loads(metadata['parsing_provenance'])
        self.assertEqual(parser['parser'], 'specific-parser')
        self.assertEqual(metadata['parser'], 'specific-parser')
        self.assertEqual(parser['parser_version'], '4.2')
        self.assertEqual(parser['options'], {'ocr': False})
        self.assertEqual(parser['packages'], {'parser-package': '4.2'})
        self.assertEqual(parsed, original)

    def test_all_added_vector_metadata_remains_scalar(self):
        metadata = self.make(cached_parse())[0].metadata
        self.assertTrue(all(not isinstance(value, (dict, list, tuple)) for value in metadata.values()))
        self.assertEqual(metadata['file_sha256'], 'f' * 64)

    def test_original_processing_versions_call_remains_compatible_for_new_parse(self):
        with patch.object(provenance, 'version', side_effect=lambda package: 'now-' + package):
            snapshot = provenance.processing_versions('mineru')
        self.assertEqual(snapshot['parser'], 'mineru')
        self.assertEqual(snapshot['packages']['mineru'], 'now-mineru')
        self.assertNotIn('parsing_provenance', snapshot)
        self.assertEqual(len(snapshot['configuration_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
