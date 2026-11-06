"""Legacy reconstruction must never sacrifice source files or saved citations."""
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from refmind import storage
from refmind.config import settings
from refmind.services import ingestion, reindex
from refmind.storage.runtime_lock import acquire_runtime_lock, release_runtime_lock
from refmind.rag.retrieval import filter_ready_documents


class LegacyReindexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.tmp.name)
        for key, name in {'database_path':'test.db','upload_dir':'uploads','parsed_dir':'parsed',
                          'docstore_dir':'assets','chroma_persist_dir':'vectors'}.items():
            self.stack.enter_context(patch.object(settings,key,root/name))
        storage.init_db()
        self.group = storage.create_group('legacy').id
        self.pdf = settings.upload_dir/'original.pdf'
        self.pdf.write_bytes(b'%PDF-verified-fixture')
        self.digest = hashlib.sha256(self.pdf.read_bytes()).hexdigest()
        self.parsed = settings.parsed_dir/'old.json'
        self.parsed.write_text(json.dumps({'file_sha256':self.digest,
            'processing_versions':reindex.processing_versions('fixture')}),encoding='utf-8')
        self.doc = storage.create_document(self.group,'original.pdf',str(self.pdf),status='ready')
        storage.update_document(self.doc,parsed_json_path=str(self.parsed),summary='keep summary')
        self.document = storage.get_document(self.doc)
        self.parsed_before = self.parsed.read_bytes()
        self.stack.enter_context(patch.object(reindex,'parse_pdf',return_value={
            'parser':'fixture','pages':[{'page':1,'text':'Current 29 A.'}],'markdown':'Current 29 A.'}))
        self.stack.enter_context(patch.object(ingestion,'_enrich_figure_blocks',return_value=0))
        self.stack.enter_context(patch.object(ingestion,'prepare_ingestion_batch',return_value=object()))
        self.delete = self.stack.enter_context(patch.object(ingestion,'_delete_chunks_by_doc_id'))
        self.write = self.stack.enter_context(patch.object(ingestion,'ingest_documents',side_effect=lambda g,d,**kw:len(d)))
        self.stack.enter_context(patch.object(ingestion,'invalidate_retriever'))

    def rebuild(self):
        return reindex.rebuild_legacy_index(self.doc,expected_sha256=self.digest)

    def assert_originals(self):
        row = storage.get_document(self.doc)
        self.assertIsNotNone(row)
        self.assertEqual(row.original_path,str(self.pdf))
        self.assertEqual(hashlib.sha256(self.pdf.read_bytes()).hexdigest(),self.digest)
        self.assertEqual(self.parsed.read_bytes(),self.parsed_before)
        self.assertEqual(row.summary,'keep summary')

    def test_rebuild_preserves_identity_and_filters_stale_then_skips_repeat(self):
        rebuilt = self.rebuild()
        self.assertEqual((rebuilt.id,rebuilt.library_index),(self.doc,self.document.library_index))
        self.assertEqual((rebuilt.status,rebuilt.file_sha256),('ready',self.digest))
        self.assertTrue(rebuilt.index_version.startswith('reindex-'))
        docs=self.write.call_args.args[1]
        stale=Document(page_content='Old 11 A.',metadata={**docs[0].metadata,'version':'old'})
        self.assertEqual(filter_ready_documents(self.group,[stale,*docs]),docs)
        self.assertEqual(self.rebuild(),rebuilt)
        self.assertEqual(self.write.call_count,1)
        self.assert_originals()

    def test_source_mismatch_and_record_conflict_do_not_withdraw_or_embed(self):
        for change in ('source','record','missing'):
            with self.subTest(change=change):
                if change=='source': self.pdf.write_bytes(b'changed')
                else:
                    self.pdf.write_bytes(b'%PDF-verified-fixture')
                    self.parsed.write_text(json.dumps({'file_sha256':'a'*64} if change=='record' else {}))
                with self.assertRaises(ValueError): self.rebuild()
                self.assertEqual(storage.get_document(self.doc).status,'ready')
        self.write.assert_not_called()
        self.delete.assert_not_called()

    def test_api_writer_lock_rejects_rebuild_and_remains_held(self):
        acquire_runtime_lock(settings.database_path,'api')
        try:
            with self.assertRaises(RuntimeError): self.rebuild()
            with self.assertRaises(RuntimeError): acquire_runtime_lock(settings.database_path,'streamlit')
        finally:
            release_runtime_lock(settings.database_path)
        self.write.assert_not_called()

    def test_embedding_failure_withdraws_legacy_and_retry_succeeds(self):
        with patch.object(ingestion,'prepare_ingestion_batch',side_effect=RuntimeError('embedding failure')):
            with self.assertRaisesRegex(RuntimeError,'embedding failure'): self.rebuild()
        self.assertEqual(storage.get_document(self.doc).status,'reindex_failed')
        self.delete.assert_not_called()
        self.assertEqual(filter_ready_documents(self.group,[Document(page_content='old',metadata={'doc_id':self.doc})]),[])
        with patch.object(ingestion,'remove_document') as remove:
            ingestion.recover_incomplete_ingestions()
            remove.assert_not_called()
        self.assert_originals()
        self.assertEqual(self.rebuild().status,'ready')

    def test_partial_vector_failure_keeps_source_and_retry_replaces_partial(self):
        self.write.side_effect=RuntimeError('partial commit')
        with self.assertRaisesRegex(RuntimeError,'partial commit'): self.rebuild()
        self.assertEqual(storage.get_document(self.doc).status,'reindex_failed')
        self.assert_originals()
        self.write.side_effect=lambda g,d,**kw:len(d)
        self.assertEqual(self.rebuild().status,'ready')
        self.assertEqual(self.delete.call_count,2)

    def test_hard_interruption_state_is_retained_on_startup(self):
        storage.update_document(self.doc,status='reindexing')
        with patch.object(ingestion,'remove_document') as remove:
            ingestion.recover_incomplete_ingestions()
            remove.assert_not_called()
        self.assert_originals()
        self.assertEqual(self.rebuild().status,'ready')

    def test_source_changed_during_embedding_cannot_publish(self):
        def mutate(_docs):
            self.pdf.write_bytes(b'changed during embedding')
            return object()
        with patch.object(ingestion,'prepare_ingestion_batch',side_effect=mutate):
            with self.assertRaises(ValueError): self.rebuild()
        self.assertEqual(storage.get_document(self.doc).status,'reindex_failed')
        self.assertIsNone(storage.get_document(self.doc).index_version)
        self.write.assert_not_called()

    def test_lost_state_before_publish_does_not_restore_ready(self):
        def conflict(g,docs,**kw):
            storage.update_document(self.doc,status='deleting')
            return len(docs)
        self.write.side_effect=conflict
        with self.assertRaisesRegex(RuntimeError,'状态改变'): self.rebuild()
        self.assertEqual(storage.get_document(self.doc).status,'deleting')
        self.assertIsNone(storage.get_document(self.doc).index_version)
        self.assert_originals()

    def test_old_schema_gets_nullable_columns_before_rebuilding(self):
        with storage.connect() as conn:
            conn.execute('ALTER TABLE documents DROP COLUMN index_version')
            conn.execute('ALTER TABLE documents DROP COLUMN file_sha256')
        self.assertEqual(self.rebuild().status,'ready')

    def test_changed_or_unknown_embedding_configuration_never_reaches_embedding(self):
        for config in ({}, {'embedding_provider':'other','embedding_model':'unknown'}):
            with self.subTest(config=config):
                self.parsed.write_text(json.dumps({'file_sha256':self.digest,'processing_versions':config}))
                with self.assertRaisesRegex(ValueError,'嵌入配置'): self.rebuild()
                self.assertEqual(storage.get_document(self.doc).status,'ready')
        self.delete.assert_not_called()
        self.write.assert_not_called()

    def test_rebuild_uses_separate_asset_root_and_preserves_existing_assets(self):
        old_asset = settings.docstore_dir / f'doc_{self.doc}' / 'images' / 'old.png'
        old_asset.parent.mkdir(parents=True)
        old_asset.write_bytes(b'historical image')
        row=self.rebuild()
        root=ingestion._enrich_figure_blocks.call_args.kwargs['asset_root']
        self.assertTrue(root.is_relative_to(settings.docstore_dir / f'doc_{self.doc}' / 'reindexes'))
        self.assertEqual(root.name,row.index_version)
        self.assertEqual(old_asset.read_bytes(),b'historical image')
