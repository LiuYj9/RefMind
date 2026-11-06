"""Maintenance requests must remain explicit, authorized and offline."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from contextlib import ExitStack,contextmanager
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock,patch
from dataclasses import replace
from fastapi.testclient import TestClient
from refmind import storage
from refmind.config import settings
from refmind.services import reindex,ingestion
from refmind.web import store,maintenance
from refmind.web.api import create_app, _attempts
from refmind.storage.runtime_lock import release_runtime_lock


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        for key,name in {'database_path':'fixture.db','upload_dir':'uploads','parsed_dir':'parsed',
                         'docstore_dir':'images','chroma_persist_dir':'vectors'}.items():
            self.stack.enter_context(patch.object(settings,key,Path(self.temp.name)/name))
        store.initialize()
        self.owner=store.create_user('owner','test-password-owner')
        self.viewer=store.create_user('viewer','test-password-viewer')
        self.outsider=store.create_user('outside','test-password-outside',admin=True)
        self.group=store.create_library(self.owner['id'],'Shared','shared')['id']
        with storage.connect() as conn:
            conn.execute('INSERT INTO rm_members VALUES(?,?,?)',(self.group,self.viewer['id'],'viewer'))
        pdf=settings.upload_dir/'fixture.pdf'
        pdf.write_bytes(b'%PDF-maintenance-fixture')
        self.digest=hashlib.sha256(pdf.read_bytes()).hexdigest()
        parsed=settings.parsed_dir/'fixture.json'
        parsed.write_text(json.dumps({'file_sha256':self.digest,'processing_versions':reindex.processing_versions('fixture')}))
        self.doc=storage.create_document(self.group,'fixture.pdf',str(pdf),status='ready')
        storage.update_document(self.doc,parsed_json_path=str(parsed))
        self.url=f'/api/documents/{self.doc}/reindex-requests'
        self.inspect_url=f'/api/documents/{self.doc}/index-maintenance'
        self.app=create_app(start_workers=False)
        _attempts.clear()

    @contextmanager
    def client(self, username='owner'):
        with TestClient(self.app) as client:
            login=client.post('/api/auth/login',json={'username':username,'password':'test-password-'+username})
            self.assertEqual(login.status_code,200)
            client.headers['X-CSRF-Token']=login.json()['csrf']
            yield client

    def request(self):
        return maintenance.request_rebuild(self.owner['id'],self.doc,self.digest)

    def test_owner_request_is_persistent_deduplicated_and_never_submitted_online(self):
        with self.client() as client:
            self.app.state.submit=Mock(side_effect=AssertionError('online execution'))
            info=client.get(self.inspect_url).json()
            self.assertTrue(info['can_request'])
            response=client.post(self.url,json={'expected_sha256':info['expected_sha256']})
            self.assertEqual(response.status_code,200,response.text)
            item=response.json()
            repeated=client.post(self.url,json={'expected_sha256':self.digest}).json()
            self.assertEqual(item['id'],repeated['id'])
            self.assertEqual(item['status'],maintenance.PENDING)
            self.assertNotIn('payload',item)
            self.app.state.submit.assert_not_called()
            self.assertEqual(storage.get_document(self.doc).status,'ready')
        with self.client() as client:
            self.assertEqual(client.get('/api/tasks/'+item['id']).json()['status'],maintenance.PENDING)

    def test_readonly_unrelated_admin_and_csrf_boundaries(self):
        with self.client('viewer') as client:
            self.assertFalse(client.get(self.inspect_url).json()['can_request'])
            self.assertEqual(client.post(self.url,json={'expected_sha256':self.digest}).status_code,403)
        with self.client('outside') as client:
            self.assertEqual(client.get(self.inspect_url).status_code,403)
            self.assertEqual(client.post(self.url,json={'expected_sha256':self.digest}).status_code,403)
        with self.client() as client:
            del client.headers['X-CSRF-Token']
            self.assertEqual(client.post(self.url,json={'expected_sha256':self.digest}).status_code,403)

    def test_source_changed_after_preview_rejects_request_without_withdrawal(self):
        info=maintenance.inspect_document(self.owner['id'],self.doc)
        Path(storage.get_document(self.doc).original_path).write_bytes(b'changed')
        with self.assertRaises(ValueError): self.request()
        self.assertEqual(storage.get_document(self.doc).status,'ready')
        self.assertIsNotNone(info['expected_sha256'])

    def test_cancel_and_retry_do_not_launch_workers(self):
        with self.client() as client:
            self.app.state.submit=Mock(side_effect=AssertionError('worker'))
            task=client.post(self.url,json={'expected_sha256':self.digest}).json()
            self.assertEqual(client.post('/api/tasks/'+task['id']+'/cancel').status_code,200)
            self.assertEqual(store.task(task['id'])['status'],'cancelled')
            response=client.post('/api/tasks/'+task['id']+'/retry')
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['status'],maintenance.PENDING)
            self.app.state.submit.assert_not_called()

    def test_revoked_editor_cannot_retry_or_execute(self):
        with storage.connect() as conn: conn.execute("UPDATE rm_members SET role='editor' WHERE user_id=?",(self.viewer['id'],))
        task=maintenance.request_rebuild(self.viewer['id'],self.doc,self.digest)
        with storage.connect() as conn: conn.execute("UPDATE rm_members SET role='viewer' WHERE user_id=?",(self.viewer['id'],))
        with patch.object(reindex,'_rebuild_legacy_index_locked') as rebuild:
            with self.assertRaises(PermissionError): maintenance.execute_request(task['id'],expected_sha256=self.digest)
            with self.assertRaises(PermissionError): maintenance.retry_request(self.viewer['id'],task['id'])
            rebuild.assert_not_called()
        self.assertEqual(store.task(task['id'])['status'],'failed')

    def test_api_lock_prevents_offline_execution_before_task_claim(self):
        with self.client():
            task=self.request()
            with patch.object(reindex,'_rebuild_legacy_index_locked') as rebuild:
                with self.assertRaises(RuntimeError): maintenance.execute_request(task['id'],expected_sha256=self.digest)
                rebuild.assert_not_called()
            self.assertEqual(store.task(task['id'])['status'],maintenance.PENDING)

    def test_disabled_requester_and_wrong_confirmation_never_rebuild(self):
        task=self.request()
        with self.assertRaises(ValueError): maintenance.execute_request(task['id'],expected_sha256='0'*64)
        self.assertEqual(store.task(task['id'])['status'],maintenance.PENDING)
        with storage.connect() as conn: conn.execute('UPDATE rm_users SET disabled=1 WHERE id=?',(self.owner['id'],))
        with patch.object(reindex,'_rebuild_legacy_index_locked') as rebuild:
            with self.assertRaises(PermissionError): maintenance.execute_request(task['id'],expected_sha256=self.digest)
            rebuild.assert_not_called()

    def test_worker_failure_preserves_error_and_explicit_retry(self):
        task=self.request()
        with patch.object(reindex,'_rebuild_legacy_index_locked',side_effect=RuntimeError('private path')):
            with self.assertRaises(RuntimeError): maintenance.execute_request(task['id'],expected_sha256=self.digest)
        failed=store.task(task['id'])
        self.assertEqual(failed['status'],'failed')
        self.assertNotIn('private path',failed['error'])
        retried=maintenance.retry_request(self.owner['id'],task['id'])
        self.assertEqual(retried['status'],maintenance.PENDING)
        self.assertIsNone(retried['error'])

    def test_commit_before_task_result_failure_can_reconcile_without_embedding(self):
        task=self.request()
        storage.update_document(self.doc,index_version='reindex-committed',file_sha256=self.digest)
        store.update_task(task['id'],status='failed',error='result write interrupted')
        retried=maintenance.retry_request(self.owner['id'],task['id'])
        self.assertEqual(retried['status'],maintenance.PENDING)
        with patch.object(ingestion,'prepare_ingestion_batch',side_effect=AssertionError('duplicate cost')):
            completed=maintenance.execute_request(task['id'],expected_sha256=self.digest)
        self.assertEqual(completed['status'],'completed')
        self.assertEqual(completed['result']['index_version'],'reindex-committed')

    def test_pending_duplicate_does_not_expose_another_editors_task(self):
        task=self.request()
        with storage.connect() as conn: conn.execute("UPDATE rm_members SET role='editor' WHERE user_id=?",(self.viewer['id'],))
        with self.assertRaisesRegex(ValueError,'已有待处理'): maintenance.request_rebuild(self.viewer['id'],self.doc,self.digest)
        with self.assertRaises(PermissionError): store.task(task['id'],self.viewer['id'])

    def test_request_quota(self):
        for _ in range(10): store.create_task(self.owner['id'],'search',{'query':'fixture'})
        with self.assertRaises(store.TaskQueueFull): self.request()
        self.assertEqual(storage.get_document(self.doc).status,'ready')

    def test_parallel_requests_have_one_persisted_identity(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            items=list(pool.map(lambda _:self.request(),range(4)))
        self.assertEqual(len({item['id'] for item in items}),1)

    def test_partial_identity_cannot_offer_rebuild(self):
        storage.update_document(self.doc,file_sha256=self.digest)
        info=maintenance.inspect_document(self.owner['id'],self.doc)
        self.assertFalse(info['can_request'])
        with self.assertRaises(ValueError): self.request()

    def test_pending_maintenance_counts_toward_other_queue_capacity(self):
        self.request()
        for i in range(9): store.create_task(self.owner['id'],'search',{'query':str(i)})
        with self.assertRaises(store.TaskQueueFull): store.create_task(self.owner['id'],'search',{'query':'extra'})
