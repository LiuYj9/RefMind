import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import httpx
from refmind.llm.openai_metering import MeteredTransport, billing_stage, connect


class BillingStageTest(unittest.TestCase):
    def test_concurrent_stages_do_not_leak_into_other_calls_or_change_budget(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'OPENAI_USAGE_DB': str(Path(folder)/'ledger.db'), 'OPENAI_DAILY_BUDGET_USD': '2', 'OPENAI_TOTAL_BUDGET_USD': '10'}):
            transport = MeteredTransport('chat')
            transport.inner.close()
            transport.inner = httpx.MockTransport(lambda request: httpx.Response(200, request=request, json={'usage': {'prompt_tokens': 10, 'completion_tokens': 1}}))
            def request():
                transport.handle_request(httpx.Request('POST', 'https://example.test', json={'model': 'gpt-5.4-mini', 'messages': [], 'max_completion_tokens': 100}))
            def run(stage):
                with billing_stage(stage):
                    request()
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(run, ['generation', 'evidence_review']))
            try:
                with billing_stage('should_not_persist'):
                    raise RuntimeError('aborted before call')
            except RuntimeError:
                pass
            request()
            with connect() as connection:
                rows = connection.execute('SELECT role,status,cost FROM calls').fetchall()
            self.assertEqual({r[0] for r in rows}, {'generation', 'evidence_review', 'chat'})
            self.assertTrue(all(r[1] == 'ok' and r[2] > 0 for r in rows))
            transport.close()
