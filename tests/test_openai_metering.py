import os,tempfile,unittest,sqlite3,json
from pathlib import Path
from unittest.mock import patch
import httpx
from refmind.llm.openai_metering import MeteredTransport,BudgetExceeded,connect

class MeterTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory()
  self.env=patch.dict(os.environ,{"OPENAI_USAGE_DB":str(Path(self.tmp.name)/"usage.db"),"OPENAI_TOTAL_BUDGET_USD":"10","OPENAI_DAILY_BUDGET_USD":"2"})
  self.env.start()
 def tearDown(self):
  self.env.stop();self.tmp.cleanup()
 def transport(self,code=200,usage=True):
  t=MeteredTransport("test")
  t.inner.close()
  t.inner=httpx.MockTransport(lambda r:httpx.Response(code,json={"usage":{"prompt_tokens":100,"completion_tokens":10}} if usage else {},request=r))
  return t
 def call(self,t):
  return t.handle_request(httpx.Request("POST","https://example.test/v1/chat/completions",json={"model":"gpt-5.4-mini","messages":[],"max_completion_tokens":100}))
 def test_success_records_tokens(self):
  self.call(self.transport())
  with connect() as c: row=c.execute("SELECT status,cost,input_tokens,output_tokens FROM calls").fetchone()
  self.assertEqual(row[0],"ok");self.assertAlmostEqual(row[1],.00012);self.assertEqual(row[2:],(100,10))
 def test_budget_blocks_before_network(self):
  with patch.dict(os.environ,{"OPENAI_TOTAL_BUDGET_USD":"0"}):
   with self.assertRaises(BudgetExceeded):self.call(self.transport())
 def test_unknown_usage_retains_reservation(self):
  self.call(self.transport(500,False))
  with connect() as c: row=c.execute("SELECT cost,status,reserved FROM calls").fetchone()
  self.assertIsNone(row[0]);self.assertEqual(row[1],"http_error");self.assertGreater(row[2],0)
 def test_network_failure_remains_charged_conservatively(self):
  t=self.transport();t.inner=httpx.MockTransport(lambda r:(_ for _ in ()).throw(httpx.ConnectError("failure")))
  with self.assertRaises(httpx.ConnectError):self.call(t)
  with connect() as c:self.assertEqual(c.execute("SELECT status FROM calls").fetchone()[0],"unknown")
 def test_unpriced_model_denied(self):
  with self.assertRaises(ValueError):self.transport().handle_request(httpx.Request("POST","https://example.test",json={"model":"unknown"}))
 def test_old_embedding_store_protected(self):
  from refmind.rag.document_processor import get_vectorstore
  with patch.dict(os.environ,{"EMBEDDING_PROVIDER":"openai","REFMIND_EVAL_WORKSPACE":""}):
   with self.assertRaises(ValueError):get_vectorstore(1)

 def test_proxy_config_does_not_conflict_with_supplied_client(self):
  from refmind.llm.factory import _openai_chat
  with patch.dict(os.environ,{"OPENAI_API_KEY":"sk-test-not-real","OPENAI_PROXY":"http://127.0.0.1:7890"}):
   model=_openai_chat("test")
   self.assertEqual(model.model_name,"gpt-5.4-mini-2026-03-17")
   model.http_client.close()
 def test_compressed_response_not_decoded_twice(self):
  import gzip
  t=self.transport()
  raw=json.dumps({"usage":{"prompt_tokens":100,"completion_tokens":10},"content":"ok"}).encode()
  t.inner=httpx.MockTransport(lambda r:httpx.Response(200,headers={"Content-Encoding":"gzip"},content=gzip.compress(raw),request=r))
  response=self.call(t)
  self.assertEqual(response.json()["content"],"ok")
 def test_concurrent_requests_cannot_bypass_reservation(self):
  from concurrent.futures import ThreadPoolExecutor
  with patch.dict(os.environ,{"OPENAI_TOTAL_BUDGET_USD":"0.015"}):
   def invoke(_):
    try:self.call(self.transport(500,False));return True
    except BudgetExceeded:return False
   with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(invoke,range(8)))
  self.assertEqual(sum(results),1)

 def test_fixed_refusal_with_references_is_recognized(self):
  from evaluation.classification import classify_refusal
  self.assertTrue(classify_refusal("NO EVIDENCE。\n\n---\n\n### 参考来源\nsource",("NO EVIDENCE",)))
 def test_partial_refusal_remains_unclassified(self):
  from evaluation.classification import classify_refusal
  self.assertIsNone(classify_refusal("Some supported facts. NO EVIDENCE",("NO EVIDENCE",)))
 def test_pure_text_evaluation_cannot_load_images(self):
  from types import SimpleNamespace
  from unittest.mock import Mock
  from evaluation.worker import configure_text_only
  from refmind.llm import image_understanding
  original=Mock(side_effect=AssertionError("image loading must not happen"))
  graph=SimpleNamespace(build_visual_content=original)
  with (patch.dict(os.environ,{"REFMIND_EVAL_TEXT_ONLY":"true"}),
        patch.object(image_understanding,"build_visual_content",original)):
   configure_text_only(graph)
   self.assertEqual(graph.build_visual_content([{"image_path":"private.png"}]),[])
  original.assert_not_called()

 def test_text_only_covers_both_reviewers_and_their_refusal_confirmation(self):
  from unittest.mock import Mock
  from evaluation.worker import configure_text_only
  from langchain_core.documents import Document
  from refmind import llm as llm_api
  from refmind.agents.orchestration import AnswerReviewAgent,EvidenceReviewAgent,MultiAgentConfig
  from refmind.llm import image_understanding
  from refmind.rag import graph

  document=Document(page_content="Synthetic measurement.",metadata={
      "image_path":"never-open-fixture.png","_evidence_pack_complete":True,"evidence_level":"full_text"})
  insufficient={"answerable":False,"coverage":"insufficient"}
  responses=[insufficient,{"answerable":True,"coverage":"direct","keep":[1]},
             insufficient,{"answerable":True,"coverage":"direct","answer":"Reviewed."}]
  config=MultiAgentConfig(enable_evidence_review=True,enable_answer_review=True,confirm_refusal=True)
  for flag in (None,"false","true"):
   with self.subTest(text_only=flag):
    generation_builder=Mock(return_value=[{"type":"image_url","image_url":{"url":"synthetic"}}])
    review_builder=Mock(return_value=[{"type":"image_url","image_url":{"url":"synthetic"}}])
    text_model=Mock()
    visual_model=Mock()
    text_model.invoke.side_effect=[json.dumps(response) for response in responses]
    visual_model.invoke.side_effect=[json.dumps(response) for response in responses]
    with (patch.dict(os.environ),patch.object(graph,"build_visual_content",generation_builder),
          patch.object(image_understanding,"build_visual_content",review_builder),
          patch.object(llm_api,"get_multimodal_llm",return_value=visual_model) as visual_factory):
     if flag is None:
      os.environ.pop("REFMIND_EVAL_TEXT_ONLY",None)
     else:
      os.environ["REFMIND_EVAL_TEXT_ONLY"]=flag
     configure_text_only(graph)
     selected,evidence_warning=EvidenceReviewAgent(text_model,config).review("Question",[document])
     answer,answer_warning=AnswerReviewAgent(text_model,config).review("Question",selected,"Draft.")
     self.assertEqual(selected,[document])
     self.assertEqual(answer,"Reviewed.")
     self.assertIn("二次复核",evidence_warning)
     self.assertIn("二次复核",answer_warning)
     if flag=="true":
      self.assertEqual(graph.build_visual_content([document]),[])
      generation_builder.assert_not_called()
      self.assertEqual(text_model.invoke.call_count,4)
      self.assertTrue(all(isinstance(call.args[0],str) for call in text_model.invoke.call_args_list))
      review_builder.assert_not_called()
      visual_factory.assert_not_called()
      visual_model.invoke.assert_not_called()
     else:
      self.assertIs(graph.build_visual_content,generation_builder)
      self.assertIs(image_understanding.build_visual_content,review_builder)
      text_model.invoke.assert_not_called()
      self.assertEqual(review_builder.call_count,4)
      self.assertEqual(visual_factory.call_count,4)
      self.assertEqual(visual_model.invoke.call_count,4)

 def test_fake_workspace_does_not_allow_daily_index(self):
  from refmind.rag.document_processor import get_vectorstore
  with patch.dict(os.environ,{"EMBEDDING_PROVIDER":"openai","REFMIND_EVAL_WORKSPACE":self.tmp.name}):
   with self.assertRaises(ValueError):get_vectorstore(1)
