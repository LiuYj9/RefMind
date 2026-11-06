"""Distinct external, multi-turn and structural metrics using synthetic fixtures only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock

from evaluation.dataset import GoldenItem,GoldenSet
from evaluation.run_suite import provider_replay_component,load_replay
from evaluation.suites import external_ranking,evaluate_external,evaluate_dialogues,structure_scores,evaluate_structure
from evaluation.review import apply_decisions, main as review_main
from evaluation.runtime import parse_overrides,effective_overrides


def golden(track,metadata,*,count=1):
    items=tuple(GoldenItem(f"case{i}","synthetic fixture question "+str(i),"synthetic expected result",(),
        track=track,domain="test_fixture",family_ids=(f"fixture{i}",),annotation_status="draft",
        task_metadata=metadata) for i in range(count))
    return GoldenSet((),items,0,(),"fixture-hash",3)


class ExternalSuiteTests(unittest.TestCase):
    def test_qrels_exact_work_aliases_and_duplicates(self):
        gold={"qrels":{"10.1000/a":2,"10.1000/b":0},"required_work_ids":["10.1000/a"]}
        papers=[{"doi":"https://doi.org/10.1000/A"},{"doi":"10.1000/a"},{"doi":"10.1000/b"}]
        score=external_ranking(papers,gold,(1,2))
        self.assertEqual(score["by_k"]["@1"]["known_recall"],1)
        self.assertAlmostEqual(score["duplicate_rate"],1/3)
        self.assertEqual(score["by_k"]["@2"]["precision"],.5)

    def test_unjudged_results_do_not_become_false_negatives(self):
        score=external_ranking([{"doi":"10.1000/unknown"},{"doi":"10.1000/a"}],
                               {"qrels":{"10.1000/a":2}},(2,))
        self.assertIsNone(score["by_k"]["@2"]["precision"])
        self.assertIsNone(score["by_k"]["@2"]["ndcg"])
        self.assertEqual(score["by_k"]["@2"]["known_recall"],1)

    def test_metadata_or_unknown_identity_does_not_count_as_fulltext(self):
        data=golden("external",{"external_gold":{"required_work_ids":["10.1000/a"],"fulltext_sha256":"a"*64}})
        result={"papers":[{"doi":"10.1000/a"}],"status":"discovered",
                "fetch":{"identity_status":"unknown","sha256":"a"*64},
                "parse":{"status":"ready","source_sha256":"a"*64},
                "answer":{"evidence_level":"abstract","documents":[]}}
        report=evaluate_external(data,lambda request:result,allow_draft=True)
        self.assertFalse(report["items"][0]["fulltext"]["fulltext_stage_complete"])
        self.assertEqual(report["items"][0]["ranking"]["by_k"]["@1"]["required_work_recall"],1)

    def test_actual_production_discovery_retains_metadata_without_abstract(self):
        case={"parameters":{"doi":"10.1000/example","provider":"crossref"},
              "provider_payloads":{"crossref":{"message":{"DOI":"10.1000/example","title":["Synthetic fixture paper"],
                 "published":{"date-parts":[[2024]]},"type":"journal-article"}}}}
        component=provider_replay_component({"q":case})
        result=component({"id":"q","query":"Find the fixture DOI"})
        self.assertFalse(result["failed"])
        self.assertEqual(len(result["papers"]),1)
        self.assertEqual(result["papers"][0]["abstract"],"")

    def test_replay_dataset_hash_must_match(self):
        data=golden("external",{"external_gold":{"required_work_ids":["doi:a"]}})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"replay.json"
            path.write_text(json.dumps({"track":"external","dataset_sha256":"wrong","cases":{"case0":{}}}))
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                load_replay(path,data,"external")


class DialogueSuiteTests(unittest.TestCase):
    def test_four_turn_groups_reset_state_and_never_receive_gold(self):
        turns=[{"question":f"fixture turn {i}","expected":{"status":"answered","state":{"turn_count":i+1}}} for i in range(4)]
        data=golden("dialogue",{"turns":turns},count=2)
        calls=[]
        def component(request,state):
            self.assertNotIn("expected",request)
            calls.append((request["scenario_id"],state.get("turn_count",0)))
            return {"status":"answered","state":{"turn_count":state.get("turn_count",0)+1},"documents":[]}
        report=evaluate_dialogues(data,component,allow_draft=True)
        self.assertEqual(report["summary"]["groups"],2)
        self.assertEqual(report["summary"]["turns"],8)
        self.assertEqual(report["summary"]["group_pass_rate"],1)
        self.assertEqual(calls[4],("case1",0))

    def test_one_failed_turn_fails_whole_group_but_runs_four(self):
        turns=[{"question":str(i),"expected":{"status":"answered"}} for i in range(4)]
        def component(request,state):
            if request["turn_index"]==1:
                raise RuntimeError("private provider failure")
            return {"status":"answered"}
        report=evaluate_dialogues(golden("dialogue",{"turns":turns}),component,allow_draft=True)
        self.assertEqual(len(report["items"][0]["turns"]),4)
        self.assertEqual(report["summary"]["group_pass_rate"],0)
        self.assertNotIn("private provider failure",json.dumps(report))


class StructureSuiteTests(unittest.TestCase):
    def test_partial_regions_do_not_count_unannotated_objects_as_false_positives(self):
        gold={"annotation_coverage":"partial_regions","objects":[{"id":"a","type":"table","bbox":[0,0,100,100],"bbox_coordinate_system":"normalized_1000"}]}
        score=structure_scores([dict(gold["objects"][0]),{"type":"text"}],gold)
        self.assertEqual(score["by_type"]["table"]["recall"],1)
        self.assertIsNone(score["by_type"]["table"]["precision"])
        self.assertIsNone(score["by_type"]["text"]["f1"])

    def test_coordinate_conversion_and_reversed_reading_order(self):
        gold={"page_size":[500,800],"objects":[
            {"id":"a","type":"text","bbox":[0,0,500,500],"bbox_coordinate_system":"normalized_1000"},
            {"id":"b","type":"equation","bbox":[0,500,500,1000],"bbox_coordinate_system":"normalized_1000","text":"E = mc^2"}],
            "reading_order":["a","b"]}
        predicted=[{"type":"text","bbox":[0,0,250,400],"bbox_coordinate_system":"pdf_points","reading_order":2},
                   {"type":"equation","bbox":[0,400,250,800],"bbox_coordinate_system":"pdf_points","reading_order":1,"text":"E=mc^2"}]
        scores=structure_scores(predicted,gold)
        self.assertEqual(scores["by_type"]["equation"]["recall"],1)
        self.assertEqual(scores["reading_order"]["accuracy"],0)
        self.assertEqual(scores["equation_literal_match"]["rate"],1)

    def test_missing_objects_are_not_dropped_from_reading_order_denominator(self):
        gold={"objects":[{"id":"a","type":"text","bbox":[0,0,100,100],"bbox_coordinate_system":"normalized_1000"},
                         {"id":"b","type":"text","bbox":[0,200,100,300],"bbox_coordinate_system":"normalized_1000"}],
              "reading_order":["a","b"]}
        score=structure_scores([],gold)
        self.assertEqual(score["reading_order"]["accuracy"],0)
        self.assertEqual(score["by_type"]["text"]["recall"],0)


class ReviewToolsTests(unittest.TestCase):
    def test_assignment_exports_pending_decisions_and_csv_without_approval(self):
        try:
            from tests.test_evaluation_v3 import fixture,write
        except ImportError:
            from test_evaluation_v3 import fixture,write
        from evaluation.dataset import load_golden_set
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=write(root,fixture(root));data=load_golden_set(source,project_root=root)
            packet=root/'packet.json';table=root/'tasks.csv'
            with patch('evaluation.review.load_golden_set',return_value=data),patch('builtins.print'):
                review_main(['export','--golden',str(source),'--output',str(packet),'--task-table',str(table),'--assigned-reviewer','雪蝉999'])
            content=json.loads(packet.read_text(encoding='utf-8'))
            self.assertEqual(content['assigned_reviewer'],'雪蝉999')
            self.assertIsNone(content['reviewed_at'])
            self.assertEqual(content['decisions'][0]['status'],'pending')
            self.assertEqual(content['sources'][0]['id'],'p1')
            self.assertTrue(table.exists())

    def test_apply_requires_exact_dataset_hash_and_creates_audit(self):
        # Synthetic fixture only; this does not approve any real RefMind annotation.
        try:
            from tests.test_evaluation_v3 import fixture,write
        except ImportError:
            from test_evaluation_v3 import fixture,write
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=write(root,fixture(root));before=source.read_bytes()
            decisions=root/"decisions.json";output=root/"reviewed.json"
            decisions.write_text(json.dumps({"dataset_sha256":hashlib.sha256(before).hexdigest(),"reviewer":"fixture reviewer",
                "reviewed_at":"2026-11-05","decisions":[{"id":"q1","status":"approved"}]}))
            result=apply_decisions(source,decisions,output,project_root=root)
            self.assertTrue(Path(result["audit"]).exists())
            self.assertEqual(source.read_bytes(),before)
            with self.assertRaises(ValueError):
                apply_decisions(source,decisions,output,project_root=root)


class RuntimeTests(unittest.TestCase):
    def test_storage_and_credentials_cannot_be_overridden(self):
        for argument in ("DATABASE_PATH=somewhere","DASHSCOPE_API_KEY=secret","REFMIND_MCP_SERVERS=[]"):
            with self.assertRaises(ValueError):
                parse_overrides([argument])

    def test_overrides_must_be_consumed_not_just_echoed(self):
        overrides=parse_overrides(["CONTEXT_MAX_TOKENS=6000","MULTI_AGENT_CONFIRM_REFUSAL=false"])
        effective=effective_overrides(SimpleNamespace(context_max_tokens=6000,refusal_confirmation_enabled=False),overrides)
        self.assertFalse(effective["MULTI_AGENT_CONFIRM_REFUSAL"]["effective"])
        with self.assertRaisesRegex(ValueError,"not consumed"):
            effective_overrides(SimpleNamespace(context_max_tokens=4000,refusal_confirmation_enabled=False),overrides)

    def test_research_dialogue_uses_shared_wrapper_and_message_history(self):
        from evaluation.worker import dialogue_component
        from langchain_core.messages import HumanMessage,AIMessage
        memory=Mock(session_id=17)
        with patch("refmind.storage.create_session",return_value=SimpleNamespace(id=17)), \
             patch("refmind.rag.memory.RelevantMemory",return_value=memory), \
             patch("refmind.rag.graph.answer_research_question",return_value={"answer":"fixture answer","status":"answered","documents":[]}) as research, \
             patch("refmind.rag.answer_question") as legacy:
            component=dialogue_component(3,{"s1":42},pipeline="research")
            first=component({"scenario_id":"fixture","question":"first","corpus_scope":["s1"]},{})
            component({"scenario_id":"fixture","question":"follow-up","corpus_scope":["s1"]},first["state"])
            legacy.assert_not_called()
            self.assertEqual(research.call_count,2)
            self.assertIsInstance(research.call_args.kwargs["history"][0],HumanMessage)
            self.assertIsInstance(research.call_args.kwargs["history"][1],AIMessage)


if __name__=="__main__":
    unittest.main()
