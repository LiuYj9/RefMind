import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from refmind.rag import graph, research_pipeline
from refmind.rag.evidence_bundle import describe_task


D1_QUESTION = "H-A 转子优化与 Sugouchi 2 MW motor 分别解决什么设计问题？"
LLZO_QUESTION = "为什么该研究不能把Li有序化或四方晶格畸变单独当成稳定四方相的充分机制？"
PARALLEL_QUESTION = "方案甲和方案乙分别解决什么问题？"
EXPLICIT_COMPARISON = "请比较方案甲和方案乙各自解决的设计问题。"
GROUP = 7


def source_ids(numbers):
    return [f"library:{GROUP}:{number}" for number in numbers]


def classify(question, *, allowed=(1, 2, 3), required=(1, 2), resolution="selected_sources"):
    return describe_task(question, source_titles=[f"Paper {n}" for n in allowed],
                         source_ids=source_ids(allowed), required_source_ids=source_ids(required),
                         target_resolution=resolution)


class PackingReached(Exception):
    """Stop the real pipeline at packing before any model is requested."""


class ParallelComparisonIntentTest(unittest.TestCase):
    def test_d1_selection_does_not_bind_the_named_objects_to_sources(self):
        description = classify(D1_QUESTION, allowed=range(1, 8), required=(2, 6))

        self.assertEqual(description.task_type, "method")
        self.assertEqual(description.source_ids, source_ids(range(1, 8)))
        self.assertEqual(description.required_source_ids, source_ids((2, 6)))

    def test_generic_parallel_objects_do_not_create_cross_paper_intent(self):
        for question in (PARALLEL_QUESTION, "原型甲与原型乙各自面对哪些限制？"):
            with self.subTest(question=question):
                self.assertEqual(classify(question).task_type, "method")

    def test_two_selected_sources_do_not_change_a_question_about_stages(self):
        for question in ("初始化和终止阶段分别使用什么条件？", "这两篇论文的初始化和终止阶段分别使用什么条件？"):
            with self.subTest(question=question):
                description = classify(question, required=(1, 2))
                self.assertEqual(description.task_type, "method")
                self.assertEqual(description.required_source_ids, source_ids((1, 2)))

    def test_three_selected_sources_do_not_change_a_question_about_internal_components(self):
        for question in ("升温与冷却分别采用哪种控制方式？", "编码器和解码器各自如何更新参数？"):
            with self.subTest(question=question):
                description = classify(question, required=(1, 2, 3))
                self.assertEqual(description.task_type, "method")
                self.assertEqual(description.required_source_ids, source_ids((1, 2, 3)))

    def test_display_titles_do_not_bind_parallel_objects_to_source_ids(self):
        identities = ["library:7:1", "asset:7:2"]
        description = describe_task(PARALLEL_QUESTION, source_titles=["方案甲", "方案乙"], source_ids=identities,
                                    required_source_ids=identities, target_resolution="selected_sources")

        self.assertEqual(description.task_type, "method")
        self.assertEqual(description.source_ids, identities)
        self.assertEqual(description.required_source_ids, identities)

    def test_allowed_only_scope_does_not_create_targets_or_change_the_route(self):
        description = classify(D1_QUESTION, allowed=range(1, 8), required=(), resolution="unresolved_allowed_scope")

        self.assertEqual(description.task_type, "method")
        self.assertEqual(description.required_source_ids, [])
        self.assertEqual(description.source_ids, source_ids(range(1, 8)))
        self.assertEqual(description.target_resolution, "unresolved_allowed_scope")

    def test_single_duplicate_or_legacy_target_selection_does_not_enable_rule(self):
        for required in ((1,), (1, 1)):
            with self.subTest(required=required):
                description = classify(PARALLEL_QUESTION, required=required)
                self.assertEqual(description.task_type, "method")
                self.assertEqual(description.required_source_ids, source_ids((1,)))
        legacy = describe_task(PARALLEL_QUESTION, source_ids=source_ids((1, 2, 3)))
        self.assertEqual(legacy.task_type, "method")
        self.assertEqual(legacy.required_source_ids, source_ids((1, 2, 3)))

    def test_other_target_resolution_does_not_enable_rule(self):
        description = classify(PARALLEL_QUESTION, resolution="externally_bound_targets")

        self.assertEqual(description.task_type, "method")
        self.assertEqual(description.required_source_ids, source_ids((1, 2)))
        self.assertEqual(description.target_resolution, "externally_bound_targets")

    def test_process_statement_cannot_lend_its_parallel_objects_to_later_question(self):
        for boundary in ("，", ", ", "。", ". ", "；", "\n"):
            question = f"方案甲和方案乙分别完成测试{boundary}流程如何安排？"
            with self.subTest(boundary=boundary):
                self.assertEqual(classify(question).task_type, "method")
        self.assertEqual(classify("利用砂纸和酒精分别打磨与清洗的实验流程是什么？").task_type, "method")

    def test_llzo_mechanism_and_single_paper_steps_remain_method(self):
        for question in (LLZO_QUESTION, "这篇论文的初始化和终止阶段分别使用什么条件？"):
            with self.subTest(question=question):
                self.assertEqual(classify(question, allowed=(1,), required=(1,)).task_type, "method")

    def test_original_type_precedence_and_english_routing_remain(self):
        questions = (
            ("请总结甲方案和乙方案分别如何选择条件？", "summary"),
            ("表 2 中方案甲和方案乙分别解决什么问题？", "table"),
            ("方案甲和方案乙分别用多少材料？", "numeric"),
            (EXPLICIT_COMPARISON, "comparison"),
            ("请比较初始化和终止阶段各自使用的条件。", "comparison"),
            ("Compare the design of motor A and motor B.", "comparison"),
            ("What problems do motor A and motor B address respectively?", "method"),
            ("Which requirements do prototype A and prototype B each satisfy?", "method"),
        )
        for question, task_type in questions:
            with self.subTest(question=question):
                self.assertEqual(classify(question).task_type, task_type)

    def test_descriptor_preserves_query_conditions_and_source_metadata(self):
        question = "原型甲与原型乙分别如何在 4.2 K 下运行？"
        description = classify(question)

        self.assertEqual(description.to_dict(), {
            "original_question": question, "resolved_query": question, "task_type": "method",
            "conditions": ["4.2 K"], "source_titles": ["Paper 1", "Paper 2", "Paper 3"],
            "needs_clarification": False, "resolution": "original", "source_ids": source_ids((1, 2, 3)),
            "required_source_ids": source_ids((1, 2)), "target_resolution": "selected_sources",
        })

    def test_ambiguous_reference_keeps_its_clarification(self):
        question = "前者和后者分别解决什么问题？"
        description = classify(question)

        self.assertTrue(description.needs_clarification)
        self.assertEqual(description.resolution, "ambiguous_reference")
        self.assertEqual(description.original_question, question)
        self.assertEqual(description.resolved_query, question)
        self.assertEqual(description.required_source_ids, source_ids((1, 2)))

    def test_pipeline_routes_retrieval_and_packing_without_widening_scope(self):
        allowed = [1, 2, 3]
        selected = [2, 3]
        rows = {n: SimpleNamespace(id=n, group_id=GROUP, status="ready", paper_title=f"Paper {n}", filename=f"{n}.pdf")
                for n in allowed}
        docs = [Document(page_content=f"Synthetic passage {n}",
                         metadata={"group_id": GROUP, "doc_id": n, "chunk_id": str(n)}) for n in selected]
        for question, selection, expected_scope, required, task_type in (
                (PARALLEL_QUESTION, selected, tuple(selected), source_ids(selected), "method"),
                (PARALLEL_QUESTION, None, tuple(allowed), [], "method"),
                (EXPLICIT_COMPARISON, selected, tuple(selected), source_ids(selected), "comparison")):
            with self.subTest(question=question, selection=selection):
                with (patch.object(research_pipeline.storage, "get_document", side_effect=rows.get),
                      patch.object(research_pipeline, "retrieve_for_task", return_value=docs) as retrieve,
                      patch.object(research_pipeline, "rerank", return_value=docs),
                      patch.object(graph, "pack_evidence", side_effect=PackingReached) as pack,
                      patch.object(graph, "get_llm") as model):
                    with self.assertRaises(PackingReached):
                        research_pipeline.answer_research_question(question, GROUP,
                            selected_doc_ids=selection, allowed_doc_ids=allowed)
                retrieve.assert_called_once_with(GROUP, question,
                    selected_doc_ids=expected_scope, task_type=task_type)
                pack.assert_called_once()
                self.assertEqual(pack.call_args.kwargs["task_type"], task_type)
                self.assertEqual(pack.call_args.kwargs["required_source_ids"], required)
                model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
