import unittest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from refmind.rag.evidence_bundle import paired_history, describe_task, pack_evidence


class EvidenceBundleTest(unittest.TestCase):
    def test_history_never_keeps_orphan_answer(self):
        history = [AIMessage(content="orphan"), HumanMessage(content="Q1"), AIMessage(content="A1"), HumanMessage(content="pending")]
        self.assertEqual([m.content for m in paired_history(history)], ["Q1", "A1"])

    def test_changed_condition_does_not_inherit_old_number(self):
        task = describe_task("它在 4.2 K 下是多少？", [HumanMessage(content="77 K 电流多少"), AIMessage(content="old")], ["A", "B"])
        self.assertTrue(task.needs_clarification)
        self.assertNotIn("77", task.resolved_query)

    def test_comparison_packs_each_source_before_second_chunk(self):
        docs = [Document(page_content="a" * 40, metadata={"doc_id": i, "group_id": 1, "paper_title": str(i), "chunk_id": str(j)}) for j, i in enumerate([1, 1, 2])]
        cost = lambda text: len(text)
        bundle = pack_evidence("compare", docs, task_type="comparison", evidence_tokens=180, counter=cost)
        self.assertEqual([d.metadata["doc_id"] for d in bundle.documents[:2]], [1, 2])

    def test_formula_is_not_cut_and_omission_is_reported(self):
        doc = Document(page_content="x" * 500, metadata={"content_type": "equation", "is_atomic": True})
        bundle = pack_evidence("x", [doc], evidence_tokens=100, counter=len)
        self.assertEqual(bundle.documents, [])
        self.assertEqual(bundle.trace["omitted"][0]["reason"], "budget_cannot_fit_whole_evidence")
        self.assertEqual(len(doc.page_content), 500)

    def test_table_view_keeps_actual_headers_units_and_footnote(self):
        doc = Document(page_content="Table 1\n| Sample | Current (A) |\n|---|---|\n" + "\n".join(f"| X{i} | {i} |" for i in range(20)) + "\nNote: at 77 K.", metadata={"content_type": "table", "is_atomic": True})
        bundle = pack_evidence("X19", [doc], evidence_tokens=210, counter=len)
        self.assertEqual(len(bundle.documents), 1)
        text = bundle.documents[0].page_content
        self.assertIn("Current (A)", text)
        self.assertIn("77 K", text)
        self.assertIn("X19", text)
        self.assertEqual(bundle.documents[0].metadata["source_text"], doc.page_content)

    def test_total_budget_includes_history_and_output(self):
        history = [HumanMessage(content="q" * 200), AIMessage(content="a" * 200)]
        bundle = pack_evidence("Q", [Document(page_content="evidence")], history=history, model_context_tokens=1000, output_reserve=300, evidence_tokens=150, counter=len)
        self.assertEqual(bundle.history, [])
        self.assertLessEqual(bundle.trace["prompt_history_tokens"] + bundle.trace["evidence_tokens"] + 300, 1000)


if __name__ == "__main__":
    unittest.main()
