import unittest
from copy import deepcopy

from langchain_core.documents import Document

from refmind.citations import build_citation_label
from refmind.rag.evidence_bundle import pack_evidence


INTRO = "（以下仅展示选中原始行；其余行未纳入本轮证据）"
HEADER = "Table 1\n| Sample | Current (A) |\n|---|---|"
FOOTNOTE = "Note: all measurements at 77 K."
TARGET = "| target | 1200.0000 |"
SHORT = "| X | 1 |"
QUESTION = "target"
OUTPUT_RESERVE = 128
IMAGE_RESERVE = 4096


def table(rows, **metadata):
    return Document(page_content="\n".join([HEADER, *rows, FOOTNOTE]), metadata={
        "content_type": "table", "is_atomic": True, "chunk_id": "table", **metadata,
    })


def cost(doc, position, content=None):
    return len(build_citation_label(doc, position) + "\n" + (doc.page_content if content is None else content)) + 8


def row_view(row):
    return "\n".join([INTRO, HEADER, row, FOOTNOTE])


def pack(docs, capacity, *, evidence_tokens=2000):
    return pack_evidence(QUESTION, docs, evidence_tokens=evidence_tokens,
                         model_context_tokens=len(QUESTION) + 512 + OUTPUT_RESERVE + capacity,
                         output_reserve=OUTPUT_RESERVE, counter=len)


class TableViewBudgetTest(unittest.TestCase):
    def assert_within_budgets(self, bundle):
        trace = bundle.trace
        self.assertEqual(trace["evidence_tokens"], sum(cost(doc, i) for i, doc in enumerate(bundle.documents, 1)))
        self.assertEqual(trace["image_token_reserve"], IMAGE_RESERVE * sum(bool(doc.metadata.get("image_path")) for doc in bundle.documents))
        self.assertLessEqual(trace["evidence_tokens"], trace["evidence_budget"])
        self.assertLessEqual(sum(trace[key] for key in ("prompt_history_tokens", "evidence_tokens", "image_token_reserve", "output_reserve")),
                             trace["model_context_tokens"])

    def test_earlier_image_limits_table_rows_without_losing_the_matching_row(self):
        figure = Document(page_content="Figure 1", metadata={"image_path": "figure.png"})
        doc = table([SHORT] * 12 + [TARGET])
        capacity = cost(figure, 1) + IMAGE_RESERVE + cost(doc, 2, row_view(TARGET))
        bundle = pack([figure, doc], capacity)

        self.assert_within_budgets(bundle)
        self.assertEqual(len(bundle.documents), 2)
        self.assertEqual(bundle.documents[1].page_content, row_view(TARGET))

    def test_table_image_cannot_be_admitted_when_even_one_row_does_not_fit(self):
        doc = table([TARGET] * 12, image_path="table.png")
        fallback = Document(page_content="Independent supporting passage")
        capacity = IMAGE_RESERVE + cost(doc, 1, row_view(TARGET)) - 1
        bundle = pack([doc, fallback], capacity)

        self.assert_within_budgets(bundle)
        self.assertEqual([d.page_content for d in bundle.documents], [fallback.page_content])
        self.assertEqual(bundle.trace["omitted"][0]["chunk_id"], "table")

    def test_table_image_reservation_is_retained_when_a_view_fits(self):
        doc = table([SHORT] * 12 + [TARGET], image_path="table.png")
        capacity = IMAGE_RESERVE + cost(doc, 1, row_view(TARGET))
        bundle = pack([doc], capacity)

        self.assert_within_budgets(bundle)
        self.assertEqual(len(bundle.documents), 1)
        self.assertEqual(bundle.documents[0].page_content, row_view(TARGET))
        self.assertEqual(bundle.documents[0].metadata["image_path"], "table.png")

    def test_tenth_dynamic_citation_uses_its_actual_width(self):
        earlier = [Document(page_content=f"Earlier evidence {i}") for i in range(9)]
        doc = table([TARGET, SHORT] + ["| large | " + "9" * 90 + " |"] * 3,
                    resource_id="session-paper", evidence_level="full_text")
        prior_cost = sum(cost(d, i) for i, d in enumerate(earlier, 1))
        self.assertEqual(cost(doc, 10, row_view(TARGET)), cost(doc, 1, row_view(TARGET)) + 1)

        for slack, expected in ((-1, SHORT), (0, TARGET)):
            with self.subTest(slack=slack):
                budget = prior_cost + cost(doc, 10, row_view(TARGET)) + slack
                bundle = pack([*earlier, doc], 5000, evidence_tokens=budget)

                self.assert_within_budgets(bundle)
                self.assertEqual(len(bundle.documents), 10)
                self.assertEqual(bundle.documents[-1].page_content, row_view(expected))

    def test_text_view_keeps_original_headers_footnotes_and_input(self):
        doc = table([SHORT] * 12 + [TARGET], source_locations=[{"page": 4}])
        original = deepcopy(doc.model_dump())
        budget = cost(doc, 1, row_view(TARGET))
        bundle = pack([doc], 5000, evidence_tokens=budget)

        self.assert_within_budgets(bundle)
        self.assertEqual(len(bundle.documents), 1)
        view = bundle.documents[0]
        self.assertEqual(view.page_content, row_view(TARGET))
        self.assertEqual(view.metadata["source_text"], doc.page_content)
        self.assertEqual(view.metadata["source_rows"], [16])
        self.assertEqual(view.metadata["evidence_view"], "explicit_table_rows")
        self.assertEqual(doc.model_dump(), original)


if __name__ == "__main__":
    unittest.main()
