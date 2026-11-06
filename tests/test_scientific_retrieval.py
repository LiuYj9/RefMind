import unittest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from refmind.rag.scientific_retrieval import scientific_aliases, retrieve_for_task


class ScientificRetrievalTests(unittest.TestCase):
    def test_spaced_and_joined_units_match_without_unit_conversion(self):
        self.assertEqual(scientific_aliases("77 K 34 nΩ 1e-3 Ω I_c"), scientific_aliases("77K 34nΩ 0.001Ω I_{c}"))
        self.assertNotEqual(scientific_aliases("1 mA"), scientific_aliases("1 A"))

    def test_neighbors_cannot_cross_paper_or_section(self):
        docs = [Document(page_content=str(i), metadata={"doc_id": doc, "group_id": 1, "chunk_index": i, "section": section, "version": "a"}) for i, doc, section in [(0, 1, "method"), (1, 1, "method"), (2, 1, "results"), (0, 2, "method")]]
        with patch("refmind.rag.retrieval.get_retriever", return_value=Mock(invoke=Mock(return_value=[docs[1]]))), patch("refmind.rag.document_processor.load_group_documents", return_value=docs), patch("refmind.rag.retrieval.filter_ready_documents", side_effect=lambda g, values, ids: values):
            result = retrieve_for_task(1, "q", selected_doc_ids=(1,))
        self.assertEqual([d.page_content for d in result], ["1", "0"])

    def test_summary_uses_distinct_sections(self):
        docs = [Document(page_content=str(i), metadata={"doc_id": 1, "group_id": 1, "chunk_index": i, "section": s}) for i, s in enumerate(["Introduction", "Introduction", "Methods", "Results", "References"])]
        with patch("refmind.rag.retrieval.get_retriever", return_value=None), patch("refmind.rag.document_processor.load_group_documents", return_value=docs), patch("refmind.rag.retrieval.filter_ready_documents", side_effect=lambda g, values, ids: values):
            result = retrieve_for_task(1, "summary", task_type="summary")
        self.assertEqual([d.page_content for d in result], ["0", "2", "3"])
