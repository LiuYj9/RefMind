"""论文题名、稳定库内序号、完整引用与批量删除回归测试。"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from refmind import storage
from refmind.citations import (
    CitationTarget,
    build_citation_label,
    decode_citation_target,
    encode_citation_target,
    enrich_citation_metadata,
    finalize_answer_citations,
    finalize_stored_answer_citations,
    normalize_answer_citations,
    strip_citation_rendering,
)
from refmind.config import settings
from refmind.parsing import extract_paper_title
from refmind.rag.graph import format_documents
from refmind.services import ingestion


class PaperTitleTests(unittest.TestCase):
    def test_layout_heading_wins_over_upload_filename(self) -> None:
        parsed = {
            "blocks": [
                {
                    "type": "heading",
                    "text": "High-Temperature Superconducting Motors for Aviation",
                    "page": 1,
                    "reading_order": 0,
                },
                {
                    "type": "heading",
                    "text": "1 Introduction",
                    "page": 1,
                    "reading_order": 3,
                },
            ]
        }
        title = extract_paper_title(parsed, Path("missing.pdf"), "download_123.pdf")
        self.assertEqual(
            title, "High-Temperature Superconducting Motors for Aviation"
        )

    def test_first_page_text_is_used_when_structured_title_is_missing(self) -> None:
        parsed = {
            "pages": [
                {
                    "page": 1,
                    "text": (
                        "Journal of Applied Energy\n"
                        "A Semi-Superconducting Synchronous Motor Design\n"
                        "Alice Zhang and Bob Li\nAbstract"
                    ),
                }
            ]
        }
        title = extract_paper_title(parsed, Path("missing.pdf"), "scan-001.pdf")
        self.assertEqual(title, "A Semi-Superconducting Synchronous Motor Design")


class CitationTests(unittest.TestCase):
    def test_full_label_contains_paper_page_section_and_paragraph(self) -> None:
        document = Document(
            page_content="HTS motor evidence",
            metadata={
                "library_index": 4,
                "paper_title": "Semi-superconducting Motor",
                "page": 5,
                "page_end": 6,
                "section": "2 Motor Structure",
                "chunk_index": 3,
            },
        )
        label = build_citation_label(document)
        self.assertEqual(
            label,
            "[文献4《Semi-superconducting Motor》，第5–6页，§2 Motor Structure，段落4]",
        )
        formatted = format_documents([document])
        self.assertIn(label, formatted)
        self.assertNotIn("[片段1]", formatted)

    def test_old_fragment_reference_is_deterministically_expanded(self) -> None:
        document = Document(
            page_content="evidence",
            metadata={
                "library_index": 2,
                "paper_title": "HTS Motor Review",
                "page": 7,
                "paragraph_index": 11,
            },
        )
        answer = normalize_answer_citations("该电机采用超导励磁 [片段1]。", [document])
        self.assertEqual(
            answer,
            "该电机采用超导励磁 [文献2《HTS Motor Review》，第7页，段落11]。",
        )
        decorated = normalize_answer_citations(
            "该电机采用超导励磁 [片段1 | 来源: upload.pdf | 第7页]。",
            [document],
        )
        self.assertEqual(
            decorated,
            "该电机采用超导励磁 [文献2《HTS Motor Review》，第7页，段落11]。",
        )

    def test_old_vector_metadata_is_enriched_from_sqlite_document(self) -> None:
        document = Document(
            page_content="legacy",
            metadata={"doc_id": 9, "filename": "random.pdf", "chunk_index": 0},
        )
        row = SimpleNamespace(
            id=9,
            filename="random.pdf",
            paper_title="Original Paper Title",
            library_index=3,
        )
        with patch("refmind.citations.storage.list_documents", return_value=[row]):
            enriched = enrich_citation_metadata([document], 1)[0]
        self.assertEqual(enriched.metadata["paper_title"], "Original Paper Title")
        self.assertEqual(enriched.metadata["library_index"], 3)
        self.assertEqual(enriched.metadata["paragraph_index"], 1)

    def test_compact_citations_are_numbered_by_first_paper_occurrence(self) -> None:
        first = Document(
            page_content="first",
            metadata={
                "group_id": 3,
                "doc_id": 70,
                "library_index": 7,
                "paper_title": "Electromagnetic Shielding Technique",
                "page": 1,
                "paragraph_index": 4,
            },
        )
        second = Document(
            page_content="second",
            metadata={
                "group_id": 3,
                "doc_id": 20,
                "library_index": 2,
                "paper_title": "HTS Aircraft Motor",
                "page": 8,
                "paragraph_index": 9,
            },
        )
        answer = (
            f"结论A{build_citation_label(second)}"
            f"，结论B{build_citation_label(first)}。"
        )

        rendered = finalize_answer_citations(answer, [first, second])
        body, separator, references = rendered.partition("\n\n---\n\n")

        self.assertTrue(separator)
        self.assertNotIn("<!-- refmind-reference-list -->", rendered)
        self.assertIn('[[1]](?open_citation=g3-d20-p8-r9 "', body)
        self.assertIn('[[2]](?open_citation=g3-d70-p1-r4 "', body)
        self.assertLess(body.index("[1]"), body.index("[2]"))
        self.assertIn("### 参考来源", references)
        self.assertIn("文献2《HTS Aircraft Motor》", references)
        self.assertIn("文献7《Electromagnetic Shielding Technique》", references)

    def test_same_paper_reuses_number_and_aggregates_locations(self) -> None:
        documents = [
            Document(
                page_content="one",
                metadata={
                    "group_id": 1,
                    "doc_id": 7,
                    "library_index": 7,
                    "paper_title": "Rotor Windings",
                    "page": page,
                    "paragraph_index": paragraph,
                },
            )
            for page, paragraph in ((1, 4), (6, 12))
        ]
        answer = "证据" + "以及".join(
            build_citation_label(document) for document in documents
        )

        rendered = finalize_answer_citations(answer, documents)
        body, separator, references = rendered.partition("\n\n---\n\n")

        self.assertTrue(separator)
        self.assertNotIn("<!-- refmind-reference-list -->", rendered)
        self.assertEqual(body.count("[[1]]("), 2)
        self.assertNotIn("[[2]](", body)
        self.assertEqual(references.count("\n- "), 1)
        self.assertIn("第1页，段落4；第6页，段落12", references)
        self.assertEqual(finalize_answer_citations(rendered, documents), rendered)

    def test_reviewer_bare_numbers_are_rebound_to_clickable_evidence(self) -> None:
        documents = [
            Document(
                page_content="evidence",
                metadata={
                    "group_id": 5,
                    "doc_id": doc_id,
                    "library_index": library_index,
                    "paper_title": title,
                    "page": library_index,
                    "paragraph_index": 1,
                },
            )
            for doc_id, library_index, title in (
                (40, 4, "First Paper"),
                (90, 9, "Second Paper"),
            )
        ]

        rendered = finalize_answer_citations("比较结果[2][1]。", documents)
        body = rendered.partition("\n\n---\n\n")[0]

        self.assertIn('[[1]](?open_citation=g5-d90-p9-r1 "', body)
        self.assertIn('[[2]](?open_citation=g5-d40-p4-r1 "', body)

    def test_citation_target_round_trip_rejects_invalid_scope(self) -> None:
        target = CitationTarget(group_id=3, doc_id=70, page=1, paragraph=4)
        encoded = encode_citation_target(target)
        self.assertEqual(encoded, "g3-d70-p1-r4")
        self.assertEqual(decode_citation_target(encoded), target)
        self.assertIsNone(decode_citation_target("g0-d70-p1-r4"))
        self.assertIsNone(decode_citation_target("../../paper.pdf"))

    def test_saved_full_labels_are_upgraded_without_rewriting_database(self) -> None:
        row = SimpleNamespace(
            id=70,
            filename="upload.pdf",
            paper_title="Electromagnetic Shielding Technique",
            library_index=7,
        )
        legacy = (
            "旧回答"
            "[文献7《Electromagnetic Shielding Technique》，第1页，段落4]"
        )
        with patch("refmind.citations.storage.list_documents", return_value=[row]):
            rendered = finalize_stored_answer_citations(legacy, 3)

        self.assertIn('[[1]](?open_citation=g3-d70-p1-r4 "', rendered)
        self.assertIn("### 参考来源", rendered)
        self.assertEqual(legacy.count("### 参考来源"), 0)
        history_text = strip_citation_rendering(rendered)
        self.assertEqual(history_text, "旧回答[1]")
        self.assertNotIn("open_citation", history_text)

    def test_legacy_reference_marker_is_removed_from_saved_answers(self) -> None:
        legacy = (
            "结论[[1]](?open_citation=g3-d70-p1-r4 \"来源\")\n\n---\n\n"
            "<!-- refmind-reference-list -->\n"
            "### 参考来源\n\n"
            "- [[1]](?open_citation=g3-d70-p1-r4 \"来源\") 文献7《论文》"
        )

        rendered = finalize_stored_answer_citations(legacy, 3)

        self.assertNotIn("<!-- refmind-reference-list -->", rendered)
        self.assertIn("### 参考来源", rendered)
        self.assertEqual(strip_citation_rendering(legacy), "结论[1]")

    def test_existing_compact_links_gain_visible_square_brackets(self) -> None:
        old = (
            "结论[2](http://localhost:8888/?open_citation=g14-d60-p1-r2)"
            "[3](http://localhost:8888/?open_citation=g14-d57-p1-r1)"
        )

        rendered = finalize_stored_answer_citations(old, 14)

        self.assertEqual(
            rendered,
            "结论[[2]](http://localhost:8888/?open_citation=g14-d60-p1-r2)"
            "[[3]](http://localhost:8888/?open_citation=g14-d57-p1-r1)",
        )

    def test_external_academic_citation_uses_safe_paper_link_without_fake_page(self) -> None:
        document = Document(
            page_content="摘要证据",
            metadata={
                "evidence_origin": "academic_search",
                "evidence_level": "abstract",
                "paper_title": "An HTS Motor Study",
                "publication_year": 2025,
                "provider": "Semantic Scholar",
                "external_url": "https://www.semanticscholar.org/paper/demo?utm_source=api",
                "doi": "10.1000/demo",
            },
        )
        label = build_citation_label(document)
        self.assertEqual(
            label,
            "[GS文献1《An HTS Motor Study》，2025，Semantic Scholar，摘要]",
        )
        rendered = finalize_answer_citations(f"方法有效{label}。", [document])

        self.assertIn(
            "[[1]](https://www.semanticscholar.org/paper/demo?utm_source=api",
            rendered,
        )
        self.assertIn("外部论文《An HTS Motor Study》", rendered)
        self.assertNotIn("页码未知", rendered)
        self.assertEqual(strip_citation_rendering(rendered), "方法有效[1]。")

    def test_external_academic_citation_rejects_dangerous_url(self) -> None:
        document = Document(
            page_content="摘要证据",
            metadata={
                "evidence_origin": "academic_search",
                "evidence_level": "abstract",
                "paper_title": "Unsafe link",
                "provider": "Crossref",
                "external_url": "javascript:alert(1)",
            },
        )
        label = build_citation_label(document)
        rendered = finalize_answer_citations(f"结论{label}", [document])

        self.assertIn("[[1]](#参考来源", rendered)
        self.assertNotIn("javascript:", rendered)
        self.assertEqual(strip_citation_rendering(rendered), "结论[1]")


class DocumentSchemaTests(unittest.TestCase):
    def test_legacy_database_is_migrated_and_numbers_stay_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "legacy.db"
            with closing(sqlite3.connect(database)) as conn:
                conn.executescript(
                    """
                    CREATE TABLE groups (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        group_id INTEGER NOT NULL,
                        filename TEXT NOT NULL,
                        original_path TEXT,
                        parsed_json_path TEXT,
                        summary TEXT,
                        num_chunks INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE
                    );
                    INSERT INTO groups (id, name) VALUES (1, 'library');
                    INSERT INTO documents (group_id, filename, status)
                    VALUES (1, 'first.pdf', 'ready'), (1, 'second.pdf', 'ready');
                    """
                )
                conn.commit()

            with (
                patch.object(settings, "database_path", database),
                patch.object(settings, "chroma_persist_dir", root / "chroma"),
                patch.object(settings, "upload_dir", root / "uploads"),
                patch.object(settings, "parsed_dir", root / "parsed"),
                patch.object(settings, "docstore_dir", root / "docstore"),
            ):
                storage.init_db()
                original = storage.list_documents(1)
                self.assertEqual([doc.library_index for doc in original], [1, 2])
                storage.update_document(1, paper_title="First Original Title")

                replacement_id = storage.create_document(1, "first.pdf")
                third_id = storage.create_document(1, "third.pdf")
                replacement = storage.get_document(replacement_id)
                third = storage.get_document(third_id)

            self.assertEqual(replacement.library_index, 1)
            self.assertEqual(third.library_index, 3)


class BulkDocumentRemovalTests(unittest.TestCase):
    def test_batch_delete_is_scope_checked_and_failure_isolated(self) -> None:
        rows = {
            1: SimpleNamespace(id=1, group_id=7, status="ready"),
            2: SimpleNamespace(id=2, group_id=7, status="ready"),
            3: SimpleNamespace(id=3, group_id=8, status="ready"),
            4: SimpleNamespace(id=4, group_id=7, status="parsing"),
        }

        def remove(doc_id: int) -> None:
            if doc_id == 2:
                raise RuntimeError("chroma busy")

        with (
            patch.object(
                ingestion.storage,
                "get_document",
                side_effect=lambda doc_id: rows.get(doc_id),
            ),
            patch.object(ingestion, "remove_document", side_effect=remove),
        ):
            result = ingestion.remove_documents(7, [1, 2, 3, 4, 1])

        self.assertEqual(result["deleted"], [1])
        self.assertIn(2, result["failed"])
        self.assertIn(3, result["failed"])
        self.assertIn(4, result["failed"])


if __name__ == "__main__":
    unittest.main()
