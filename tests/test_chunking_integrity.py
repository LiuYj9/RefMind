"""分块证据完整性与定位的离线回归，不调用模型或写知识库。"""
import json
import unittest

from refmind.rag.layout_chunker import layout_chunks
from refmind.rag.document_processor import parsed_to_documents


def block(identity, text, kind="paragraph", page=1, end=None):
    return {"block_id": identity, "text": text, "type": kind,
            "page": page, "page_end": end if end is not None else page}


class ChunkIntegrityTests(unittest.TestCase):
    def split(self, blocks):
        return layout_chunks(blocks, 1000, 1800, 200)

    def test_large_html_table_is_one_unchanged_chunk(self):
        text = '<table><tr><th>A</th></tr>' + '<tr><td rowspan="2">10</td></tr>' * 150 + '</table>'
        chunks = self.split([block("t", text, "table", 2, 3)])
        self.assertEqual([c.text for c in chunks], [text])
        self.assertTrue(chunks[0].is_atomic)
        self.assertTrue(chunks[0].oversized)
        self.assertEqual((chunks[0].page_start, chunks[0].page_end), (2, 3))

    def test_large_latex_equation_remains_balanced_and_whole(self):
        text = r"\begin{aligned}" + r"x_{i} &= y_{i} + z_{i} \\" * 150 + r"\end{aligned}"
        chunks = self.split([block("e", text, "equation")])
        self.assertEqual([c.text for c in chunks], [text])
        self.assertTrue(chunks[0].oversized)

    def test_definition_is_attached_without_duplicate_indexing(self):
        for definition in ("where x is voltage.", "其中，x 是电压。", "式中，x 为电压。", "In which x is voltage."):
            with self.subTest(definition=definition):
                chunks = self.split([block("e", "x=IR", "equation"), block("d", definition)])
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0].text, "x=IR\n\n" + definition)
                self.assertEqual(chunks[0].block_ids, ["e", "d"])

    def test_cross_page_definition_has_separate_source_locations(self):
        docs = parsed_to_documents({"blocks": [block("e", "x=IR", "equation", 2),
                                   block("d", "where R is resistance.", page=3)]}, 1, "p.pdf")
        self.assertEqual(len(docs), 1)
        meta = docs[0].metadata
        self.assertEqual((meta["page_start"], meta["page_end"], meta["citation_scope"]), (2, 3, "range"))
        self.assertEqual(json.loads(meta["source_locations"]), [
            {"block_id": "e", "page_start": 2, "page_end": 2},
            {"block_id": "d", "page_start": 3, "page_end": 3}])

    def test_explanation_is_not_taken_across_heading_table_or_equation(self):
        for kind in ("heading", "table", "equation"):
            with self.subTest(kind=kind):
                chunks = self.split([block("e", "x=IR", "equation"), block("boundary", "Boundary", kind),
                                     block("d", "where x is voltage.")])
                original = next(c for c in chunks if "e" in c.block_ids)
                self.assertEqual(original.text, "x=IR")

    def test_unrelated_distant_or_long_text_is_not_attached(self):
        for following in (block("d", "Results show improvement."),
                          block("d", "where x is voltage.", page=3),
                          block("d", "where " + "x" * 700)):
            with self.subTest(following=following):
                chunks = self.split([block("e", "x=IR", "equation"), following])
                self.assertEqual(chunks[0].text, "x=IR")

    def test_only_first_definition_paragraph_is_attached(self):
        chunks = self.split([block("e", "x=IR", "equation"), block("d", "where x is voltage."),
                             block("next", "where the next experiment begins.")])
        self.assertEqual(chunks[0].block_ids, ["e", "d"])
        self.assertEqual(chunks[1].block_ids, ["next"])

    def test_normal_text_keeps_page_boundaries(self):
        chunks = self.split([block("a", "A" * 400, page=2), block("b", "B" * 400, page=3)])
        self.assertEqual([(c.page_start, c.page_end) for c in chunks], [(2, 2), (3, 3)])

    def test_long_plain_text_still_uses_langchain_overlap_and_has_no_lost_characters(self):
        text = "".join(chr(0x4e00 + i) for i in range(4000))
        chunks = self.split([block("a", text, page=2)])
        self.assertEqual([len(c.text) for c in chunks], [1000, 1000, 1000, 1000, 800])
        self.assertEqual(chunks[0].text[-200:], chunks[1].text[:200])
        self.assertEqual(chunks[0].text + "".join(chunk.text[200:] for chunk in chunks[1:]), text)
        self.assertTrue(all((c.page_start, c.page_end) == (2, 2) for c in chunks))

    def test_upstream_unknown_cross_page_offsets_stay_a_range(self):
        chunks = self.split([block("a", "A" * 3000, page=2, end=3)])
        self.assertTrue(all((c.page_start, c.page_end) == (2, 3) for c in chunks))

    def test_legacy_tables_also_stay_whole(self):
        text = "Table: measurements\n" + "| A | 10 |\n" * 500
        docs = parsed_to_documents({"pages": [], "tables": [{"table_id": "t", "normalized_text": text,
                                   "page_start": 2, "page_end": 3}]}, 1, "p.pdf")
        self.assertEqual([d.page_content for d in docs], [text])
        self.assertTrue(docs[0].metadata["is_atomic"])
        self.assertTrue(docs[0].metadata["oversized"])

    def test_multiblock_equation_does_not_inherit_misleading_single_bbox(self):
        equation = block("e", "x=IR", "equation")
        equation["bbox"] = [1, 2, 3, 4]
        docs = parsed_to_documents({"blocks": [equation, block("d", "where R is resistance.")]}, 1, "p.pdf")
        self.assertEqual(docs[0].metadata["bbox"], "[]")
        self.assertTrue(all(isinstance(v, (str, int, float, bool)) for v in docs[0].metadata.values()))


if __name__ == "__main__":
    unittest.main()
