import importlib


def test_merge_cross_page_table_fragments():
    pdf_parser = importlib.import_module("refmind.parsing.pdf_parser")

    items = [
        {"text": "| 列1 | 列2 |\n| --- | --- |", "page_idx": 0},
        {"text": "| A | B |\n| C | D |", "page_idx": 1},
        {"text": "普通正文", "page_idx": 2},
    ]

    blocks = pdf_parser._normalize_mineru_items(items)

    assert len(blocks) == 2
    assert "| 列1 | 列2 |" in blocks[0]["text"]
    assert "| A | B |" in blocks[0]["text"]
    assert "| C | D |" in blocks[0]["text"]
    assert blocks[0]["page"] == 1
    assert blocks[1]["text"] == "普通正文"
    assert blocks[1]["page"] == 3
