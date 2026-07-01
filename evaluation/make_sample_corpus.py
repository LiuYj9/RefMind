"""生成一份内容与页码已知的合成论文 PDF，让评测流程不依赖真实 PDF 即可跑通。

每页只放一个唯一事实点，方便标注证据页。真实评测请换成自己的论文与人工标注。
"""

from __future__ import annotations

from pathlib import Path

# 每页一个主题，事实点唯一且明确，便于标注证据页
PAGES: list[tuple[str, str]] = [
    (
        "RefMindNet: A Hybrid Retrieval-Augmented Reading Assistant",
        "Abstract. We present RefMindNet, a retrieval-augmented system for reading "
        "scientific papers. On the SciQA benchmark, RefMindNet improves Top-3 "
        "retrieval accuracy by 12.5 percent over a dense-only baseline. The system "
        "targets question answering that is strictly grounded in uploaded documents.",
    ),
    (
        "1. Method",
        "RefMindNet combines BM25 keyword retrieval with dense vector retrieval. "
        "The two retrievers are fused by an ensemble with equal weights of 0.5 and "
        "0.5. Chinese text is tokenized with the jieba tokenizer so that keyword "
        "matching works for mixed Chinese-English scientific text.",
    ),
    (
        "2. Experimental Setup",
        "The retriever was evaluated on a corpus of 20000 annotated question answer "
        "pairs. The embedding dimension is 1024. At inference time the retrieval "
        "top-k is set to 5 for each retriever before fusion.",
    ),
    (
        "3. Results",
        "Grounding the generator on retrieved context reduces the hallucination "
        "rate from 38 percent to 21 percent. The average end-to-end query latency "
        "is 3.2 seconds. Answer faithfulness improves substantially when the "
        "no-context refusal policy is enabled.",
    ),
    (
        "4. Conclusion",
        "RefMindNet demonstrates that hybrid retrieval with strict grounding is "
        "effective for scientific reading. Future work includes adding a "
        "cross-encoder reranking stage on top of the ensemble retriever.",
    ),
]


def build_sample_pdf(output_path: str | Path) -> Path:
    """用 PyMuPDF 逐页写入文本（自动换行），生成合成论文 PDF。"""
    import fitz  # PyMuPDF

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    for title, body in PAGES:
        page = doc.new_page()  # 默认 A4
        # 标题
        page.insert_textbox(
            fitz.Rect(56, 60, 540, 120),
            title,
            fontsize=16,
            fontname="helv",
            align=fitz.TEXT_ALIGN_LEFT,
        )
        # 正文（自动换行，避免超出页面被截断）
        page.insert_textbox(
            fitz.Rect(56, 130, 540, 760),
            body,
            fontsize=12,
            fontname="helv",
            align=fitz.TEXT_ALIGN_LEFT,
        )
    doc.save(str(output_path))
    doc.close()
    return output_path


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out = build_sample_pdf(here / "corpus" / "RefMindNet.pdf")
    print(f"已生成合成语料：{out}")
