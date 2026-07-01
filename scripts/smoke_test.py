"""端到端冒烟测试：解析 -> 入库 -> 混合检索 -> 问答 -> 翻译 -> 摘要。

运行：python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 避免 Windows GBK 控制台无法输出中文 / Emoji 的编码错误
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# 允许从项目根目录导入 refmind
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refmind import storage
from refmind.config import settings
from refmind.llm import translate
from refmind.parsing import parse_pdf
from refmind.rag import RelevantMemory, answer_question, get_retriever
from refmind.services import ingest_pdf, remove_group


def log(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def main() -> int:
    settings.ensure_dirs()
    storage.init_db()

    sample = settings.project_root / "data" / "sample_paper.pdf"
    if not sample.exists():
        print("缺少测试 PDF：data/sample_paper.pdf")
        return 1

    # 1) 解析（验证 MinerU 是否被正确识别与调用）
    log("步骤 1：解析 PDF")
    parsed = parse_pdf(sample)
    print("解析器:", parsed["parser"])
    print("页数:", len(parsed["pages"]))
    print("正文前 120 字:", parsed["markdown"][:120].replace("\n", " "))
    assert parsed["markdown"].strip(), "解析结果为空"

    # 2) 入库（嵌入走真实 DashScope 接口）
    log("步骤 2：建组并入库")
    group = storage.create_group(f"smoke_{int(time.time())}")
    try:
        doc = ingest_pdf(
            group.id, sample, "sample_paper.pdf",
            progress=lambda p, m: print(f"  [{p:.0%}] {m}", flush=True),
        )
        print("文档状态:", doc.status, "| 分块数:", doc.num_chunks)
        print("自动摘要:", (doc.summary or "")[:150])
        assert doc.num_chunks > 0, "未生成任何分块"

        # 3) 混合检索
        log("步骤 3：混合检索")
        retriever = get_retriever(group.id)
        assert retriever is not None, "检索器为空"
        hits = retriever.invoke("BM25 和向量检索是如何融合的？")
        print("命中片段数:", len(hits))
        for h in hits[:2]:
            print("  -", h.metadata.get("filename"), "第",
                  h.metadata.get("page"), "页:", h.page_content[:60].replace("\n", " "))
        assert hits, "检索无结果"

        # 4) 问答（真实 LLM）
        log("步骤 4：RAG 问答")
        memory = RelevantMemory(storage.create_session(group.id).id)
        result = answer_question(
            "这篇论文提出的方法在准确率上提升了多少？", group.id, memory=memory
        )
        print("回答:", result["answer"])
        print("引用片段数:", len(result["documents"]))
        assert result["answer"].strip(), "回答为空"

        # 5) 无关问题应触发兜底话术
        log("步骤 5：越界问题兜底")
        oob = answer_question("请介绍一下量子计算机的发展历史。", group.id)
        print("回答:", oob["answer"])

        # 6) 翻译
        log("步骤 6：翻译")
        zh = translate("Hybrid retrieval fuses BM25 and dense vector scores.", "中文")
        print("译文:", zh)
        assert zh.strip(), "翻译为空"

        log("全部步骤通过 [PASS]")
        return 0
    finally:
        log("清理测试数据")
        remove_group(group.id)
        print("已删除测试组", group.id)


if __name__ == "__main__":
    raise SystemExit(main())
