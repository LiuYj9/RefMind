"""PDF 解析。

首选方案：调用 MinerU（https://github.com/opendatalab/MinerU）CLI，对文本、
公式、表格进行高精度解析。若未安装 MinerU（或设置 ``USE_FALLBACK_PARSER=true``），
则回退到 PyMuPDF 逐页提取文本，保证应用可用。

两种方案都将输出归一化为统一结构::

    {
        "source": "paper.pdf",
        "parser": "mineru" | "pymupdf",
        "markdown": "<完整 markdown / 纯文本>",
        "pages": [{"page": 1, "text": "..."}, ...]
    }
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..config import settings


class ParseError(RuntimeError):
    """当所有可用后端都无法解析 PDF 时抛出。"""


def _resolve_mineru_binary() -> str | None:
    """解析 MinerU 可执行文件的完整路径。

    依次在系统 PATH 与当前 Python 解释器所在目录（虚拟环境的 Scripts/bin）
    中查找，找不到时返回 None。
    """
    binary = settings.mineru_binary
    # 1) 系统 PATH
    found = shutil.which(binary)
    if found:
        return found
    # 2) 当前解释器目录（venv 未激活时 Scripts 不在 PATH 中）
    exe_dir = str(Path(sys.executable).parent)
    found = shutil.which(binary, path=exe_dir)
    if found:
        return found
    return None


def _mineru_available() -> bool:
    """检测 MinerU 命令是否可用。"""
    return _resolve_mineru_binary() is not None


def parse_pdf(
    file_path: str | Path, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """将 PDF 解析为 RefMind 归一化结构。

    在 MinerU 可用且未强制回退时优先使用 MinerU。
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise ParseError(f"未找到 PDF 文件：{file_path}")

    if not settings.use_fallback_parser and _mineru_available():
        try:
            return _parse_with_mineru(file_path, output_dir)
        except Exception as exc:  # noqa: BLE001 - MinerU 任意失败都回退
            # MinerU 可能因模型下载、GPU、格式等多种原因失败，此处优雅降级
            print(f"[pdf_parser] MinerU 解析失败（{exc}），改用 PyMuPDF 回退。")

    return _parse_with_pymupdf(file_path)


def _parse_with_mineru(
    file_path: Path, output_dir: str | Path | None
) -> dict[str, Any]:
    """调用 MinerU CLI 并归一化其输出。"""
    out_dir = (
        Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="mineru_"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = _resolve_mineru_binary() or settings.mineru_binary
    cmd = [
        binary,
        "-p",
        str(file_path),
        "-o",
        str(out_dir),
        "-b",
        settings.mineru_backend,
    ]
    # method 仅对 pipeline / hybrid-* 后端有效
    if settings.mineru_backend.startswith(("pipeline", "hybrid")):
        cmd += ["-m", settings.mineru_method]

    # 透传模型下载源等环境变量
    env = os.environ.copy()
    if settings.mineru_model_source:
        env["MINERU_MODEL_SOURCE"] = settings.mineru_model_source

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise ParseError(
            f"MinerU 退出码 {proc.returncode}："
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )

    markdown, pages = _collect_mineru_output(out_dir, file_path.stem)
    if not markdown.strip():
        raise ParseError("MinerU 未提取到任何文本。")

    return {
        "source": file_path.name,
        "parser": "mineru",
        "markdown": markdown,
        "pages": pages,
    }


def _collect_mineru_output(
    out_dir: Path, stem: str
) -> tuple[str, list[dict[str, Any]]]:
    """定位 MinerU 输出的 markdown / content_list，兼容不同版本目录结构。"""
    # 优先使用带页码信息的结构化内容列表
    content_lists = list(out_dir.rglob("*_content_list.json"))
    pages: list[dict[str, Any]] = []
    if content_lists:
        try:
            items = json.loads(content_lists[0].read_text(encoding="utf-8"))
            page_map: dict[int, list[str]] = {}
            for item in items:
                text = item.get("text") or item.get("table_caption") or ""
                if isinstance(text, list):
                    text = "\n".join(str(t) for t in text)
                page_idx = int(item.get("page_idx", 0)) + 1
                if text.strip():
                    page_map.setdefault(page_idx, []).append(text.strip())
            pages = [
                {"page": p, "text": "\n".join(chunks)}
                for p, chunks in sorted(page_map.items())
            ]
        except (json.JSONDecodeError, ValueError, TypeError):
            pages = []

    md_files = list(out_dir.rglob("*.md"))
    markdown = ""
    if md_files:
        # 通常体积最大的 markdown 文件即为正文
        md_path = max(md_files, key=lambda p: p.stat().st_size)
        markdown = md_path.read_text(encoding="utf-8", errors="ignore")

    if not pages and markdown:
        pages = [{"page": 1, "text": markdown}]
    if not markdown and pages:
        markdown = "\n\n".join(p["text"] for p in pages)

    return markdown, pages


def _parse_with_pymupdf(file_path: Path) -> dict[str, Any]:
    """回退解析：使用 PyMuPDF 逐页提取纯文本。"""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise ParseError(
            "MinerU 与 PyMuPDF 均不可用。请安装 PyMuPDF"
            "（pip install pymupdf）或 MinerU 以启用 PDF 解析。"
        ) from exc

    pages: list[dict[str, Any]] = []
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": i, "text": text})

    if not pages:
        raise ParseError(
            f"{file_path.name} 中没有可提取的文本（可能是扫描件）。"
            "对纯图片文档请使用带 OCR 的 MinerU。"
        )

    markdown = "\n\n".join(f"<!-- page {p['page']} -->\n{p['text']}" for p in pages)
    return {
        "source": file_path.name,
        "parser": "pymupdf",
        "markdown": markdown,
        "pages": pages,
    }


def save_parsed(parsed: dict[str, Any], dest: str | Path) -> Path:
    """将归一化解析结果以 JSON 持久化，返回写入路径。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dest


def extract_text_from_parsed(parsed: dict[str, Any]) -> str:
    """返回解析结果的最佳全文表示。"""
    if parsed.get("markdown"):
        return parsed["markdown"]
    return "\n\n".join(p.get("text", "") for p in parsed.get("pages", []))
