"""PDF 解析：优先用 MinerU，未安装或失败时回退 PyMuPDF。

两种方式统一输出 {"source", "parser", "markdown", "pages":[{"page","text"}]}。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..config import settings

_TABLE_NO_RE = re.compile(r"(?:table|tab\.?|表)\s*([a-z]?\d+(?:[.-]\d+)*)", re.I)
_CONTINUED_RE = re.compile(r"(continued|cont\.|续表|接上表|续)", re.I)


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
        except Exception as exc:  # noqa: BLE001
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

    markdown, pages, tables = _collect_mineru_output(out_dir, file_path.stem)
    if not markdown.strip():
        raise ParseError("MinerU 未提取到任何文本。")

    return {
        "source": file_path.name,
        "parser": "mineru",
        "markdown": markdown,
        "pages": pages,
        "tables": tables,
    }


def _collect_mineru_output(
    out_dir: Path, stem: str
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """定位 MinerU 输出的 markdown / content_list，兼容不同版本目录结构。"""
    # 优先使用带页码信息的结构化内容列表
    content_lists = list(out_dir.rglob("*_content_list.json"))
    pages: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
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
            tables = _merge_continued_tables(_extract_tables(items))
        except (json.JSONDecodeError, ValueError, TypeError):
            pages = []
            tables = []

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

    return markdown, pages, tables


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_as_text(v) for v in value]
        return "\n".join(p for p in parts if p).strip()
    if isinstance(value, dict):
        for key in ("text", "html", "markdown", "content"):
            text = _as_text(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _table_text(item: dict[str, Any]) -> str:
    for key in (
        "table_body",
        "table_html",
        "table_markdown",
        "html",
        "markdown",
        "text",
    ):
        text = _as_text(item.get(key))
        if text:
            return text
    return ""


def _looks_like_table(item: dict[str, Any]) -> bool:
    kind = _as_text(
        item.get("type") or item.get("category") or item.get("block_type")
    ).lower()
    if "table" in kind:
        return True
    if item.get("table_caption") and _as_text(item.get("text")):
        return True
    return any(
        item.get(key)
        for key in ("table_body", "table_html", "table_markdown")
    )


def _split_table_row(line: str) -> list[str]:
    line = line.strip()
    if not line:
        return []
    if "|" in line:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
    elif "\t" in line:
        cells = [cell.strip() for cell in line.split("\t")]
    else:
        return []
    if cells and all(re.fullmatch(r"[:\-\s]+", cell or "-") for cell in cells):
        return []
    return cells


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            attr_map = dict(attrs)
            try:
                self._colspan = max(1, int(attr_map.get("colspan") or 1))
            except ValueError:
                self._colspan = 1
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            text = " ".join("".join(self._cell).split())
            self._row.extend([text] * self._colspan)
            self._cell = None
            self._colspan = 1
        elif tag == "tr" and self._row is not None:
            if any(cell.strip() for cell in self._row):
                self.rows.append(self._row)
            self._row = None


def _rows_from_html(text: str) -> list[list[str]]:
    if "<tr" not in text.lower() and "<table" not in text.lower():
        return []
    parser = _HTMLTableParser()
    parser.feed(text)
    return parser.rows


def _rows_from_text(text: str) -> list[list[str]]:
    html_rows = _rows_from_html(text)
    if html_rows:
        return html_rows

    rows: list[list[str]] = []
    for line in text.splitlines():
        row = _split_table_row(line)
        if row:
            rows.append(row)
    return rows


def _header(rows: list[list[str]]) -> list[str]:
    return next((row for row in rows if any(cell.strip() for cell in row)), [])


def _col_count(rows: list[list[str]]) -> int:
    return max((len(row) for row in rows), default=0)


def _caption_no(caption: str) -> str:
    match = _TABLE_NO_RE.search(caption or "")
    return match.group(1).lower() if match else ""


def _similarity(left: list[str], right: list[str]) -> float:
    a = " | ".join(cell.lower().strip() for cell in left)
    b = " | ".join(cell.lower().strip() for cell in right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _page_height(item: dict[str, Any]) -> float | None:
    for key in ("page_height", "height"):
        try:
            value = float(item.get(key))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    size = item.get("page_size") or item.get("page")
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        try:
            return float(size[1])
        except (TypeError, ValueError):
            return None
    if isinstance(size, dict):
        try:
            return float(size.get("height"))
        except (TypeError, ValueError):
            return None
    return None


def _bbox(item: dict[str, Any]) -> list[float]:
    raw = item.get("bbox") or item.get("box")
    if isinstance(raw, dict):
        raw = [raw.get(k) for k in ("x0", "y0", "x1", "y1")]
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return []
    try:
        return [float(v) for v in raw[:4]]
    except (TypeError, ValueError):
        return []


def _extract_tables(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not _looks_like_table(item):
            continue
        text = _table_text(item)
        caption = _as_text(item.get("table_caption") or item.get("caption"))
        if not text and not caption:
            continue
        rows = _rows_from_text(text)
        page_no = int(item.get("page_idx", 0)) + 1
        tables.append(
            {
                "table_id": f"table_{len(tables) + 1:03d}",
                "caption": caption,
                "page_start": page_no,
                "page_end": page_no,
                "continued": False,
                "raw_text": text,
                "rows": rows,
                "col_count": _col_count(rows),
                "bbox": _bbox(item),
                "page_height": _page_height(item),
            }
        )
    return tables


def _near_page_bottom(table: dict[str, Any]) -> bool:
    bbox = table.get("bbox") or []
    height = table.get("page_height")
    return bool(bbox and height and bbox[3] / height >= 0.78)


def _near_page_top(table: dict[str, Any]) -> bool:
    bbox = table.get("bbox") or []
    height = table.get("page_height")
    return bool(bbox and height and bbox[1] / height <= 0.25)

# 合并连续表格的逻辑
def _is_continued_table(prev: dict[str, Any], cur: dict[str, Any]) -> bool:
    if cur["page_start"] != prev["page_end"] + 1:
        return False

    prev_no = _caption_no(prev.get("caption", ""))
    cur_no = _caption_no(cur.get("caption", ""))
    cur_caption = cur.get("caption", "")
    if cur_no and prev_no and cur_no != prev_no:
        return False
    if cur_no and not prev_no and not _CONTINUED_RE.search(cur_caption):
        return False

    score = 2
    same_cols = prev.get("col_count") and prev.get("col_count") == cur.get("col_count")
    if same_cols:
        score += 3

    header_sim = _similarity(_header(prev.get("rows", [])), _header(cur.get("rows", [])))
    if header_sim >= 0.82:
        score += 4
    elif header_sim >= 0.55:
        score += 2

    if _CONTINUED_RE.search(cur_caption):
        score += 5
    elif not cur_caption:
        score += 2
    elif prev_no and cur_no == prev_no:
        score += 4
    else:
        score -= 3

    if _near_page_bottom(prev):
        score += 2
    if _near_page_top(cur):
        score += 2

    return score >= 7


def _merge_rows(prev: list[list[str]], cur: list[list[str]]) -> list[list[str]]:
    if prev and cur and _similarity(_header(prev), _header(cur)) >= 0.82:
        return prev + cur[1:]
    return prev + cur


def _merge_two_tables(prev: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    prev["page_end"] = cur["page_end"]
    prev["continued"] = True
    prev["raw_text"] = "\n".join(
        text for text in (prev.get("raw_text", ""), cur.get("raw_text", "")) if text
    )
    prev["rows"] = _merge_rows(prev.get("rows", []), cur.get("rows", []))
    prev["col_count"] = max(prev.get("col_count", 0), cur.get("col_count", 0))
    if not prev.get("caption"):
        prev["caption"] = cur.get("caption", "")
    return prev


def _fill_down_rows(rows: list[list[str]]) -> list[list[str]]:
    width = _col_count(rows)
    filled: list[list[str]] = []
    carry = [""] * width
    for row in rows:
        current = (row + [""] * width)[:width]
        for idx, cell in enumerate(current):
            if cell:
                carry[idx] = cell
            elif carry[idx]:
                current[idx] = carry[idx]
        filled.append(current)
    return filled


def _table_to_text(table: dict[str, Any]) -> str:
    caption = table.get("caption") or table.get("table_id", "table")
    page_start = table.get("page_start")
    page_end = table.get("page_end")
    page_text = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
    lines = [f"Table: {caption}", f"Pages: {page_text}"]

    rows = _fill_down_rows(table.get("rows", []))
    if rows:
        columns = rows[0]
        lines.append("Columns: " + "; ".join(columns))
        for idx, row in enumerate(rows[1:], start=1):
            pairs = []
            for col, cell in zip(columns, row):
                name = col or "column"
                if cell:
                    pairs.append(f"{name}={cell}")
            if pairs:
                lines.append(f"Row {idx}: " + "; ".join(pairs))
    elif table.get("raw_text"):
        lines.extend(["Content:", table["raw_text"]])

    return "\n".join(lines).strip()


def _merge_continued_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for table in tables:
        if merged and _is_continued_table(merged[-1], table):
            _merge_two_tables(merged[-1], table)
        else:
            merged.append(table)

    for idx, table in enumerate(merged, start=1):
        table["table_id"] = f"table_{idx:03d}"
        table["normalized_text"] = _table_to_text(table)
    return merged


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
        "tables": [],
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
