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
import uuid
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..config import settings

_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^]]*]\([^)]*\)")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# MinerU 可能需要下载/加载模型，因此给予较宽裕的上限；
# 无论超时还是命令失败，上层都会继续尝试 PyMuPDF 降级解析。
_MINERU_TIMEOUT_SECONDS = 15 * 60

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
    if not file_path.is_file():
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
    if output_dir is None:
        # 内部临时目录必须在成功和异常路径上都回收，
        # 避免长期运行时积累 MinerU 的图片与中间 JSON。
        with tempfile.TemporaryDirectory(prefix="mineru_") as tmp_dir:
            return _parse_with_mineru_in_dir(file_path, Path(tmp_dir))

    # 显式目录下仍为本次任务创建唯一子目录，避免误读其中其他论文的旧产物。
    run_dir = Path(output_dir) / f"{file_path.stem}_{uuid.uuid4().hex}"
    return _parse_with_mineru_in_dir(file_path, run_dir)


def _parse_with_mineru_in_dir(file_path: Path, out_dir: Path) -> dict[str, Any]:
    """在指定目录运行 MinerU；目录生命周期由上层决定。"""
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

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=_MINERU_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ParseError(
            f"MinerU 解析超过 {_MINERU_TIMEOUT_SECONDS} 秒，已终止。"
        ) from exc
    except OSError as exc:
        raise ParseError(f"无法启动 MinerU：{exc}") from exc

    if proc.returncode != 0:
        raise ParseError(
            f"MinerU 退出码 {proc.returncode}："
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )

    markdown, pages, tables = _collect_mineru_output(out_dir, file_path.stem)
    extracted_text = "\n".join(
        [markdown]
        + [str(page.get("text", "")) for page in pages]
        + [
            str(table.get("normalized_text") or table.get("raw_text") or "")
            for table in tables
        ]
    )
    if not _has_meaningful_text(extracted_text):
        raise ParseError(
            "MinerU 未提取到可用文本（可能是 OCR 未识别的扫描件）。"
        )

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
    pages: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []

    # 同一输出目录可能保留过其他 PDF 的历史产物。按输入文件名
    # 打分而不是盲目取第一个/最大文件，防止将旧论文入库。
    content_items: list[dict[str, Any]] | None = None
    content_lists = _rank_mineru_outputs(
        out_dir.rglob("*_content_list.json"), stem, "_content_list.json"
    )
    for content_path in content_lists:
        try:
            content_items = _read_content_list(content_path)
            break
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            # 某个候选文件损坏时继续尝试次优候选。
            continue

    if content_items is not None:
        page_map: dict[int, list[str]] = {}
        for item in content_items:
            text = _as_text(item.get("text") or item.get("table_caption"))
            if text:
                page_map.setdefault(_content_page_number(item), []).append(text)
        pages = [
            {"page": page, "text": "\n".join(chunks)}
            for page, chunks in sorted(page_map.items())
        ]
        tables = _merge_continued_tables(_extract_tables(content_items))

    md_files = _rank_mineru_outputs(out_dir.rglob("*.md"), stem, ".md")
    markdown = ""
    for md_path in md_files:
        try:
            markdown = md_path.read_text(encoding="utf-8", errors="replace")
            break
        except OSError:
            continue

    if not pages and markdown:
        pages = [{"page": 1, "text": markdown}]
    if not markdown and pages:
        markdown = "\n\n".join(p["text"] for p in pages)

    return markdown, pages, tables


def _output_stem(path: Path, suffix: str) -> str:
    """去除 MinerU 约定后缀，得到候选产物对应的 PDF 名。"""
    name = path.name
    if name.casefold().endswith(suffix.casefold()):
        return name[: -len(suffix)]
    return path.stem


def _rank_mineru_outputs(paths: Any, stem: str, suffix: str) -> list[Path]:
    """按与输入 PDF 的匹配度排序 MinerU 产物。"""
    wanted = stem.casefold()
    ranked: list[tuple[int, int, int, str, Path]] = []
    for path in paths:
        candidate = _output_stem(path, suffix).casefold()
        parts = [part.casefold() for part in path.parent.parts]
        score = 0
        if candidate == wanted:
            score += 100
        elif wanted and wanted in candidate:
            score += 30
        if wanted in parts:
            score += 50
        try:
            stat = path.stat()
        except OSError:
            continue
        ranked.append((score, stat.st_mtime_ns, stat.st_size, str(path), path))

    # 只要有与 stem 匹配的文件，就排除不匹配的历史产物；
    # 旧版 MinerU 产物不带原文件名时，才允许回退到最新候选。
    if any(item[0] > 0 for item in ranked):
        ranked = [item for item in ranked if item[0] > 0]
    ranked.sort(key=lambda item: item[:4], reverse=True)
    return [item[-1] for item in ranked]


def _read_content_list(path: Path) -> list[dict[str, Any]]:
    """读取 MinerU 内容列表，兼容列表与带 items/content 外壳的版本。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("items", "content_list", "content", "blocks"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError("MinerU content_list 不是列表")
    return [item for item in raw if isinstance(item, dict)]


def _content_page_number(item: dict[str, Any]) -> int:
    """兼容 page_idx（0 起始）与 page_no/page_number（1 起始）。"""
    if "page_idx" in item:
        try:
            return max(1, int(item["page_idx"]) + 1)
        except (TypeError, ValueError):
            return 1
    for key in ("page_no", "page_number"):
        try:
            return max(1, int(item[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return 1


def _has_meaningful_text(text: str) -> bool:
    """过滤纯图片引用/注释，避免将 OCR 失败的扫描件误判为成功。"""
    visible = _MARKDOWN_IMAGE_RE.sub("", text or "")
    visible = _HTML_COMMENT_RE.sub("", visible)
    visible = _HTML_TAG_RE.sub("", visible)
    # 单个页码/水印字符不算正文；四个中英文数字字符仍兼容极短摘要页。
    return sum(char.isalnum() for char in visible) >= 4


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
        page_no = _content_page_number(item)
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
    try:
        with fitz.open(file_path) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if _has_meaningful_text(text):
                    pages.append({"page": i, "text": text})
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"PyMuPDF 无法读取 {file_path.name}：{exc}") from exc

    if not pages:
        raise ParseError(
            f"{file_path.name} 中没有可提取的文本（可能是扫描件或空 PDF）。"
            "请将 MinerU 方法设为 ocr 后重试。"
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
    temp_path: Path | None = None
    try:
        # 先在同一目录完整写入再原子替换，避免崩溃后留下半个 JSON。
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dest.parent,
            prefix=f".{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(parsed, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, dest)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
    return dest


def extract_text_from_parsed(parsed: dict[str, Any]) -> str:
    """返回解析结果的最佳全文表示。"""
    if parsed.get("markdown"):
        return parsed["markdown"]
    return "\n\n".join(p.get("text", "") for p in parsed.get("pages", []))
