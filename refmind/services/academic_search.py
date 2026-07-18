"""开放学术索引检索。

该模块只把论文题录与摘要转换为本轮临时 ``Document`` 证据，不写入 Chroma、
SQLite 文献表或长期记忆。默认使用 Semantic Scholar Academic Graph；OpenAlex
和 Crossref 作为可配置替代源。
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from langchain_core.documents import Document

from ..config import settings

LOGGER = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.I)
_PROVIDER_LABELS = {
    "auto": "自动融合",
    "semantic_scholar": "Semantic Scholar",
    "openalex": "OpenAlex",
    "crossref": "Crossref",
}
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class AcademicSearchError(RuntimeError):
    """学术索引请求或响应无效。"""


@dataclass(frozen=True)
class AcademicSearchResult:
    """一次学术检索的标准化结果。"""

    query: str
    provider: str
    documents: tuple[Document, ...] = ()
    providers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    failed: bool = False


def provider_label(provider: str) -> str:
    """返回适合 UI 展示的检索源名称。"""
    key = str(provider or "").strip().casefold()
    return _PROVIDER_LABELS.get(key, key or "未知学术索引")


def sanitize_external_url(value: object) -> str:
    """仅允许无凭据的 HTTP(S) 外链，避免引用 metadata 产生危险协议。"""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except ValueError:
        return ""
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        return ""
    return raw


def _clean_text(value: object, *, max_chars: int | None = None) -> str:
    text = html.unescape(str(value or ""))
    text = _HTML_TAG_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def _normalise_doi(value: object) -> str:
    doi = _DOI_PREFIX_RE.sub("", str(value or "").strip()).strip()
    return doi if doi and " " not in doi else ""


def _append_semantic_scholar_attribution(url: str) -> str:
    """Semantic Scholar 的公开展示链接按 API 许可附带来源参数。"""
    safe = sanitize_external_url(url)
    if not safe or "semanticscholar.org" not in urlparse(safe).netloc.casefold():
        return safe
    if "utm_source=" in safe:
        return safe
    return safe + ("&" if "?" in safe else "?") + "utm_source=api"


def _request_json(
    base_url: str,
    *,
    params: dict[str, object],
    headers: dict[str, str] | None,
    timeout: float,
    retries: int = 2,
) -> dict[str, Any]:
    """请求固定的官方 API 端点，并对限流/瞬时故障做短退避。"""
    url = f"{base_url}?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "RefMind/1.0 academic-search",
            **(headers or {}),
        },
        method="GET",
    )
    for attempt in range(max(0, retries) + 1):
        try:
            # URL 是模块内固定的官方 API 端点，不接受调用方提供主机名。
            with urlopen(  # noqa: S310
                request, timeout=max(1.0, float(timeout))
            ) as response:
                payload = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(payload) > _MAX_RESPONSE_BYTES:
                raise AcademicSearchError("学术索引响应超过安全大小限制")
            decoded = json.loads(payload.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise AcademicSearchError("学术索引返回了非对象 JSON")
            return decoded
        except HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt >= retries:
                raise AcademicSearchError(
                    f"学术索引 HTTP {exc.code}"
                ) from exc
        except URLError as exc:
            if attempt >= retries:
                reason = _clean_text(getattr(exc, "reason", "网络连接失败"), max_chars=160)
                raise AcademicSearchError(f"学术索引连接失败：{reason}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AcademicSearchError("学术索引返回了无效 JSON") from exc
        time.sleep(0.4 * (2**attempt))
    raise AcademicSearchError("学术索引请求失败")


def _paper_document(
    *,
    provider: str,
    external_id: str,
    title: str,
    abstract: str,
    authors: list[str],
    year: int | None,
    venue: str,
    doi: str,
    external_url: str,
    open_access_pdf_url: str = "",
    citation_count: int | None = None,
    publication_date: str = "",
) -> Document | None:
    clean_title = _clean_text(title, max_chars=500)
    clean_abstract = _clean_text(
        abstract,
        max_chars=max(500, int(settings.academic_search_max_abstract_chars)),
    )
    if not clean_title or not clean_abstract:
        # 题名/作者等题录只能证明论文存在，不能支撑正文问题，因此不进入 LLM 证据。
        return None

    clean_authors = [_clean_text(item, max_chars=120) for item in authors]
    clean_authors = [item for item in clean_authors if item][:20]
    clean_venue = _clean_text(venue, max_chars=240)
    clean_doi = _normalise_doi(doi)
    safe_url = sanitize_external_url(external_url)
    safe_pdf_url = sanitize_external_url(open_access_pdf_url)
    stable_id = clean_doi.casefold() or _clean_text(external_id, max_chars=200)
    if not stable_id:
        stable_id = clean_title.casefold()
    evidence_id = f"academic:{provider}:{stable_id}"

    header = [f"论文题名：{clean_title}"]
    if clean_authors:
        header.append("作者：" + ", ".join(clean_authors))
    if year:
        header.append(f"年份：{year}")
    if clean_venue:
        header.append(f"期刊/会议：{clean_venue}")
    header.append(f"摘要：{clean_abstract}")

    metadata: dict[str, object] = {
        "evidence_origin": "academic_search",
        "evidence_level": "abstract",
        "provider": provider_label(provider),
        "provider_key": provider,
        "external_id": _clean_text(external_id, max_chars=200),
        "evidence_id": evidence_id,
        "chunk_id": evidence_id,
        "paper_title": clean_title,
        "authors": ", ".join(clean_authors),
        "publication_year": year,
        "publication_date": _clean_text(publication_date, max_chars=32),
        "venue": clean_venue,
        "doi": clean_doi,
        "external_url": safe_url,
        "open_access_pdf_url": safe_pdf_url,
        "citation_count": citation_count,
        "source": safe_url,
    }
    return Document(page_content="\n".join(header), metadata=metadata)


def _semantic_scholar_search(query: str, limit: int) -> list[Document]:
    normalised_query = re.sub(r"[-‐‑‒–—]+", " ", query)
    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    payload = _request_json(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": normalised_query,
            "limit": limit,
            "fields": (
                "paperId,title,abstract,year,authors,venue,url,externalIds,"
                "citationCount,publicationDate,openAccessPdf"
            ),
        },
        headers=headers,
        timeout=settings.academic_search_timeout_seconds,
    )
    documents: list[Document] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        paper_id = _clean_text(item.get("paperId"), max_chars=100)
        external_ids = item.get("externalIds") or {}
        if not isinstance(external_ids, dict):
            external_ids = {}
        open_pdf = item.get("openAccessPdf") or {}
        if not isinstance(open_pdf, dict):
            open_pdf = {}
        authors = [
            str(author.get("name") or "")
            for author in (item.get("authors") or [])
            if isinstance(author, dict)
        ]
        semantic_url = str(item.get("url") or "")
        if not semantic_url and paper_id:
            semantic_url = f"https://www.semanticscholar.org/paper/{paper_id}"
        document = _paper_document(
            provider="semantic_scholar",
            external_id=paper_id,
            title=str(item.get("title") or ""),
            abstract=str(item.get("abstract") or ""),
            authors=authors,
            year=item.get("year") if isinstance(item.get("year"), int) else None,
            venue=str(item.get("venue") or ""),
            doi=str(external_ids.get("DOI") or ""),
            external_url=_append_semantic_scholar_attribution(semantic_url),
            open_access_pdf_url=str(open_pdf.get("url") or ""),
            citation_count=(
                item.get("citationCount")
                if isinstance(item.get("citationCount"), int)
                else None
            ),
            publication_date=str(item.get("publicationDate") or ""),
        )
        if document is not None:
            documents.append(document)
    return documents


def _abstract_from_inverted_index(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and 0 <= position < 100_000:
                positioned.append((position, str(word)))
    positioned.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned)


def _openalex_search(query: str, limit: int) -> list[Document]:
    if not settings.openalex_api_key:
        raise AcademicSearchError("OpenAlex 当前要求配置 OPENALEX_API_KEY")
    payload = _request_json(
        "https://api.openalex.org/works",
        params={
            "api_key": settings.openalex_api_key,
            "search": query,
            "sort": "relevance_score:desc",
            "per_page": limit,
        },
        headers=None,
        timeout=settings.academic_search_timeout_seconds,
    )
    documents: list[Document] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict) or item.get("is_retracted") is True:
            continue
        authors = []
        for authorship in item.get("authorships") or []:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author") or {}
            if isinstance(author, dict) and author.get("display_name"):
                authors.append(str(author["display_name"]))
        location = item.get("primary_location") or {}
        if not isinstance(location, dict):
            location = {}
        source = location.get("source") or {}
        if not isinstance(source, dict):
            source = {}
        external_url = str(
            location.get("landing_page_url") or item.get("doi") or item.get("id") or ""
        )
        document = _paper_document(
            provider="openalex",
            external_id=str(item.get("id") or ""),
            title=str(item.get("display_name") or item.get("title") or ""),
            abstract=_abstract_from_inverted_index(item.get("abstract_inverted_index")),
            authors=authors,
            year=(
                item.get("publication_year")
                if isinstance(item.get("publication_year"), int)
                else None
            ),
            venue=str(source.get("display_name") or ""),
            doi=str(item.get("doi") or ""),
            external_url=external_url,
            open_access_pdf_url=str(location.get("pdf_url") or ""),
            citation_count=(
                item.get("cited_by_count")
                if isinstance(item.get("cited_by_count"), int)
                else None
            ),
            publication_date=str(item.get("publication_date") or ""),
        )
        if document is not None:
            documents.append(document)
    return documents


def _crossref_year(item: dict[str, Any]) -> int | None:
    for field in ("published-print", "published-online", "published", "issued"):
        value = item.get(field) or {}
        parts = value.get("date-parts") if isinstance(value, dict) else None
        if (
            isinstance(parts, list)
            and parts
            and isinstance(parts[0], list)
            and parts[0]
            and isinstance(parts[0][0], int)
        ):
            return parts[0][0]
    return None


def _crossref_search(query: str, limit: int) -> list[Document]:
    params: dict[str, object] = {
        "query.bibliographic": query,
        "rows": limit,
        "select": (
            "DOI,title,abstract,author,published,published-print,published-online,"
            "issued,container-title,URL,is-referenced-by-count,type"
        ),
    }
    if settings.crossref_mailto:
        params["mailto"] = settings.crossref_mailto
    payload = _request_json(
        "https://api.crossref.org/works",
        params=params,
        headers=None,
        timeout=settings.academic_search_timeout_seconds,
    )
    message = payload.get("message") or {}
    items = message.get("items") if isinstance(message, dict) else []
    documents: list[Document] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        authors = []
        for author in item.get("author") or []:
            if not isinstance(author, dict):
                continue
            name = " ".join(
                part
                for part in (str(author.get("given") or ""), str(author.get("family") or ""))
                if part
            )
            if name:
                authors.append(name)
        titles = item.get("title") or []
        venues = item.get("container-title") or []
        doi = str(item.get("DOI") or "")
        document = _paper_document(
            provider="crossref",
            external_id=doi,
            title=str(titles[0] if isinstance(titles, list) and titles else titles),
            abstract=str(item.get("abstract") or ""),
            authors=authors,
            year=_crossref_year(item),
            venue=str(venues[0] if isinstance(venues, list) and venues else venues),
            doi=doi,
            external_url=str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")),
            citation_count=(
                item.get("is-referenced-by-count")
                if isinstance(item.get("is-referenced-by-count"), int)
                else None
            ),
        )
        if document is not None:
            documents.append(document)
    return documents


_PROVIDERS: dict[str, Callable[[str, int], list[Document]]] = {
    "semantic_scholar": _semantic_scholar_search,
    "openalex": _openalex_search,
    "crossref": _crossref_search,
}


def _search_auto(query: str, limit: int) -> tuple[list[Document], list[str], bool]:
    """并行查询无需凭证的开放源；有 OpenAlex Key 时自动加入融合。"""
    provider_keys = ["semantic_scholar", "crossref"]
    if settings.openalex_api_key:
        provider_keys.append("openalex")

    results: dict[str, list[Document]] = {key: [] for key in provider_keys}
    warnings: list[str] = []
    successes = 0
    executor = ThreadPoolExecutor(
        max_workers=len(provider_keys),
        thread_name_prefix="refmind-academic-search",
    )
    try:
        futures = {
            executor.submit(_PROVIDERS[key], query, limit): key
            for key in provider_keys
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = list(future.result() or [])
                successes += 1
            except Exception as exc:  # noqa: BLE001 - 单个开放源失败不影响融合结果
                warnings.append(f"{provider_label(key)} 检索失败：{exc}")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    # 与 provider_keys 顺序一致，避免并发完成顺序造成重复 DOI 的来源随机变化。
    documents = _deduplicate(
        [document for key in provider_keys for document in results[key]]
    )
    return documents, warnings, successes == 0


def _deduplicate(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    output: list[Document] = []
    for document in documents:
        metadata = document.metadata or {}
        key = str(metadata.get("doi") or metadata.get("evidence_id") or "").casefold()
        if not key:
            key = _clean_text(metadata.get("paper_title")).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(document)
    return output


def search_academic_papers(
    query: str,
    *,
    provider: str | None = None,
    limit: int | None = None,
) -> AcademicSearchResult:
    """检索并标准化含摘要的论文候选；任何失败都收敛为可观察结果。"""
    clean_query = _clean_text(query, max_chars=500)
    selected = str(provider or settings.academic_search_provider).strip().casefold()
    candidate_limit = max(
        1,
        min(100, int(limit or settings.academic_search_candidate_k)),
    )
    if not settings.academic_search_enabled:
        return AcademicSearchResult(
            clean_query,
            selected,
            warnings=("GS 学术检索已在设置中关闭",),
            failed=True,
        )
    if not clean_query:
        return AcademicSearchResult(
            clean_query,
            selected,
            warnings=("学术检索查询为空",),
            failed=True,
        )
    search = _PROVIDERS.get(selected)
    if selected != "auto" and search is None:
        return AcademicSearchResult(
            clean_query,
            selected,
            warnings=(f"不支持的学术检索源：{selected}",),
            failed=True,
        )

    try:
        if selected == "auto":
            documents, warnings, failed = _search_auto(
                clean_query, candidate_limit
            )
        else:
            documents = _deduplicate(search(clean_query, candidate_limit))  # type: ignore[misc]
            warnings = []
            failed = False
    except Exception as exc:  # noqa: BLE001 - 外部服务失败不得中断本地 RAG
        LOGGER.warning("%s 学术检索失败：%s", provider_label(selected), exc)
        return AcademicSearchResult(
            clean_query,
            selected,
            warnings=(f"{provider_label(selected)} 检索失败：{exc}",),
            failed=True,
        )

    if not documents:
        warnings.append(
            f"{provider_label(selected)} 未返回含可用摘要的论文；题录结果未作为回答证据"
        )
    return AcademicSearchResult(
        query=clean_query,
        provider=selected,
        documents=tuple(documents),
        providers=tuple(
            dict.fromkeys(
                str(document.metadata.get("provider_key") or "")
                for document in documents
                if document.metadata.get("provider_key")
            )
        ),
        warnings=tuple(warnings),
        failed=failed,
    )


__all__ = [
    "AcademicSearchError",
    "AcademicSearchResult",
    "provider_label",
    "sanitize_external_url",
    "search_academic_papers",
]
