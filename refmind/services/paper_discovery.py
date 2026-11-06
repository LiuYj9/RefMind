"""Synchronous, metadata-only discovery. No model, database or index writes."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import quote

from ..config import settings
from .academic_search import (
    AcademicSearchError, _abstract_from_inverted_index, _clean_text,
    _crossref_year, _normalise_doi, _request_json, sanitize_external_url,
)


@dataclass(frozen=True)
class SearchFilters:
    year_from: int | None = None
    year_to: int | None = None
    open_access_only: bool = False
    publication_types: tuple[str, ...] = ()
    fields_of_study: tuple[str, ...] = ()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("filters must be an object")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def __post_init__(self):
        for value in (self.year_from, self.year_to):
            if value is not None and (type(value) is not int or not 1000 <= value <= 2200):
                raise ValueError("invalid publication year")
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from must not exceed year_to")
        if type(self.open_access_only) is not bool:
            raise ValueError("open_access_only must be boolean")
        for name in ("publication_types", "fields_of_study"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or len(value) > 20 or any(
                not isinstance(v, str) or not v.strip() or len(v) > 100 for v in value
            ):
                raise ValueError("invalid " + name)
            object.__setattr__(self, name, tuple(v.strip() for v in value))


@dataclass(frozen=True)
class PaperLocation:
    url: str = ""
    landing_url: str = ""
    is_oa: bool | None = None
    license: str = ""
    version: str = "unknown"
    provider: str = ""

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("paper location must be an object")
        flag = data.get("is_oa")
        if flag is not None and type(flag) is not bool:
            raise ValueError("invalid is_oa")
        return cls(sanitize_external_url(data.get("url")),
                   sanitize_external_url(data.get("landing_url")), flag,
                   _clean_text(data.get("license", ""), max_chars=300),
                   _clean_text(data.get("version", "unknown"), max_chars=100),
                   _clean_text(data.get("provider", ""), max_chars=100))


@dataclass(frozen=True)
class PaperCandidate:
    paper_id: str
    title: str
    abstract: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str = ""
    providers: tuple[str, ...] = ()
    locations: tuple[PaperLocation, ...] = ()
    external_ids: dict[str, str] = field(default_factory=dict)
    citation_count: int | None = None
    publication_type: str = ""
    abstract_source: str = ""
    publication_date: str = ""
    venue: str = ""
    fields_of_study: tuple[str, ...] = ()
    is_retracted: bool | None = None
    field_sources: dict[str, list[dict]] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or not isinstance(data.get("paper_id"), str) or not data["paper_id"]:
            raise ValueError("paper_id required")
        title = _clean_text(data.get("title") or "", max_chars=1000)
        if not title:
            raise ValueError("paper title required")
        year = data.get("year")
        if year is not None and (type(year) is not int or not 1000 <= year <= 2200):
            raise ValueError("invalid publication year")
        def strings(key):
            raw = data.get(key) or ()
            if not isinstance(raw, (list, tuple)) or any(not isinstance(v, str) for v in raw):
                raise ValueError("invalid " + key)
            return tuple(raw)
        locations = data.get("locations") or ()
        if not isinstance(locations, (list, tuple)) or len(locations) > 100:
            raise ValueError("invalid locations")
        ids = data.get("external_ids") or {}
        if not isinstance(ids, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in ids.items()):
            raise ValueError("invalid external_ids")
        sources = data.get("field_sources") or {}
        if not isinstance(sources, dict) or any(not isinstance(k, str) or not isinstance(v, list) or any(not isinstance(row, dict) for row in v) for k, v in sources.items()):
            raise ValueError("invalid field_sources")
        return cls(data["paper_id"], title, _clean_text(data.get("abstract", ""), max_chars=12000),
                   strings("authors"), year, normalize_doi(data.get("doi", "")), strings("providers"),
                   tuple(PaperLocation.from_dict(v) for v in locations), dict(ids),
                   data.get("citation_count") if type(data.get("citation_count")) is int else None,
                   _clean_text(data.get("publication_type", "")), _clean_text(data.get("abstract_source", "")),
                   _clean_text(data.get("publication_date", "")), _clean_text(data.get("venue", "")),
                   strings("fields_of_study"), data.get("is_retracted") if type(data.get("is_retracted")) is bool else None, sources)


@dataclass(frozen=True)
class DiscoveryResult:
    papers: tuple[PaperCandidate, ...] = ()
    next_cursor: str | None = None
    warnings: tuple[str, ...] = ()
    failed: bool = False
    providers: tuple[str, ...] = ()
    partial: bool = False
    provider_status: dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("result must be an object")
        return cls(tuple(PaperCandidate.from_dict(p) for p in data.get("papers", [])),
                   data.get("next_cursor"), tuple(data.get("warnings", [])), bool(data.get("failed", False)),
                   tuple(data.get("providers", [])), bool(data.get("partial", False)), dict(data.get("provider_status", {})))


def normalize_doi(value: object) -> str:
    doi = _normalise_doi(value).casefold()
    return doi if re.fullmatch(r"10\.\d{4,9}/[^\s]+", doi) else ""


def _candidate(provider: str, item: dict) -> PaperCandidate | None:
    locations = []
    ids = {}
    if provider == "semantic_scholar":
        title, abstract = item.get("title"), item.get("abstract")
        authors = [a.get("name", "") for a in item.get("authors") or [] if isinstance(a, dict)]
        raw_ids = item.get("externalIds") or {}
        ids = {str(k): str(v) for k, v in raw_ids.items() if v} if isinstance(raw_ids, dict) else {}
        doi, year, venue = normalize_doi(ids.get("DOI")), item.get("year"), item.get("venue")
        native_id, count = item.get("paperId"), item.get("citationCount")
        pdf = item.get("openAccessPdf") or {}
        pdf = pdf if isinstance(pdf, dict) else {}
        locations.append(PaperLocation(sanitize_external_url(pdf.get("url")),
                         sanitize_external_url(item.get("url")), bool(pdf.get("url")) or item.get("isOpenAccess"),
                         _clean_text(pdf.get("license", "")), _clean_text(pdf.get("version", "unknown")), provider))
        types = item.get("publicationTypes") or []
        kind = str(types[0]) if types else ""
        fields = tuple(str(v) for v in item.get("fieldsOfStudy") or [])
    elif provider == "openalex":
        title = item.get("display_name") or item.get("title")
        abstract = _abstract_from_inverted_index(item.get("abstract_inverted_index"))
        authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships") or [] if isinstance(a, dict) and isinstance(a.get("author"), dict)]
        doi, year = normalize_doi(item.get("doi")), item.get("publication_year")
        primary = item.get("primary_location") or {}
        primary = primary if isinstance(primary, dict) else {}
        venue = (primary.get("source") or {}).get("display_name", "") if isinstance(primary.get("source") or {}, dict) else ""
        native_id, count, kind = item.get("id"), item.get("cited_by_count"), item.get("type", "")
        raw_locations = [item.get("best_oa_location"), primary, *(item.get("locations") or [])]
        for loc in raw_locations:
            if isinstance(loc, dict):
                locations.append(PaperLocation(sanitize_external_url(loc.get("pdf_url")), sanitize_external_url(loc.get("landing_page_url")),
                                 loc.get("is_oa") if type(loc.get("is_oa")) is bool else None,
                                 _clean_text(loc.get("license", "")), _clean_text(loc.get("version", "unknown")), provider))
        field_labels = []
        for topic in item.get("topics") or []:
            if isinstance(topic, dict):
                field_labels.extend([topic.get("display_name"), *(
                    (topic.get(level) or {}).get("display_name") for level in ("domain", "field", "subfield")
                    if isinstance(topic.get(level) or {}, dict)
                )])
        fields = tuple(dict.fromkeys(str(v) for v in field_labels if v))
        ids = {str(k): str(v) for k, v in (item.get("ids") or {}).items() if v} if isinstance(item.get("ids") or {}, dict) else {}
    else:
        titles, venues = item.get("title") or [], item.get("container-title") or []
        title = titles[0] if isinstance(titles, list) and titles else titles
        venue = venues[0] if isinstance(venues, list) and venues else venues
        abstract, doi, year = item.get("abstract"), normalize_doi(item.get("DOI")), _crossref_year(item)
        authors = [" ".join(str(a.get(k) or "") for k in ("given", "family")).strip() for a in item.get("author") or [] if isinstance(a, dict)]
        native_id, count, kind = doi, item.get("is-referenced-by-count"), item.get("type", "")
        licenses = [str(v.get("URL") or "") for v in item.get("license") or [] if isinstance(v, dict)]
        license_url = licenses[0] if licenses else ""
        for link in item.get("link") or []:
            if isinstance(link, dict) and str(link.get("content-type", "")).casefold() == "application/pdf":
                locations.append(PaperLocation(sanitize_external_url(link.get("URL")), sanitize_external_url(item.get("URL")),
                                 True if "creativecommons.org/" in license_url else None, license_url, "unknown", provider))
        locations.append(PaperLocation(landing_url=sanitize_external_url(item.get("URL")), provider=provider))
        fields = tuple(str(v) for v in item.get("subject") or [])
    clean_title = _clean_text(title or "", max_chars=1000)
    if not clean_title:
        return None
    clean_abstract = _clean_text(abstract or "", max_chars=12000)
    paper_id = "doi:" + doi if doi else provider + ":" + str(native_id or hashlib.sha256(clean_title.casefold().encode()).hexdigest()[:24])
    unique_locations = tuple(dict.fromkeys(locations))
    candidate = PaperCandidate(paper_id, clean_title, clean_abstract,
                          tuple(_clean_text(v, max_chars=200) for v in authors if _clean_text(v)),
                          year if type(year) is int and 1000 <= year <= 2200 else None, doi, (provider,), unique_locations,
                          {**ids, provider: str(native_id or "")}, count if type(count) is int else None,
                          _clean_text(kind or ""), provider if clean_abstract else "", _clean_text(item.get("publicationDate") or item.get("publication_date") or ""),
                          _clean_text(venue or ""), fields, item.get("is_retracted") if type(item.get("is_retracted")) is bool else None)

    values = candidate.to_dict()
    values["field_sources"] = {name: [{"provider": provider, "value": values[name]}]
        for name in ("title", "abstract", "authors", "year", "doi", "venue", "publication_date", "publication_type", "citation_count", "fields_of_study", "is_retracted")
        if values[name] is not None and values[name] != "" and values[name] != () and values[name] != []}
    return PaperCandidate.from_dict(values)


def merge_candidates(papers):
    """Merge work metadata, retaining distinct source locations and versions."""
    merged = {}
    for p in papers:
        key = p.doi or p.paper_id
        if key not in merged:
            merged[key] = p
            continue
        old = merged[key]
        values = old.to_dict()
        provenance = {key: list(rows) for key, rows in old.field_sources.items()}
        for field_name, rows in p.field_sources.items():
            combined = provenance.setdefault(field_name, [])
            for row in rows:
                if row not in combined:
                    combined.append(row)
        values["field_sources"] = provenance
        for name in ("abstract", "abstract_source", "year", "venue", "publication_date", "publication_type"):
            if not values.get(name):
                values[name] = getattr(p, name)
        values.update(providers=list(dict.fromkeys((*old.providers, *p.providers))),
                      locations=[asdict(v) for v in dict.fromkeys((*old.locations, *p.locations))],
                      external_ids={**p.external_ids, **old.external_ids},
                      authors=old.authors or p.authors,
                      citation_count=max((v for v in (old.citation_count, p.citation_count) if v is not None), default=None),
                      fields_of_study=list(dict.fromkeys((*old.fields_of_study, *p.fields_of_study))))
        merged[key] = PaperCandidate.from_dict(values)
    return tuple(merged.values())


def _matches(p: PaperCandidate, filters: SearchFilters):
    if filters.year_from is not None and (p.year is None or p.year < filters.year_from):
        return False
    if filters.year_to is not None and (p.year is None or p.year > filters.year_to):
        return False
    if filters.open_access_only and not any(l.is_oa is True for l in p.locations):
        return False
    norm = lambda v: re.sub(r"[^a-z0-9]", "", v.casefold())
    aliases = {"article": "journalarticle", "journalarticle": "journalarticle", "proceedingsarticle": "conference", "proceedings": "conference"}
    canonical = lambda v: aliases.get(norm(v), norm(v))
    if filters.publication_types and canonical(p.publication_type) not in {canonical(v) for v in filters.publication_types}:
        return False
    if filters.fields_of_study and not any(
        norm(wanted) in norm(actual) for wanted in filters.fields_of_study for actual in p.fields_of_study
    ):
        return False
    return True


def _fetch_provider(provider, query, doi, filters, limit, cursor, deadline):
    headers, params = {}, {}
    if provider == "semantic_scholar":
        base = "https://api.semanticscholar.org/graph/v1/paper/"
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key
        params["fields"] = "paperId,title,abstract,authors,year,venue,url,externalIds,openAccessPdf,isOpenAccess,citationCount,publicationDate,publicationTypes,fieldsOfStudy"
        if doi:
            url = base + quote("DOI:" + doi, safe="")
        else:
            url = base + "search"
            params.update(query=re.sub(r"[-‐‑‒–—]+", " ", query), limit=limit, offset=int(cursor or 0))
            if filters.year_from or filters.year_to:
                params["year"] = f"{filters.year_from or ''}-{filters.year_to or ''}"
            if filters.open_access_only:
                params["openAccessPdf"] = ""
            if filters.fields_of_study:
                params["fieldsOfStudy"] = ",".join(filters.fields_of_study)
    elif provider == "openalex":
        url = "https://api.openalex.org/works" + ("/" + quote("doi:" + doi, safe="") if doi else "")
        if settings.openalex_api_key:
            params["api_key"] = settings.openalex_api_key
        if not doi:
            params.update(search=query, per_page=limit, cursor=cursor or "*", sort="relevance_score:desc")
            parts = []
            if filters.year_from:
                parts.append(f"from_publication_date:{filters.year_from}-01-01")
            if filters.year_to:
                parts.append(f"to_publication_date:{filters.year_to}-12-31")
            if filters.open_access_only:
                parts.append("open_access.is_oa:true")
            if parts:
                params["filter"] = ",".join(parts)
    else:
        url = "https://api.crossref.org/works" + ("/" + quote(doi, safe="") if doi else "")
        if settings.crossref_mailto:
            params["mailto"] = settings.crossref_mailto
        if not doi:
            params.update({"query.bibliographic": query, "rows": limit, "offset": int(cursor or 0)})
            parts = []
            if filters.year_from:
                parts.append(f"from-pub-date:{filters.year_from}-01-01")
            if filters.year_to:
                parts.append(f"until-pub-date:{filters.year_to}-12-31")
            if parts:
                params["filter"] = ",".join(parts)
    payload = _request_json(url, params=params, headers=headers or None,
                            timeout=max(.01, deadline - time.monotonic()), deadline=deadline)
    if doi:
        raw = payload.get("message", {}) if provider == "crossref" else payload
        items, next_cursor = [raw], None
    elif provider == "semantic_scholar":
        items, next_cursor = payload.get("data") or [], payload.get("next")
    elif provider == "openalex":
        items = payload.get("results") or []
        next_cursor = (payload.get("meta") or {}).get("next_cursor") if items else None
    else:
        message = payload.get("message") or {}
        items = message.get("items") or []
        offset = int(cursor or 0) + len(items)
        next_cursor = offset if items and offset < int(message.get("total-results", offset)) else None
    papers = [p for raw in items if isinstance(raw, dict) for p in [_candidate(provider, raw)] if p is not None]
    papers = [p for p in papers if (not doi or p.doi == doi) and _matches(p, filters)]
    return papers, next_cursor


def discover_papers(query: str = "", *, doi: str | None = None, filters: SearchFilters | None = None,
                    provider: str = "auto", limit: int = 20, cursor: str | None = None,
                    timeout_seconds: float = 20, on_progress=None) -> DiscoveryResult:
    filters = filters or SearchFilters()
    if not isinstance(filters, SearchFilters):
        raise ValueError("filters must be SearchFilters")
    query = _clean_text(query, max_chars=500)
    normalized_doi = normalize_doi(doi) if doi else normalize_doi(query)
    if doi and not normalized_doi:
        raise ValueError("invalid DOI")
    if not query and not normalized_doi:
        raise ValueError("query or DOI required")
    if provider not in {"auto", "semantic_scholar", "openalex", "crossref"}:
        raise ValueError("unsupported provider")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120:
        raise ValueError("invalid timeout")
    if not settings.academic_search_enabled:
        return DiscoveryResult(warnings=("外部论文检索已关闭",), failed=True)
    # This endpoint is called only after the user explicitly starts external discovery.
    providers = ["semantic_scholar", "openalex", "crossref"] if provider == "auto" else [provider]
    signature = hashlib.sha256(json.dumps([query, normalized_doi, asdict(filters), providers], sort_keys=True).encode()).hexdigest()
    state = {p: ("*" if p == "openalex" else 0) for p in providers}
    turn = 0
    if cursor:
        try:
            if len(cursor) > 16000:
                raise ValueError()
            decoded = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
            if decoded["signature"] != signature or set(decoded["state"]) != set(providers):
                raise ValueError()
            state, turn = decoded["state"], int(decoded["turn"])
            for p, value in state.items():
                if value is not None and ((p != "openalex" and (type(value) is not int or value < 0 or value > 100000)) or (p == "openalex" and (not isinstance(value, str) or len(value) > 8000))):
                    raise ValueError()
        except (ValueError, KeyError, TypeError, UnicodeError):
            raise ValueError("invalid cursor or query/filter mismatch") from None
    active = [p for p in providers if state[p] is not None]
    if not active:
        return DiscoveryResult(providers=tuple(providers))
    active = active[turn % len(active):] + active[:turn % len(active)]
    quotas = {p: limit // len(active) + (i < limit % len(active)) for i, p in enumerate(active)}
    deadline = time.monotonic() + timeout_seconds
    executor = ThreadPoolExecutor(max_workers=len(active), thread_name_prefix="refmind-discovery")
    futures = {executor.submit(_fetch_provider, p, query, normalized_doi, filters, quotas[p], state[p], deadline): p for p in active if quotas[p]}
    warnings, statuses, pages = [], {p: "running" for p in active if quotas[p]}, {}
    pending = set(futures)
    try:
        while pending and time.monotonic() < deadline:
            done, pending = wait(pending, timeout=max(0, deadline-time.monotonic()), return_when=FIRST_COMPLETED)
            for future in done:
                p = futures[future]
                try:
                    pages[p], state[p] = future.result()
                    statuses[p] = "completed"
                except Exception as exc:
                    statuses[p] = "failed"
                    warnings.append(f"{p}: {type(exc).__name__}")
                if on_progress is not None:
                    on_progress(DiscoveryResult(merge_candidates(paper for key in providers for paper in pages.get(key, [])),
                        warnings=tuple(warnings), providers=tuple(providers), partial=True, provider_status=dict(statuses)))
        for future in pending:
            p = futures[future]
            future.cancel()
            statuses[p] = "timeout"
            warnings.append(f"{p}: deadline exceeded")
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
    if filters.fields_of_study:
        warnings.append("学科标签依赖来源分类；缺少可核验学科标签的记录不会通过该筛选")
    next_cursor = None
    if not normalized_doi and any(v is not None for v in state.values()):
        next_cursor = base64.urlsafe_b64encode(json.dumps({"signature": signature, "state": state, "turn": turn + 1}).encode()).decode()
    ok = sum(v == "completed" for v in statuses.values())
    return DiscoveryResult(merge_candidates(p for key in providers for p in pages.get(key, [])), next_cursor,
                           tuple(warnings), ok == 0, tuple(p for p in providers if p in statuses), ok < len(statuses), statuses)


def related_papers(candidate: PaperCandidate, *, kind: str = "references", limit: int = 20,
                   cursor: str | None = None, timeout_seconds: float = 20) -> DiscoveryResult:
    """Explicit one-hop Semantic Scholar relation lookup; never expands recursively."""
    if not isinstance(candidate, PaperCandidate) or kind not in {"references", "citations"}:
        raise ValueError("candidate and references/citations kind required")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("invalid relation limit")
    if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120:
        raise ValueError("invalid timeout")
    if not settings.academic_search_enabled:
        return DiscoveryResult(warnings=("外部论文检索已关闭",), failed=True)
    identifier = candidate.external_ids.get("semantic_scholar") or ("DOI:" + candidate.doi if candidate.doi else "")
    if not identifier:
        return DiscoveryResult(warnings=("论文缺少可追踪的DOI或Semantic Scholar标识",), failed=True)
    signature = hashlib.sha256((candidate.paper_id + ":" + kind).encode()).hexdigest()
    offset = 0
    if cursor:
        try:
            if len(cursor) > 4000:
                raise ValueError()
            state = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
            if state["signature"] != signature or type(state["offset"]) is not int or not 0 <= state["offset"] <= 100000:
                raise ValueError()
            offset = state["offset"]
        except (ValueError, KeyError, TypeError, UnicodeError):
            raise ValueError("invalid relation cursor") from None
    headers = {"x-api-key": settings.semantic_scholar_api_key} if settings.semantic_scholar_api_key else None
    try:
        payload = _request_json("https://api.semanticscholar.org/graph/v1/paper/" + quote(identifier, safe="") + "/" + kind,
            params={"limit": limit, "offset": offset, "fields": "paperId,title,abstract,authors,year,venue,url,externalIds,openAccessPdf,isOpenAccess,citationCount,publicationTypes,fieldsOfStudy"},
            headers=headers, timeout=timeout_seconds, deadline=time.monotonic() + timeout_seconds)
        relation_key = "citedPaper" if kind == "references" else "citingPaper"
        papers = [paper for entry in payload.get("data") or [] if isinstance(entry, dict)
                  for raw in [entry.get(relation_key)] if isinstance(raw, dict)
                  for paper in [_candidate("semantic_scholar", raw)] if paper is not None]
        next_offset = payload.get("next")
        next_cursor = base64.urlsafe_b64encode(json.dumps({"signature": signature, "offset": next_offset}).encode()).decode() if type(next_offset) is int else None
        return DiscoveryResult(merge_candidates(papers), next_cursor, providers=("semantic_scholar",), provider_status={"semantic_scholar": "completed"})
    except Exception as exc:
        return DiscoveryResult(warnings=("semantic_scholar: " + type(exc).__name__,), failed=True,
                               providers=("semantic_scholar",), provider_status={"semantic_scholar": "failed"})


__all__ = ["SearchFilters", "PaperLocation", "PaperCandidate", "DiscoveryResult", "discover_papers", "related_papers", "merge_candidates", "normalize_doi"]
