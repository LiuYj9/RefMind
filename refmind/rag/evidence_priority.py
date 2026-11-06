"""Experimental, stable evidence ordering; never delete, rewrite or infer targets.

This pure module does not open images/files, use providers, or consume source
identity, page, title, annotation or answer fields. The caller owns the feature
flag, which must remain disabled by default. A uses explicit metadata only;
B_safe additionally recognizes conservative bibliographic text.
"""
from __future__ import annotations

import re
from typing import Literal

from langchain_core.documents import Document

PriorityMode = Literal['references', 'empty_figures', 'combined']
DetectionStrategy = Literal['A', 'B_safe']

_REFERENCE_SECTION = re.compile(
    r'^(?:(?:\d+(?:\.\d+)*|[IVXLC]+)\s*[.)]?\s*)?(?:references|bibliography|参考文献)$', re.I)
_ENTRY = re.compile(r'(?m)^\s*\[\d{1,3}\]\s+')
_YEAR = re.compile(r'\b(?:19|20)\d{2}\b')
_PUBLICATION = re.compile(r'\b(?:IEEE|doi|arXiv|Accessed)\b|\b(?:vol|pp|Proc|Trans)\.|https?://', re.I)
_BODY_SIGNAL = re.compile(
    r'\b(?:we\s+(?:propose|present|show|evaluate|use|consider|observed)|'
    r'our\s+(?:method|model|approach|results|study)|this\s+(?:paper|study|work))\b', re.I)
_PROSE_PREFIX = re.compile(r'\b(?:we|our|is|are|was|were|can|could|should|must|therefore|however|thus)\b', re.I)
_REFERENCE_QUERY = re.compile(
    r'参考文献|引用|引文|文献追踪|出处|溯源|参考来源|'
    r'\b(?:citat(?:ion|ions)|cite[ds]?|citing|references?|bibliograph(?:y|ies)|doi)\b', re.I)
_VISUAL_QUERY = re.compile(
    r'图|表|曲线|图像|\b(?:fig(?:ure)?s?\.?|tables?|plots?|charts?|images?|diagrams?)\b', re.I)
_EMPTY_FIGURE = {
    '', 'figure',
    'figure visual summary unavailable; original figure is stored for multimodal answering.',
    'visual summary unavailable; original figure is stored for multimodal answering.',
}


def _confirmed_reference_prefix(prefix: str) -> bool:
    residual = re.sub(r'(?im)^\s*(?:references|bibliography|参考文献)\s*$', '', prefix).strip()
    if not residual:
        return True
    return bool(_YEAR.search(residual) and _PUBLICATION.search(residual)
                and re.search(r'["“”]|\b(?:vol|pp)\.|https?://|\bdoi\b', residual, re.I)
                and not _PROSE_PREFIX.search(residual))


def evidence_kind(document: Document, *, strategy: DetectionStrategy = 'B_safe') -> dict:
    if strategy not in {'A', 'B_safe'}:
        raise ValueError('strategy must be A or B_safe')
    metadata = document.metadata or {}
    text = str(document.page_content or '')
    content_type = str(metadata.get('content_type') or metadata.get('block_type') or '').strip().casefold()
    content_type = {'image': 'figure', 'caption': 'figure_caption', 'formula': 'equation'}.get(content_type, content_type)
    sections = [str(metadata.get(key) or '').strip() for key in ('section_path', 'section', 'section_title')]
    image_declared = bool(metadata.get('image_path') or metadata.get('image_paths') or metadata.get('image_id'))
    if content_type in {'table', 'equation'}:
        return {'kind': 'body_or_uncertain', 'reason': 'preserve_scientific_structure', 'original_image_declared': image_declared}
    if content_type in {'reference', 'references', 'bibliography'} or any(_REFERENCE_SECTION.fullmatch(value) for value in sections):
        return {'kind': 'reference', 'reason': 'explicit_reference_metadata', 'original_image_declared': image_declared}
    if strategy == 'B_safe':
        entries = list(_ENTRY.finditer(text))
        prefix = text[:entries[0].start()] if entries else text
        if (len(entries) >= 2 and len(prefix.strip()) <= 240
                and len(_YEAR.findall(text)) >= 2 and len(_PUBLICATION.findall(text)) >= 2
                and _confirmed_reference_prefix(prefix) and not _BODY_SIGNAL.search(text)):
            return {'kind': 'reference', 'reason': 'confirmed_numbered_publication_entries', 'original_image_declared': image_declared}
    compact = re.sub(r'\s+', ' ', text).strip().casefold()
    if content_type in {'figure', 'figure_caption'} and compact in _EMPTY_FIGURE:
        return {'kind': 'empty_figure', 'reason': 'exact_empty_figure_placeholder', 'original_image_declared': image_declared}
    return {'kind': 'body_or_uncertain', 'reason': 'retain_body_or_uncertain_text', 'original_image_declared': image_declared}


def evidence_priority_reason(document: Document, question: str, *, mode: PriorityMode = 'combined',
                             strategy: DetectionStrategy = 'B_safe') -> dict:
    if mode not in {'references', 'empty_figures', 'combined'}:
        raise ValueError('unknown evidence priority mode')
    classification = evidence_kind(document, strategy=strategy)
    if classification['original_image_declared']:
        priority, reason = 0, 'original_image_declared_protected'
    elif classification['kind'] == 'reference' and _REFERENCE_QUERY.search(question):
        priority, reason = 0, 'reference_lookup_query_protected'
    elif mode in {'references', 'combined'} and classification['kind'] == 'reference':
        priority, reason = 1, 'reference_only_deprioritized'
    elif classification['kind'] == 'empty_figure' and _VISUAL_QUERY.search(question):
        priority, reason = 0, 'visual_query_protected'
    elif mode in {'empty_figures', 'combined'} and classification['kind'] == 'empty_figure':
        priority, reason = 2, 'empty_figure_without_original_deprioritized'
    else:
        priority, reason = 0, 'retain_original_relative_priority'
    return {**classification, 'classification_reason': classification['reason'], 'reason': reason, 'priority': priority}


def evidence_priority(document: Document, question: str, *, mode: PriorityMode = 'combined',
                      strategy: DetectionStrategy = 'B_safe') -> int:
    return evidence_priority_reason(document, question, mode=mode, strategy=strategy)['priority']


def prioritize_evidence(question: str, documents: list[Document], *, mode: PriorityMode = 'combined',
                        strategy: DetectionStrategy = 'B_safe') -> list[Document]:
    """Return the original objects in a stable order; caller must gate activation."""
    return sorted(documents, key=lambda document: evidence_priority(document, question, mode=mode, strategy=strategy))
