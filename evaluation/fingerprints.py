"""Report fingerprints for the evaluation and scientific pipeline dependency set.

This is report provenance only. Index compatibility remains owned by worker.fingerprint.
Credentials, environment files, generated data and model weights are never read here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPORT_SOURCE_FILES = (
    "evaluation/dataset.py", "evaluation/v3_contract.py", "evaluation/evaluator.py",
    "evaluation/run_eval.py", "evaluation/worker.py", "evaluation/runtime.py",
    "evaluation/classification.py", "evaluation/suites.py", "evaluation/run_suite.py",
    "evaluation/fingerprints.py", "evaluation/metrics/fact_evidence.py",
    "evaluation/metrics/retrieval.py", "evaluation/metrics/generation.py",
    "refmind/rag/__init__.py", "refmind/rag/research_pipeline.py",
    "refmind/rag/scientific_retrieval.py", "refmind/rag/graph.py",
    "refmind/rag/evidence_bundle.py", "refmind/rag/evidence_priority.py", "refmind/rag/reranker.py",
    "refmind/rag/retrieval.py", "refmind/rag/compression.py", "refmind/rag/provenance.py",
    "refmind/rag/document_processor.py", "refmind/rag/layout_chunker.py", "refmind/rag/memory.py",
    "refmind/parsing/pdf_parser.py", "refmind/agents/__init__.py", "refmind/agents/orchestration.py",
    "refmind/citations.py", "refmind/contracts.py", "refmind/llm/__init__.py",
    "refmind/llm/factory.py", "refmind/llm/openai_metering.py", "refmind/llm/image_understanding.py",
    "refmind/config/__init__.py", "refmind/config/settings.py",
    "refmind/storage/__init__.py", "refmind/storage/repository.py", "refmind/storage/connection.py",
    "refmind/storage/models.py", "refmind/services/paper_discovery.py", "refmind/services/fulltext.py",
)


def implementation_hashes(project_root: Path | None = None) -> dict[str, str]:
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in REPORT_SOURCE_FILES}
