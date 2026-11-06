"""Freeze pointers to 18 existing DEV questions; never run models or approve labels."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SELECTION = {
    "real_v3_draft.json": ["D7-Q01", "D4-Q01", "D6-Q02", "D7-Q04", "D7-Q09", "D7-Q10", "D7-Q06", "D5-Q06", "D7-Q07", "D4-Q07"],
    "public_materials_hts_dev_v3.json": ["PUBLIC-HTS-01", "PUBLIC-HTS-05", "PUBLIC-LLZO-01",
        "PUBLIC-HTS-04", "PUBLIC-LLZO-10", "PUBLIC-HTS-08", "PUBLIC-LLZO-11", "PUBLIC-LLZO-12"],
}


def prepare(source_root: Path):
    items, datasets = [], []
    for filename, ids in SELECTION.items():
        path = source_root / "evaluation" / "datasets" / filename
        raw = path.read_bytes()
        payload = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        by_id = {item["id"]: item for item in payload["items"]}
        datasets.append({"path": str(path), "sha256": digest, "schema_version": payload["schema_version"]})
        for item_id in ids:
            item = by_id[item_id]
            if item["question_split"] != "dev":
                raise ValueError("Development ablation cannot consume holdout questions")
            items.append({"id": item_id, "question": item["question"], "category": item["category"],
                "question_split": item["question_split"], "source_dataset": filename,
                "source_dataset_sha256": digest,
                "source_item_sha256": hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "corpus_scope": item["corpus_scope"], "annotation_status": item.get("annotation_status"),
                "human_review": item.get("human_review"), "evidence_ids": [entry["id"] for entry in item.get("evidence", [])],
                "execution_status": "not_run", "source_quotes_rechecked_in_this_run": False})
    if len(items) != 18:
        raise ValueError("Expected exactly 18 planned questions")
    return {"schema": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned_not_executed", "purpose": "development_ablation_not_independent_acceptance",
        "online_model_calls": 0, "datasets": datasets, "items": items,
        "configurations": [
            {"id": "baseline", "RESEARCH_MULTI_QUERY_ENABLED": False, "RESEARCH_EXTRACTIVE_COMPRESSION_ENABLED": False},
            {"id": "multiquery_only", "RESEARCH_MULTI_QUERY_ENABLED": True, "RESEARCH_EXTRACTIVE_COMPRESSION_ENABLED": False},
            {"id": "compression_only", "RESEARCH_MULTI_QUERY_ENABLED": False, "RESEARCH_EXTRACTIVE_COMPRESSION_ENABLED": True},
            {"id": "multiquery_compression", "RESEARCH_MULTI_QUERY_ENABLED": True, "RESEARCH_EXTRACTIVE_COMPRESSION_ENABLED": True}],
        "fixed_controls": ["same committed index snapshot and source scope", "same original questions and embedding/rerank/answer models",
            "same EvidenceBundle budget", "retain selected-source semantics; never infer selection from positive evidence",
            "record plan failure, branch failure, score kinds and missing observations"],
        "pending_gates": ["Recheck PDF physical pages, quotes and citation locations against frozen source hashes",
            "Preserve existing dataset human-review and independent-acceptance gates",
            "Add new switches to the existing evaluation runtime allowlist and fingerprint before any real run",
            "No API call executed here; follow the existing runner's explicit allow-api gate",
            "Report retrieval coverage, scoped answer correctness, citation span correctness, latency and token cost separately",
            "Do not present synthetic mechanism pass rates as paper-corpus quality gains"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"questions": len(result["items"]), "status": result["status"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
