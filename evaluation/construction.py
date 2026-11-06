"""Construction quotas and truthful inventories; target slots are never fabricated questions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

DOMAINS = {
    "superconductivity_physics": {"families": 16, "dev": 120, "id_holdout": 40, "ood_holdout": 0},
    "materials_energy": {"families": 16, "dev": 120, "id_holdout": 40, "ood_holdout": 0},
    "computer_science_ml": {"families": 16, "dev": 120, "id_holdout": 40, "ood_holdout": 0},
    "mechanical_thermofluids": {"families": 6, "dev": 0, "id_holdout": 0, "ood_holdout": 60},
    "chemical_process": {"families": 6, "dev": 0, "id_holdout": 0, "ood_holdout": 60},
}
CORE_CATEGORIES = {"numerical": 100, "method": 70, "table": 80, "equation": 60,
                   "figure": 50, "comparison": 70, "summary": 50, "unanswerable": 120}
TRACK_TARGETS = {
    "local": {"unit": "primary_question", "total": 600, "splits": {"dev": 360, "id_holdout": 120, "ood_holdout": 120}},
    "external": {"unit": "external_task", "total": 120, "splits": {"dev": 72, "id_holdout": 24, "ood_holdout": 24}},
    "dialogue": {"unit": "four_turn_scenario", "total": 30, "turns_per_scenario": 4,
                 "splits": {"dev": 18, "id_holdout": 6, "ood_holdout": 6}},
    "structure": {"unit": "annotated_pdf_page", "total": 120,
                  "splits": {"dev": 72, "id_holdout": 24, "ood_holdout": 24}},
}


def manifest():
    return {"schema_version": 3, "kind": "construction_manifest", "status": "incomplete",
            "tracks": TRACK_TARGETS, "core_domains": DOMAINS, "core_categories": CORE_CATEGORIES,
            "dataset_files": ["evaluation/datasets/real_v3_draft.json"],
            "counting_rules": ["variants/provider runs/seeds are not new primary questions",
                "four dialogue turns count as one scenario", "structure samples count by PDF page",
                "targets do not represent authored or human-approved records",
                "legacy categories remain unassigned until reviewed; do not label every fact numerical"],
            "legacy": {"source": "evaluation/datasets/real_v1.json", "items": 100,
                       "dev": 70, "id_holdout": 30, "preserve_original_bytes": True}}


def migrate_legacy(source, output):
    raw_bytes = source.read_bytes()
    old = json.loads(raw_bytes.decode("utf-8-sig"))
    if old.get("schema_version") != 2:
        raise ValueError("migration only accepts the content-pinned v2 dataset")
    from .dataset import canonical_split
    sources = [{**entry, "question_split": canonical_split(entry["split"]),
                "domain": "superconductivity_physics"} for entry in old["corpus"]]
    by_name = {Path(s["path"]).name.casefold(): s for s in sources}
    by_id = {s["id"]: s for s in sources}
    items = []
    # Only categories with a direct semantic match migrate automatically.
    direct_categories = {"terminology": "method", "comparison": "comparison", "unanswerable": "unanswerable"}
    for raw in old["items"]:
        split = canonical_split(raw["split"])
        evidence = [{**ev, "id": f"{raw['id']}:e{index}",
            "source_id": by_name[ev["filename"].casefold()]["id"], "evidence_source": "pdf_text"}
            for index, ev in enumerate(raw.get("evidence", []), 1)]
        families = {by_id[ev["source_id"]]["family"] for ev in evidence}
        if raw.get("source_group") in by_id:
            families.add(by_id[raw["source_group"]]["family"])
        if not families:
            raise ValueError("legacy item cannot be associated with a source family")
        item = {**raw, "question_split": split, "split": split, "track": "local",
            "domain": "superconductivity_physics", "family_ids": sorted(families),
            "corpus_scope": [s["id"] for s in sources if split != "dev" or s["question_split"] == "dev"],
            "category": direct_categories.get(raw["category"], "legacy_" + raw["category"]),
            "annotation_status": raw.get("annotation_status", old.get("annotation_status", "draft")),
            "human_review": {"status": "pending"}, "evidence": evidence, "fact_slots": [],
            "task_metadata": {"legacy_category": raw["category"], "fact_slot_status": "pending_human_decomposition",
                              "legacy_dataset_sha256": hashlib.sha256(raw_bytes).hexdigest()}}
        if raw.get("expect_refusal"):
            item["negative_reason"] = "legacy_negative_requires_scope_and_difficulty_review"
        items.append(item)
    result = {"schema_version": 3, "annotation_status": "draft_pending_human_review",
        "page_numbering": "physical_pdf_1_based", "page_tolerance": old.get("page_tolerance", 0),
        "migration": {"source": source.name, "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                      "original_items": len(items), "original_partition_preserved": True,
                      "fact_slots_inferred": False}, "corpus": sources, "items": items}
    encoded = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != encoded:
        raise ValueError("migration output already exists and differs; preserve it and choose a new filename")
    output.write_bytes(encoded)
    return {"items": len(items), "splits": dict(Counter(i["question_split"] for i in items)),
            "human_approved": 0, "fact_slots_inferred": False, "output": str(output)}


def inventory(config, root):
    records, seen, family_splits, exposed = [], set(), {}, set()
    for rel in config.get("dataset_files", []):
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("dataset inventory path escapes project")
        if not path.exists():
            continue
        from .dataset import load_golden_set
        golden = load_golden_set(path, project_root=root)
        exposed.update(golden.exposed_item_ids())
        for source in golden.manifest:
            family_splits.setdefault(source["family"], set()).add(source.get("question_split", source.get("split", "dev")))
        for item in golden.items:
            if item.id in seen:
                raise ValueError("duplicate item ID across construction inventories")
            seen.add(item.id); records.append(item)
            for family in item.family_ids:
                family_splits.setdefault(family, set()).add(item.question_split)
    if any(len(partitions) != 1 for partitions in family_splits.values()):
        raise ValueError("family leaked across dataset files or evaluation tracks")
    tracks = {}
    for track, target in config["tracks"].items():
        selected = [item for item in records if item.track == track]
        approved = sum(item.annotation_status == "human_approved" and item.review.get("status") == "approved" for item in selected)
        counts = Counter(item.question_split for item in selected)
        tracks[track] = {"unit": target["unit"], "target": target["total"], "authored": len(selected),
                         "human_approved": approved, "missing_authored": max(0, target["total"] - len(selected)),
                         "missing_review": len(selected) - approved,
                         "splits": {s: {"target": n, "authored": counts[s], "missing": max(0, n-counts[s])}
                                    for s, n in target["splits"].items()}}
        tracks[track]["exposed_historical_holdout"]=sum(i.id in exposed for i in selected)
        tracks[track]["independent_holdout_candidate_pool"]={s:sum(i.question_split==s and i.id not in exposed for i in selected)
            for s in ("id_holdout","ood_holdout")}
        tracks[track]["independent_holdout_missing"]={s:max(0,target["splits"][s]-tracks[track]["independent_holdout_candidate_pool"][s])
            for s in ("id_holdout","ood_holdout")}
    category_counts = Counter(item.category for item in records if item.track == "local")
    domains={}
    for domain,target in config["core_domains"].items():
        selected=[item for item in records if item.track=="local" and item.domain==domain]
        counts=Counter(item.question_split for item in selected)
        domains[domain]={"target_families":target["families"],"observed_question_families":len({f for item in selected for f in item.family_ids}),
            "exposed_historical_holdout":sum(i.id in exposed for i in selected),
            "splits":{s:{"target":target[s],"authored_pool":counts[s],"missing":max(0,target[s]-counts[s]),
                          "pool_excess":max(0,counts[s]-target[s])} for s in ("dev","id_holdout","ood_holdout")}}
    return {"status": "incomplete" if any(t["missing_authored"] or t["missing_review"] for t in tracks.values()) else "complete",
            "tracks": tracks, "core_categories": {c: {"target": n, "assigned": category_counts[c],
            "unfilled_category_slots": max(0, n-category_counts[c]),"draft_pool_excess":max(0,category_counts[c]-n)} for c,n in config["core_categories"].items()},
            "core_domains":domains,"inventory_is_not_frozen_benchmark_selection":True,
            "legacy_category_review_pending": sum(n for c,n in category_counts.items() if c.startswith("legacy_"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("migrate", "manifest", "status"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.action == "migrate":
        result = migrate_legacy(args.source or args.root/"evaluation/datasets/real_v1.json",
                                args.output or args.root/"evaluation/datasets/real_v3_draft.json")
    elif args.action == "manifest":
        result = manifest()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.exists() and json.loads(args.output.read_text(encoding="utf-8")) != result:
                raise ValueError("construction manifest already exists with different contents")
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    else:
        config_path = args.root / "evaluation/datasets/construction_v3.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else manifest()
        result = inventory(config, args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
