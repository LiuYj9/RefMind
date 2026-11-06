"""Validated, content-pinned evaluation datasets; no application/model imports."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


def normalize_text(value: str) -> str:
    import unicodedata
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    ground_truth: str
    evidence: tuple[tuple[str, int], ...]
    expect_refusal: bool = False
    split: str = "dev"
    category: str = "legacy"
    evidence_details: tuple[dict, ...] = ()
    track: str = "local"
    domain: str = "legacy"
    family_ids: tuple[str, ...] = ()
    corpus_scope: tuple[str, ...] = ()
    annotation_status: str = "draft"
    review: dict = field(default_factory=dict)
    fact_slots: tuple[dict, ...] = ()
    task_metadata: dict = field(default_factory=dict)

    @property
    def question_split(self) -> str:
        return canonical_split(self.split)


def canonical_split(value: str) -> str:
    return "id_holdout" if value == "holdout" else value


@dataclass(frozen=True)
class GoldenSet:
    corpus: tuple[str, ...]
    items: tuple[GoldenItem, ...]
    page_tolerance: int
    manifest: tuple[dict, ...]
    dataset_hash: str
    schema_version: int
    independence_audit: dict = field(default_factory=dict)

    def select(self, split: str = "dev", unlock_holdout: bool = False) -> "GoldenSet":
        if split not in {"dev", "holdout", "id_holdout", "ood_holdout", "all"}:
            raise ValueError("invalid question split")
        selected = tuple(i for i in self.items if split == "all" or i.question_split == canonical_split(split))
        if any(i.question_split != "dev" for i in selected) and not unlock_holdout:
            raise ValueError("holdout is locked; use --unlock-holdout only for release acceptance")
        if not selected:
            raise ValueError("selected split is empty")
        return GoldenSet(self.corpus, selected, self.page_tolerance, self.manifest,
                         self.dataset_hash, self.schema_version, self.independence_audit)

    def scoped_manifest(self) -> tuple[dict, ...]:
        """Retrieval corpus is explicit; acceptance may include all frozen distractors."""
        if self.schema_version < 3:
            partitions = {item.question_split for item in self.items}
            return tuple(s for s in self.manifest if canonical_split(s.get("split", "dev")) in partitions)
        ids = {source_id for item in self.items for source_id in item.corpus_scope}
        return tuple(s for s in self.manifest if s["id"] in ids)

    def exposed_item_ids(self) -> set[str]:
        exposed=[]
        for item in self.items:
            for event in self.independence_audit.get("events",[]):
                if (event.get("eligible_for_blind_acceptance") is False
                    and item.question_split==event.get("affected_question_split")
                    and set(item.family_ids)&set(event.get("affected_family_ids",[]))):
                    exposed.append(item.id);break
        return set(exposed)

    def require_review(self, *, allow_draft: bool = False, independent_acceptance: bool = False) -> dict:
        """Draft opt-in is development-only; holdout unlock never bypasses human review."""
        exposed=self.exposed_item_ids()
        independent_eligible = self.schema_version>=3 and not exposed and all(i.question_split!="dev" for i in self.items)
        if independent_acceptance and not independent_eligible:
            raise ValueError("independent acceptance requires v3 unseen holdout questions; exposed historical regression or development items cannot be approved into independence")
        approved = 0
        for item in self.items:
            review = item.review
            valid = (item.annotation_status == "human_approved"
                     and review.get("status") == "approved"
                     and bool(str(review.get("reviewer", "")).strip())
                     and bool(str(review.get("reviewed_at", "")).strip()))
            if valid and self.schema_version >= 3 and item.track == "local" and not item.expect_refusal:
                valid = bool(item.fact_slots)
            if valid:
                approved += 1
            elif not allow_draft or item.question_split != "dev":
                raise ValueError("human review required; --allow-draft is only valid for dev diagnostics")
        complete=approved == len(self.items)
        return {"official": complete and not exposed, "human_review_complete":complete,"approved": approved,
                "draft": len(self.items) - approved,
                "label": "historical_regression" if exposed else "reviewed_evaluation" if complete else "draft_diagnostic",
                "independent_eligible":independent_eligible,"blind_acceptance_requested":independent_acceptance,
                "exposed_items":len(exposed),"independence_audit":self.independence_audit}


def load_golden_set(path: str | Path, *, project_root: Path | None = None,
                    verify_files: bool = False) -> GoldenSet:
    path = Path(path).resolve()
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    raw_bytes = path.read_bytes()
    data = json.loads(raw_bytes.decode("utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("dataset must be an object")
    version = data.get("schema_version", 1)
    if version == 3:
        from .v3_contract import load_v3
        return load_v3(data, raw_bytes, root, verify_files=verify_files)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported dataset schema_version")
    tolerance = data.get("page_tolerance", 0)
    if type(tolerance) is not int or tolerance < 0:
        raise ValueError("page_tolerance must be a nonnegative integer")
    manifest, names, ids, hashes = [], set(), set(), set()
    if not isinstance(data.get("corpus"), list) or not data["corpus"]:
        raise ValueError("corpus must be a nonempty list")
    for raw in data["corpus"]:
        if not isinstance(raw, (str, dict)):
            raise ValueError("corpus entries must be paths or objects")
        entry = {"path": raw} if isinstance(raw, str) else dict(raw)
        rel = entry.get("path")
        if not isinstance(rel, str) or not rel:
            raise ValueError("corpus path is required")
        resolved = (root / rel).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("corpus path escapes project root")
        name = resolved.name.casefold()
        if name in names:
            raise ValueError("ambiguous duplicate corpus filename")
        names.add(name)
        if version == 2:
            if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
                raise ValueError("v2 corpus must have SHA256")
            if entry["sha256"] in hashes:
                raise ValueError("duplicate corpus bytes, possibly leaking across splits")
            hashes.add(entry["sha256"])
            if not entry.get("id") or entry["id"] in ids:
                raise ValueError("v2 corpus IDs must be unique")
            ids.add(entry["id"])
            if type(entry.get("pages")) is not int or entry["pages"] < 1:
                raise ValueError("v2 corpus pages must be positive")
            if entry.get("split") not in {"dev", "holdout"} or not entry.get("family"):
                raise ValueError("v2 corpus split/family required")
        entry["path"] = rel
        manifest.append(entry)
    by_name = {Path(e["path"]).name.casefold(): e for e in manifest}
    items, seen_ids, seen_questions = [], set(), set()
    if not isinstance(data.get("items"), list) or not data["items"]:
        raise ValueError("items must be a nonempty list")
    for raw in data["items"]:
        if not isinstance(raw, dict):
            raise ValueError("items must be objects")
        item_id, question = raw.get("id"), raw.get("question")
        if not isinstance(item_id, str) or not item_id or item_id in seen_ids:
            raise ValueError("item ID missing or duplicated")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{item_id}: question missing")
        if normalize_text(question) in seen_questions:
            raise ValueError(f"{item_id}: duplicate question")
        seen_ids.add(item_id)
        seen_questions.add(normalize_text(question))
        refusal = raw.get("expect_refusal", False)
        if type(refusal) is not bool:
            raise ValueError(f"{item_id}: expect_refusal must be boolean")
        split = raw.get("split", "dev")
        if split not in {"dev", "holdout"}:
            raise ValueError(f"{item_id}: invalid split")
        truth = raw.get("ground_truth", "")
        if not isinstance(truth, str) or not truth.strip():
            raise ValueError(f"{item_id}: ground_truth missing")
        evidence = raw.get("evidence", [])
        if not isinstance(evidence, list) or (refusal and evidence) or (not refusal and not evidence):
            raise ValueError(f"{item_id}: evidence/refusal mismatch")
        pairs = []
        for ev in evidence:
            if not isinstance(ev, dict):
                raise ValueError(f"{item_id}: evidence must be an object")
            name, page = ev.get("filename"), ev.get("page")
            if not isinstance(name, str) or name.casefold() not in by_name:
                raise ValueError(f"{item_id}: evidence filename absent from corpus")
            source = by_name[name.casefold()]
            if type(page) is not int or page < 1 or page > source.get("pages", 10**9):
                raise ValueError(f"{item_id}: invalid evidence page")
            if version == 2:
                if split != source["split"]:
                    raise ValueError(f"{item_id}: evidence crosses split boundary")
                if not isinstance(ev.get("quote"), str) or len(normalize_text(ev["quote"])) < 8:
                    raise ValueError(f"{item_id}: source quote required")
            pairs.append((name, page))
        if len(set(pairs)) != len(pairs):
            raise ValueError(f"{item_id}: duplicated evidence")
        items.append(GoldenItem(item_id, question, truth, tuple(pairs), refusal, split,
                                raw.get("category", "legacy"), tuple(evidence),
                                annotation_status=raw.get("annotation_status", data.get("annotation_status", "draft")),
                                review=raw.get("human_review", {})))
    families: dict[str, set] = {}
    for e in manifest:
        if "family" in e:
            families.setdefault(e["family"], set()).add(e["split"])
    if any(len(splits) > 1 for splits in families.values()):
        raise ValueError("document family leaked across splits")
    golden = GoldenSet(tuple(e["path"] for e in manifest), tuple(items), tolerance,
                       tuple(manifest), hashlib.sha256(raw_bytes).hexdigest(), version)
    if verify_files:
        verify_corpus(golden, root)
    return golden


def verify_corpus(golden: GoldenSet, root: Path, *, scoped: bool = False) -> None:
    """Verify bytes, physical page numbers and exact normalized excerpts, offline."""
    import fitz
    pages = {}
    for source in golden.scoped_manifest() if scoped else golden.manifest:
        p = (root / source["path"]).resolve()
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if source.get("sha256") and source["sha256"] != actual:
            raise ValueError(f"corpus content changed: {p.name}")
        with fitz.open(p) as doc:
            if source.get("pages") and source["pages"] != len(doc):
                raise ValueError(f"corpus page count changed: {p.name}")
            pages[p.name.casefold()] = [normalize_text(pg.get_text()) for pg in doc]
    for item in golden.items:
        for ev in item.evidence_details:
            source_pages = pages[ev["filename"].casefold()]
            if ev["page"] > len(source_pages):
                raise ValueError(f"{item.id}: page outside PDF")
            if ev.get("evidence_source", "pdf_text") == "visual":
                # Visual observations require a pinned region and human review, not fake native PDF text.
                continue
            if ev.get("quote") and normalize_text(ev["quote"]) not in source_pages[ev["page"]-1]:
                raise ValueError(f"{item.id}: source quote not present on evidence page")
