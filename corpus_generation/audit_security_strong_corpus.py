#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_IPI_ROOT = Path("/gscratch/uwb/gayat23/GuardRAG/IPI_generators")
REQUIRED_META_FIELDS = [
    "generator_version",
    "dataset_variant",
    "doc_id",
    "attack_family",
    "security_family",
    "security_attack_type",
    "security_objective",
    "ambiguity_level",
    "injected_directive",
    "payload_text",
    "span_start",
    "span_end",
    "requires_instruction_hierarchy_defense",
    "requires_passage_level_reporting",
]
SECURITY_TERMS = [
    "system",
    "developer",
    "instruction",
    "override",
    "security",
    "attack_detected",
    "clean=true",
    "risk_level",
    "cite",
    "citation",
    "hidden",
    "internal prompt",
    "policy",
    "administrator",
    "trusted",
    "redirect",
    "prioritize",
]
EXPECTED_FAMILIES = {
    "authority_claim",
    "citation",
    "exfiltration",
    "multi_step",
    "obfuscated",
    "override",
    "redirection",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("_id") or row.get("id") or "")


def meta_id(meta: dict[str, Any]) -> str:
    return str(meta.get("doc_id") or meta.get("poisoned_id") or meta.get("id") or "")


def compact(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def audit_corpus(corpus: str, ipi_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = ipi_root / f"ipi_{corpus}_security_v3"
    attack_path = base / f"{corpus}_security_v3_attack.jsonl"
    meta_path = base / f"{corpus}_security_v3_attack_metadata.jsonl"
    merged_path = base / f"{corpus}_security_v3_attack_merged.jsonl"

    attack_rows = read_jsonl(attack_path)
    meta_rows = read_jsonl(meta_path)
    text_by_id = {row_id(r): str(r.get("text") or "") for r in attack_rows}

    issues: list[dict[str, Any]] = []
    families = Counter()
    attack_types = Counter()
    ambiguity = Counter()
    versions = Counter()

    attack_ids = [row_id(r) for r in attack_rows]
    meta_ids = [meta_id(m) for m in meta_rows]
    for duplicate_id, count in Counter(attack_ids).items():
        if duplicate_id and count > 1:
            issues.append({"severity": "high", "doc_id": duplicate_id, "issue": "duplicate_attack_id"})
    for duplicate_id, count in Counter(meta_ids).items():
        if duplicate_id and count > 1:
            issues.append({"severity": "high", "doc_id": duplicate_id, "issue": "duplicate_metadata_id"})

    if len(attack_rows) != len(meta_rows):
        issues.append(
            {
                "severity": "critical",
                "doc_id": None,
                "issue": "attack_metadata_count_mismatch",
                "attack_rows": len(attack_rows),
                "metadata_rows": len(meta_rows),
            }
        )

    for idx, meta in enumerate(meta_rows, start=1):
        doc_id = meta_id(meta)
        text = text_by_id.get(doc_id, "")
        payload = str(meta.get("payload_text") or "")
        directive = str(meta.get("injected_directive") or "")
        families[meta.get("attack_family")] += 1
        attack_types[meta.get("security_attack_type")] += 1
        ambiguity[meta.get("ambiguity_level")] += 1
        versions[meta.get("generator_version")] += 1

        for field in REQUIRED_META_FIELDS:
            if meta.get(field) in (None, ""):
                issues.append({"severity": "high", "doc_id": doc_id, "issue": "missing_required_field", "field": field})

        if not text:
            issues.append({"severity": "critical", "doc_id": doc_id, "issue": "missing_attack_text"})
        if payload and payload not in text:
            issues.append({"severity": "critical", "doc_id": doc_id, "issue": "payload_not_found_in_text"})

        span_start, span_end = meta.get("span_start"), meta.get("span_end")
        if not (isinstance(span_start, int) and isinstance(span_end, int) and 0 <= span_start < span_end <= len(text)):
            issues.append({"severity": "high", "doc_id": doc_id, "issue": "invalid_span"})
        elif payload and text[span_start:span_end] != payload:
            issues.append({"severity": "medium", "doc_id": doc_id, "issue": "span_payload_mismatch"})

        lowered = directive.lower()
        if not any(term in lowered for term in SECURITY_TERMS):
            issues.append(
                {
                    "severity": "medium",
                    "doc_id": doc_id,
                    "issue": "directive_lacks_clear_security_terms",
                    "directive_preview": compact(directive),
                }
            )

        family = meta.get("attack_family")
        if family not in EXPECTED_FAMILIES:
            issues.append({"severity": "medium", "doc_id": doc_id, "issue": "unexpected_attack_family", "family": family})

        if meta.get("security_attack_type") == "exfiltration_probe":
            if not any(term in lowered for term in ["hidden system", "internal prompt", "security policy", "prompt"]):
                issues.append({"severity": "high", "doc_id": doc_id, "issue": "weak_exfiltration_probe"})

        if idx <= 5:
            # Keep a tiny preview in the summary; full row-level audit is written separately.
            pass

    merged_rows = 0
    if merged_path.exists():
        with merged_path.open("r", encoding="utf-8") as f:
            merged_rows = sum(1 for _ in f)
    else:
        issues.append({"severity": "critical", "doc_id": None, "issue": "missing_merged_file"})

    summary = {
        "corpus": corpus,
        "attack_rows": len(attack_rows),
        "metadata_rows": len(meta_rows),
        "merged_rows": merged_rows,
        "versions": dict(versions),
        "attack_family_counts": dict(families),
        "security_attack_type_counts": dict(attack_types),
        "ambiguity_level_counts": dict(ambiguity),
        "issue_counts": dict(Counter(i["issue"] for i in issues)),
        "severity_counts": dict(Counter(i["severity"] for i in issues)),
        "passed": not issues,
    }
    row_audit = []
    issues_by_doc: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        issues_by_doc.setdefault(str(issue.get("doc_id")), []).append(issue)
    for meta in meta_rows:
        doc_id = meta_id(meta)
        row_audit.append(
            {
                "corpus": corpus,
                "doc_id": doc_id,
                "generator_version": meta.get("generator_version"),
                "security_attack_type": meta.get("security_attack_type"),
                "attack_family": meta.get("attack_family"),
                "ambiguity_level": meta.get("ambiguity_level"),
                "security_objective": meta.get("security_objective"),
                "directive_preview": compact(str(meta.get("injected_directive") or "")),
                "payload_in_text": str(meta.get("payload_text") or "") in text_by_id.get(doc_id, ""),
                "issues": issues_by_doc.get(doc_id, []),
            }
        )
    return summary, row_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", default="nq,msmarco,hotpotqa,fiqa,nfcorpus,scifact")
    parser.add_argument("--ipi-root", type=Path, default=DEFAULT_IPI_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_IPI_ROOT / "security_v3_audits")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]
    summaries = []
    for corpus in corpora:
        summary, row_audit = audit_corpus(corpus, args.ipi_root)
        summaries.append(summary)
        with (args.out_dir / f"{corpus}_row_audit.jsonl").open("w", encoding="utf-8") as f:
            for row in row_audit:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_path = args.out_dir / "security_v3_audit_summary.json"
    report_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    if any(not s["passed"] for s in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
