#!/usr/bin/env python3
# Author: Gayatri Malladi
#
# Build curated blackbox poisoning-rate variants from finalized RIPE-II
# main-candidate corpora. The key design choice is to keep the full canonical
# query set while only subsetting poisoned carriers, so lower poisoning rates
# reduce exposure through missing poisoned documents rather than by shrinking
# the evaluation workload.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path("/mmfs1/home/gayat23/projects/guardrag-thesis")
IPI_ROOT = ROOT / "IPI_generators"
RESULTS_ROOT = ROOT / "results"
BEIR_ROOT = ROOT / "data" / "corpus" / "beir"
DEFAULT_OUTPUT_ROOT = Path("/gscratch/uwb/gayat23/GuardRAG/IPI_generators/rate_sweep_blackbox")


def corpus_paths(corpus: str) -> Dict[str, Path]:
    ipi_dir = IPI_ROOT / f"ipi_{corpus}_main"
    scratch_ipi_dir = Path("/gscratch/uwb/gayat23/GuardRAG/IPI_generators") / f"ipi_{corpus}_main"
    merged_name = f"{corpus}_main_attack_merged.jsonl"
    scratch_merged = scratch_ipi_dir / merged_name
    local_merged = ipi_dir / merged_name
    return {
        "attack": ipi_dir / f"{corpus}_main_attack.jsonl",
        "metadata": ipi_dir / f"{corpus}_main_attack_metadata_v2.jsonl",
        "manifest": ipi_dir / f"{corpus}_main_attack_manifest.json",
        "summary": ipi_dir / f"{corpus}_main_summary.json",
        "queries": RESULTS_ROOT / f"{corpus}_main_queries_beir.jsonl",
        "queries_summary": RESULTS_ROOT / f"{corpus}_main_queries_beir_summary.json",
        # Prefer scratch when available, but fall back to the repo copy for
        # corpora whose finalized merged artifact was not mirrored to scratch.
        "merged": scratch_merged if scratch_merged.exists() else local_merged,
        "clean": BEIR_ROOT / corpus / "corpus.jsonl",
    }


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_rank(doc_id: str, seed: int) -> int:
    key = f"{seed}|{doc_id}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16)


def allocate_group_counts(group_sizes: Dict[Tuple[str, str], int], target_total: int) -> Dict[Tuple[str, str], int]:
    total = sum(group_sizes.values())
    if total == 0:
        return {group: 0 for group in group_sizes}

    exact = {group: (size / total) * target_total for group, size in group_sizes.items()}
    counts = {group: min(size, math.floor(value)) for group, (size, value) in zip(group_sizes.keys(), zip(group_sizes.values(), exact.values()))}
    # The dict comprehension above is awkward to read; rewrite for clarity and correctness.
    counts = {}
    for group, size in group_sizes.items():
        counts[group] = min(size, math.floor(exact[group]))

    used = sum(counts.values())
    remainder_order = sorted(
        group_sizes,
        key=lambda group: (exact[group] - counts[group], group_sizes[group], group),
        reverse=True,
    )
    idx = 0
    while used < target_total and remainder_order:
        group = remainder_order[idx % len(remainder_order)]
        if counts[group] < group_sizes[group]:
            counts[group] += 1
            used += 1
        idx += 1
        if idx > len(remainder_order) * max(target_total, 1) * 2:
            break

    if used < target_total:
        fallback = sorted(group_sizes, key=lambda group: (group_sizes[group] - counts[group], group), reverse=True)
        for group in fallback:
            while used < target_total and counts[group] < group_sizes[group]:
                counts[group] += 1
                used += 1

    if used > target_total:
        trim_order = sorted(
            group_sizes,
            key=lambda group: (counts[group] - exact[group], counts[group], group),
            reverse=True,
        )
        for group in trim_order:
            while used > target_total and counts[group] > 0:
                counts[group] -= 1
                used -= 1

    return counts


def select_subset(metadata_rows: List[Dict], rate_pct: int, seed: int) -> List[Dict]:
    if rate_pct >= 100:
        return list(metadata_rows)

    target_total = max(1, round(len(metadata_rows) * (rate_pct / 100.0)))
    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in metadata_rows:
        key = (
            str(row.get("strength_bucket", "unknown")),
            str(row.get("technique", "unknown")),
        )
        grouped[key].append(row)

    for rows in grouped.values():
        rows.sort(key=lambda row: stable_rank(str(row.get("doc_id") or row.get("id")), seed))

    counts = allocate_group_counts({group: len(rows) for group, rows in grouped.items()}, target_total)
    selected_ids = set()
    for group, rows in grouped.items():
        keep = counts.get(group, 0)
        for row in rows[:keep]:
            selected_ids.add(str(row.get("doc_id") or row.get("id")))

    ordered = [row for row in metadata_rows if str(row.get("doc_id") or row.get("id")) in selected_ids]
    return ordered


def build_merged_corpus(clean_path: Path, selected_meta: List[Dict], attacks_by_id: Dict[str, Dict], out_path: Path) -> int:
    replaced_ids = {str(row["original_id"]) for row in selected_meta}
    attack_rows = [attacks_by_id[str(row.get("doc_id") or row.get("id"))] for row in selected_meta]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with clean_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("_id", "")) in replaced_ids:
                continue
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        for row in attack_rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def force_symlink(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(source, dest)


def write_summary(
    corpus: str,
    rate_pct: int,
    total_queries: int,
    selected_meta: List[Dict],
    merged_count: int,
    canonical: Dict[str, Path],
    out_dir: Path,
) -> None:
    technique_counts = Counter(str(row.get("technique")) for row in selected_meta)
    strength_counts = Counter(str(row.get("strength_bucket")) for row in selected_meta)
    family_counts = Counter(str(row.get("attack_family") or row.get("family")) for row in selected_meta)

    summary = {
        "corpus": corpus,
        "rate_pct_of_attack_pool": rate_pct,
        "canonical_attack_count": len(load_jsonl(canonical["metadata"])),
        "selected_attack_count": len(selected_meta),
        "query_count_full_workload": total_queries,
        "poison_rate_in_merged_corpus": len(selected_meta) / merged_count if merged_count else 0.0,
        "technique_counts": dict(technique_counts),
        "strength_bucket_counts": dict(strength_counts),
        "family_counts": dict(family_counts),
        "uses_full_canonical_queries": True,
        "selection_method": "deterministic_stratified_hash_sample",
        "source_metadata": str(canonical["metadata"]),
        "source_attack": str(canonical["attack"]),
        "source_queries": str(canonical["queries"]),
        "source_clean": str(canonical["clean"]),
    }
    (out_dir / f"{corpus}_rate{rate_pct:03d}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def build_rate_variant(corpus: str, rate_pct: int, seed: int, output_root: Path) -> Path:
    canonical = corpus_paths(corpus)
    metadata_rows = load_jsonl(canonical["metadata"])
    attack_rows = load_jsonl(canonical["attack"])
    queries_rows = load_jsonl(canonical["queries"])

    out_dir = output_root / f"ipi_{corpus}_main_rate{rate_pct:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    attack_name = f"{corpus}_main_rate{rate_pct:03d}_attack.jsonl"
    metadata_name = f"{corpus}_main_rate{rate_pct:03d}_attack_metadata_v2.jsonl"
    merged_name = f"{corpus}_main_rate{rate_pct:03d}_attack_merged.jsonl"
    queries_name = f"{corpus}_main_rate{rate_pct:03d}_queries_beir.jsonl"

    attack_path = out_dir / attack_name
    metadata_path = out_dir / metadata_name
    merged_path = out_dir / merged_name
    queries_path = out_dir / queries_name

    if rate_pct >= 100:
        force_symlink(canonical["attack"], attack_path)
        force_symlink(canonical["metadata"], metadata_path)
        force_symlink(canonical["queries"], queries_path)
        force_symlink(canonical["merged"], merged_path)
        write_summary(
            corpus=corpus,
            rate_pct=rate_pct,
            total_queries=len(queries_rows),
            selected_meta=metadata_rows,
            merged_count=sum(1 for _ in canonical["clean"].open("r", encoding="utf-8") if _.strip()),
            canonical=canonical,
            out_dir=out_dir,
        )
        return out_dir

    selected_meta = select_subset(metadata_rows, rate_pct=rate_pct, seed=seed)
    attacks_by_id = {str(row.get("_id") or row.get("id")): row for row in attack_rows}
    selected_attacks = [
        attacks_by_id[str(row.get("doc_id") or row.get("id"))]
        for row in selected_meta
        if str(row.get("doc_id") or row.get("id")) in attacks_by_id
    ]

    write_jsonl(metadata_path, selected_meta)
    write_jsonl(attack_path, selected_attacks)
    # Keep the full canonical query workload to make poisoning-rate comparisons meaningful.
    force_symlink(canonical["queries"], queries_path)
    merged_count = build_merged_corpus(canonical["clean"], selected_meta, attacks_by_id, merged_path)
    write_summary(
        corpus=corpus,
        rate_pct=rate_pct,
        total_queries=len(queries_rows),
        selected_meta=selected_meta,
        merged_count=merged_count,
        canonical=canonical,
        out_dir=out_dir,
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build curated poisoning-rate variants from finalized RIPE-II blackbox corpora."
    )
    parser.add_argument("--corpus", required=True, choices=["nfcorpus", "scifact", "fiqa", "nq", "msmarco", "hotpotqa"])
    parser.add_argument("--rates", default="25,50,100", help="Comma-separated percent rates of the finalized attack pool.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rates = [int(part.strip()) for part in args.rates.split(",") if part.strip()]
    for rate_pct in rates:
        out_dir = build_rate_variant(
            corpus=args.corpus,
            rate_pct=rate_pct,
            seed=args.seed,
            output_root=args.output_root,
        )
        print(f"Built {args.corpus} rate {rate_pct}% -> {out_dir}")


if __name__ == "__main__":
    main()
