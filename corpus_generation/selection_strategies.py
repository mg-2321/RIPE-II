#!/usr/bin/env python3
"""
selection_strategies.py — Doc selection for black-box / gray-box / white-box attacker settings.

Author: Gayatri Malladi

Black-box  : random doc selection, no retriever knowledge.
Gray-box   : attacker knows target queries; uses surrogate SBERT retrieval to select
             likely-retrieved docs (simulates retriever internals without exact knowledge).
White-box  : attacker has exact qrels; selects highest-impact docs for target queries.
"""
from __future__ import annotations

import random
import sys
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


def should_show_progress() -> bool:
    """Only show tqdm-style progress bars in interactive terminals."""
    return sys.stderr.isatty()


def select_blackbox(
    corpus: List[Dict],
    n: int,
    rng: random.Random,
) -> Tuple[List[int], List[str], List[int]]:
    """
    Black-box selection: random sampling without replacement.
    Returns (selected_indices, target_query_ids, query_ranks).
    No query targeting — all query fields are empty.
    """
    indices = rng.sample(range(len(corpus)), min(n, len(corpus)))
    return indices, [""] * len(indices), [-1] * len(indices)


def select_graybox(
    corpus: List[Dict],
    query_texts: List[str],   # deduplicated query texts (matches query_embs rows)
    query_ids: List[str],     # deduplicated query IDs (parallel to query_texts)
    n: int,
    sbert_model,              # SentenceTransformer instance
    query_embs,               # np.ndarray shape (len(query_texts), dim)
    embed_doc_prefix: str,
    rng: random.Random,
    top_k_per_query: int = 20,
) -> Tuple[List[int], List[str], List[int]]:
    """
    Gray-box selection: for each target query, retrieve top-k docs by SBERT similarity.
    Select from the union of likely-retrieved docs (weighted by retrieval rank).

    query_texts / query_ids must be the DEDUPLICATED lists that were used to build
    query_embs — their lengths must match query_embs.shape[0].

    Returns (selected_indices, target_query_ids, query_ranks).
    """
    if not HAS_NUMPY:
        raise RuntimeError("numpy required for gray-box selection")

    num_queries = len(query_texts)
    if query_embs.shape[0] != num_queries:
        raise ValueError(
            f"select_graybox: query_embs has {query_embs.shape[0]} rows "
            f"but query_texts has {num_queries} entries — must match."
        )

    # Encode all docs once
    doc_texts = [
        embed_doc_prefix + f"{d.get('title', '')}. {d.get('text', '')[:400]}"
        for d in corpus
    ]
    doc_embs = sbert_model.encode(doc_texts, batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=should_show_progress())
    doc_embs = np.asarray(doc_embs, dtype=np.float32)

    # For each query, find top_k docs
    # doc_idx -> (best_rank, query_id)
    doc_best: Dict[int, Tuple[int, str]] = {}
    for qi in range(num_queries):
        sims = query_embs[qi] @ doc_embs.T
        top_k = min(top_k_per_query, len(corpus))
        top_indices = np.argpartition(-sims, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]
        for rank, doc_idx in enumerate(top_indices.tolist()):
            qid = query_ids[qi] if qi < len(query_ids) else str(qi)
            if doc_idx not in doc_best or rank < doc_best[doc_idx][0]:
                doc_best[int(doc_idx)] = (rank, qid)

    # Sort candidates by rank (lower is better)
    candidates = sorted(doc_best.keys(), key=lambda idx: doc_best[idx][0])
    pool = candidates[:max(n * 3, 200)]

    if HAS_NUMPY and len(pool) > n:
        # Weighted sample without replacement: weight inversely proportional to rank+1
        weights = np.array([1.0 / (doc_best[idx][0] + 1) for idx in pool], dtype=np.float64)
        weights /= weights.sum()
        chosen_positions = np.random.choice(len(pool), size=min(n, len(pool)), replace=False, p=weights)
        selected = [pool[int(i)] for i in chosen_positions]
    else:
        # Fallback: take two-thirds from top third of pool, rest from remainder
        top = pool[:max(1, len(pool) // 3)]
        rest = pool[len(top):]
        n_top  = min((n * 2) // 3, len(top))
        n_rest = min(n - n_top, len(rest))
        selected = rng.sample(top, n_top) + (rng.sample(rest, n_rest) if n_rest > 0 else [])

    qids  = [doc_best[idx][1] for idx in selected]
    ranks = [doc_best[idx][0] for idx in selected]
    return selected, qids, ranks


def select_whitebox(
    corpus: List[Dict],
    qrels: Dict[str, Dict[str, int]],   # {query_id: {doc_id: relevance}}
    queries: List[Dict],
    n: int,
) -> Tuple[List[int], List[str], List[int]]:
    """
    White-box selection: use qrels to deterministically choose the highest-impact docs.
    Returns (selected_indices, target_query_ids, query_ranks).
    """
    doc_id_to_idx = {d["_id"]: i for i, d in enumerate(corpus)}

    # Aggregate total relevance score per doc across all queries.
    # Docs that are relevant to many queries produce the highest-impact poisoning.
    doc_relevance: Dict[int, Tuple[int, int, str]] = {}  # idx -> (total_rel_score, best_rel_score, best_qid)
    for qid, doc_rels in qrels.items():
        for did, rel in doc_rels.items():
            if rel < 1:
                continue
            idx = doc_id_to_idx.get(did)
            if idx is None:
                continue
            if idx not in doc_relevance:
                doc_relevance[idx] = (0, rel, qid)
            prev_total, prev_best_rel, prev_best_qid = doc_relevance[idx]
            best_rel = prev_best_rel
            best_qid = prev_best_qid
            if rel > prev_best_rel or (rel == prev_best_rel and qid < prev_best_qid):
                best_rel = rel
                best_qid = qid
            doc_relevance[idx] = (prev_total + rel, best_rel, best_qid)

    # Deterministically rank by total relevance descending.
    # Break ties by strongest single-query relevance and then corpus index so
    # repeated runs yield the same white-box set.
    candidates = sorted(
        doc_relevance.keys(),
        key=lambda i: (-doc_relevance[i][0], -doc_relevance[i][1], i),
    )
    selected = candidates[:min(n, len(candidates))]

    qids  = [doc_relevance[idx][2] for idx in selected]
    ranks = [0] * len(selected)
    return selected, qids, ranks
