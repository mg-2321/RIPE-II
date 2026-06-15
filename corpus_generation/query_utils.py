#!/usr/bin/env python3
"""
query_utils.py — Query deduplication, normalization, and index metadata.

Author: Gayatri Malladi

Used by ipi_generator_v4_semantic_dense.py to preprocess queries before
indexing and before injecting query text into visible attack payloads.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ── Style detection ────────────────────────────────────────────────────────────

_QUESTION_PAT = re.compile(r"\b(what|who|when|where|why|how|which|does|is|are|can|did)\b", re.I)


def query_style_type(text: str) -> str:
    """
    Classify a query string as:
      'question'  — starts with a wh-word or auxiliary verb
      'title'     — short capitalized phrase (no verb)
      'keyword'   — lowercase keyword string
      'other'
    """
    t = text.strip()
    if not t:
        return "other"
    # Question-like
    if t.endswith("?") or _QUESTION_PAT.match(t):
        return "question"
    # Title-like: multiple words, most start with capital, no sentence verb
    words = t.split()
    if 2 <= len(words) <= 8:
        cap_frac = sum(1 for w in words if w and w[0].isupper()) / len(words)
        if cap_frac >= 0.6:
            return "title"
    # Keyword-like: short, lowercase
    if len(words) <= 5 and t == t.lower():
        return "keyword"
    return "other"


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize_query_for_injection(query: str) -> str:
    """
    Convert a raw query string into prose-ready text for use inside visible
    attack carrier sentences.

    Rules:
    - Strip trailing punctuation (?, .)
    - If title-like (title-case or ALL-CAPS with no verb), lowercase it
    - If question-like, strip the question mark and ensure the phrasing is
      statement-ready (e.g. "what causes X?" → "what causes X")
    - Limit to 12 tokens max to prevent awkward long injections
    """
    q = query.strip()
    q = re.sub(r"[?.!]+$", "", q).strip()

    style = query_style_type(q)
    if style == "title":
        q = q.lower()
    elif style == "question":
        # Remove leading question word for smoother prose if very short
        # e.g. "what is omega-3" → "omega-3" is too aggressive; just lowercase
        q = q.lower()

    # Cap at 12 tokens
    tokens = re.findall(r'\b[a-zA-Z0-9\-]+\b', q)
    if len(tokens) > 12:
        # Find end of 12th token in original (post-lowercased) string
        count = 0
        for m in re.finditer(r'\b[a-zA-Z0-9\-]+\b', q):
            count += 1
            if count == 12:
                q = q[:m.end()]
                break

    return q


_SAFE_TOKEN_RE = re.compile(r'\b[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9]\b|\b[a-zA-Z0-9]\b')


def safe_cap_query(query: str, max_tokens: int) -> str:
    """Cap query to max_tokens word-tokens, guaranteeing no mid-word truncation.

    Unlike a plain character-count slice (e.g. text[:60]), this function always
    ends at a word boundary. If the Nth token ends and the very next character
    in the original string is alphanumeric (i.e. we are mid-word due to unicode
    or tokenizer edge cases), it backs up to the previous space.
    """
    count = 0
    for m in _SAFE_TOKEN_RE.finditer(query):
        count += 1
        if count == max_tokens:
            capped = query[:m.end()]
            # Guard: if the next char is alphanumeric we landed mid-word
            if m.end() < len(query) and query[m.end()].isalnum():
                last_space = capped.rfind(' ')
                if last_space > 0:
                    capped = capped[:last_space]
            return capped
    return query  # fewer than max_tokens tokens — return as-is


# ── Deduplication ──────────────────────────────────────────────────────────────

def dedupe_queries(
    queries: List[Dict],
    text_field: str = "text",
    id_field: str = "_id",
    case_insensitive: bool = True,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Deduplicate a list of query dicts by text content.

    Returns:
      unique_texts   — deduplicated query texts in original order
      unique_ids     — corresponding query IDs (first seen wins)
      dup_map        — {duplicate_text: canonical_text} for tracing
    """
    seen: Dict[str, str] = {}   # normalised_text → first_original_text
    seen_ids: Dict[str, str] = {}  # normalised_text → first_id
    dup_map: Dict[str, str] = {}

    unique_texts: List[str] = []
    unique_ids:   List[str] = []

    for q in queries:
        text = q.get(text_field, "").strip()
        qid  = q.get(id_field, "")
        if not text:
            continue
        key = text.lower() if case_insensitive else text
        if key not in seen:
            seen[key]     = text
            seen_ids[key] = qid
            unique_texts.append(text)
            unique_ids.append(qid)
        else:
            dup_map[text] = seen[key]

    return unique_texts, unique_ids, dup_map


# ── Index metadata ─────────────────────────────────────────────────────────────

def build_query_index_metadata(
    unique_texts: List[str],
    unique_ids:   List[str],
    original_count: int,
) -> Dict:
    """
    Return a metadata dict describing the query index (for manifest / logging).
    """
    return {
        "original_query_count": original_count,
        "indexed_query_count":  len(unique_texts),
        "duplicates_removed":   original_count - len(unique_texts),
        "sample_queries":       unique_texts[:5],
        "style_distribution":   {
            s: sum(1 for t in unique_texts if query_style_type(t) == s)
            for s in ("question", "title", "keyword", "other")
        },
    }
