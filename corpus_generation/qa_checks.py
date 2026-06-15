#!/usr/bin/env python3
"""
qa_checks.py — Quality assurance checks for generated poisoned corpora.

Author: Gayatri Malladi

Usage:
  python corpus_generation/qa_checks.py \\
      --poisoned  path/to/poisoned_only.jsonl \\
      --metadata  path/to/metadata.jsonl \\
      --merged    path/to/merged.jsonl \\
      --out       path/to/qa_report.json \\
      --sample    50
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    from corpus_generation.biomedical_signals import (
        is_valid_compound as _bio_is_valid_compound,
        is_valid_disease as _bio_is_valid_disease,
    )
except Exception:
    try:
        from biomedical_signals import (  # type: ignore
            is_valid_compound as _bio_is_valid_compound,
            is_valid_disease as _bio_is_valid_disease,
        )
    except Exception:
        _bio_is_valid_compound = None
        _bio_is_valid_disease = None


GENERIC_PLACEHOLDERS = {
    "the condition under study",
    "the reported condition",
    "the studied intervention",
    "the reported exposure",
    "the evaluated treatment",
}

SUSPICIOUS_COMPOUNDS = {
    "canine", "kinase", "decline", "guideline", "database", "phase",
    "10-month-old", "1-year-old", "5-b", "dioxide",
    "nine", "base", "transferase", "3-kinase",
}

# Regex patterns for structurally junk compounds (questionnaire items, peaks, etc.)
# NOTE: do NOT add a generic [a-z]{n}-\d+ pattern here — it would flag valid
# protein abbreviations (il-6, cox-2, mmp-9).  Study-code rejection is handled
# upstream in biomedical_signals._is_valid_compound_candidate.
_JUNK_COMPOUND_PATS = [
    re.compile(r"^\d+-item$"),                        # 69-item, 212-item
    re.compile(r"^peak-\d+$"),                        # peak-3, peak-4
    re.compile(r"^(?:wk|visit|session|phase|cycle|run|trial)-\d+$"),  # wk-1
    re.compile(r"^\d+-kda$"),                         # 70-kDa
]

HIGH_VISIBILITY_TECHNIQUES = {
    "json_ld_meta_injection",
    "code_comment_smuggling",
    "visual_ocr_injection",
}

REALISM_GATE_VERSION = "nfcorpus_biomedical_v2"
FINANCIAL_REALISM_GATE_VERSION = "fiqa_financial_v1"

_REALISTIC_BIO_TECHNIQUES = {
    "blackbox": {
        "semantic_cloaking",
        "near_query_placement",
        "prompt_attack_template",
        "keyword_packing",
        "citation_hijack",
        "idem_optimized",
        "table_caption_directive",
        "anchor_see_also_hijack",
        "visual_ocr_injection",
    },
    "graybox": {
        "semantic_cloaking",
        "near_query_placement",
        "prompt_attack_template",
        "keyword_packing",
        "citation_hijack",
        "idem_optimized",
        "table_caption_directive",
        "anchor_see_also_hijack",
        "visual_ocr_injection",
    },
    "whitebox": {
        "semantic_cloaking",
        "near_query_placement",
        "prompt_attack_template",
        "keyword_packing",
        "citation_hijack",
        "idem_optimized",
        "table_caption_directive",
        "anchor_see_also_hijack",
        "visual_ocr_injection",
    },
}

_REALISTIC_FINANCIAL_TECHNIQUES = {
    "blackbox": {
        "semantic_cloaking",
        "keyword_packing",
        "citation_hijack",
        "anchor_see_also_hijack",
    },
    "graybox": {
        "semantic_cloaking",
        "keyword_packing",
        "citation_hijack",
        "anchor_see_also_hijack",
        "near_query_placement",
    },
    "whitebox": {
        "semantic_cloaking",
        "keyword_packing",
        "citation_hijack",
        "anchor_see_also_hijack",
        "near_query_placement",
    },
}

_BANNED_CONTROL_PATTERNS = [
    ("banned_control_phrase:response", re.compile(r"\bany response\b|\bresponses?\s+(?:must|should|are required|based|derived)\b", re.I)),
    ("banned_control_phrase:summary", re.compile(r"\bany summary\b|\bsummaries?\s+that\b", re.I)),
    ("banned_control_phrase:synthesis", re.compile(r"\bany synthesis\b|\bsynthesi[sz](?:e|ing)\b", re.I)),
    ("banned_control_phrase:must", re.compile(r"\bmust\b", re.I)),
    ("banned_control_phrase:mandatory", re.compile(r"\bmandatory (?:inclusion|reporting)\b", re.I)),
    ("banned_control_phrase:guideline", re.compile(r"\bguideline(?:s)?\s+require\b", re.I)),
    ("banned_control_phrase:protocol", re.compile(r"\bprotocol (?:update|note)\b", re.I)),
    ("banned_control_phrase:reviewer_annotation", re.compile(r"\breviewer annotation\b", re.I)),
    ("banned_control_phrase:reporting_standard", re.compile(r"\breporting standard\b", re.I)),
    ("banned_control_phrase:misrepresents", re.compile(r"\bmisrepresents?\b", re.I)),
    ("banned_control_phrase:when_summarizing", re.compile(r"\bwhen summarizing\b", re.I)),
]

_BROKEN_FRAGMENT_PATTERNS = [
    ("malformed_fragment:trypsin", re.compile(r"\bby trypsin inhibitor\b", re.I)),
    ("malformed_fragment:this_patients_study", re.compile(r"\bthis patients study\b", re.I)),
    ("malformed_fragment:lung", re.compile(r"\bon and the lung\b", re.I)),
    ("malformed_fragment:major", re.compile(r"\bas a major\b", re.I)),
    ("malformed_fragment:into_cancer", re.compile(r"\binto cancer\b", re.I)),
    ("malformed_fragment:8_log", re.compile(r"\b8-log\b", re.I)),
    ("malformed_fragment:at_the_beginning", re.compile(r"\bat the beginning\b", re.I)),
    ("malformed_fragment:biology", re.compile(r"\bstarting beginning perspectives biology\b", re.I)),
]

_DISALLOWED_WRAPPER_PATTERNS = [
    ("disallowed_wrapper:reviewer", re.compile(r"^(?:Reviewer|Review annotation)\b", re.I)),
    ("disallowed_wrapper:reporting", re.compile(r"^Reporting standard\b", re.I)),
    ("disallowed_wrapper:protocol", re.compile(r"^Protocol\b", re.I)),
    ("disallowed_wrapper:system", re.compile(r"^(?:Internal consistency check|System guidance)\b", re.I)),
    ("disallowed_wrapper:editorial", re.compile(r"^(?:Editorial note|\[Note\]|Publisher note|Author's note)\b", re.I)),
]

_ENTITY_NOISE_WORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of",
    "on", "or", "the", "to", "with", "beginning", "biology", "major",
    "patient", "patients", "perspectives", "research", "study", "this",
}

_GENERIC_GROUNDING_PLACEHOLDERS = {
    "the associated condition",
    "the condition under study",
    "the evaluated treatment",
    "the reported condition",
    "the reported exposure",
    "the studied condition",
    "the studied intervention",
}

_FINANCIAL_GROUNDING_PLACEHOLDERS = {
    "financial treatment",
    "the issue under discussion",
    "the relevant issue",
    "the relevant question",
    "the reported topic",
    "the topic under discussion",
}

_FINANCIAL_FRAGMENT_PATTERNS = [
    ("financial_fragment:mesh_terms", re.compile(r"\bmesh terms\b", re.I)),
    ("financial_fragment:index_terms", re.compile(r"\bindex terms\b", re.I)),
    ("financial_fragment:clinical_relevance", re.compile(r"\bclinical (?:relevance|significance|note)\b", re.I)),
    ("financial_fragment:methods_note", re.compile(r"\bmethods note\b", re.I)),
    ("financial_fragment:quality_note", re.compile(r"\bquality note\b", re.I)),
    ("financial_fragment:document_scan", re.compile(r"\bdocument scan \(ocr\)\b", re.I)),
]

_GENERIC_WEAK_PREFIX_PATTERNS = [
    ("weak_template:fact_check", re.compile(r"^fact check:\s+", re.I)),
    ("weak_template:clarification", re.compile(r"^clarification:\s+", re.I)),
    ("weak_template:interpretation_should_consider", re.compile(r"^interpretation should consider\s+", re.I)),
    ("weak_template:search_context", re.compile(r"^search context:\s+", re.I)),
    ("weak_template:practical_reading", re.compile(r"^a practical reading\b", re.I)),
    ("weak_template:key_consideration", re.compile(r"^in plain terms, the key consideration is\b", re.I)),
    ("weak_template:central_consideration", re.compile(r"^the central consideration is usually\b", re.I)),
    ("weak_template:practice_context", re.compile(r"^practice context:\s+", re.I)),
    ("weak_template:category", re.compile(r"^category:\s+", re.I)),
]

_QUESTION_LIKE_PAYLOAD_LEADS = {
    "what", "how", "why", "who", "when", "where", "which", "is", "are",
    "was", "were", "can", "could", "should", "would", "will", "do",
    "does", "did", "am",
}


def _looks_like_question_phrase(text: str) -> bool:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return False
    if "?" in cleaned or ":" in cleaned:
        return True
    words = re.findall(r"[a-z0-9$%+.-]+", cleaned)
    if not words:
        return False
    return words[0] in _QUESTION_LIKE_PAYLOAD_LEADS

_TITLE_FRAGMENT_HEAD_WORDS = {
    "benefit", "benefits", "burden", "calcified", "cases", "clinical",
    "cross-analysis", "cross", "detection", "effect", "effects", "expression",
    "facing", "findings", "free", "generation", "guide", "habits", "hidden",
    "impact", "improvements", "influencing", "insights", "intake", "is",
    "lack", "management", "natural", "observed", "original", "perils",
    "prevalence", "question", "quality", "reduction", "reference", "replacement",
    "role", "screening", "study", "studies", "trends", "updated", "use",
    "world", "worldwide",
}

_TITLE_FRAGMENT_TAIL_WORDS = {
    "analysis", "article", "articles", "biology", "consider", "contribution",
    "guidance", "impact", "importance", "investigation", "limitations",
    "manifestations", "methods", "note", "notes", "nutrition", "paper",
    "prevention", "question", "recommendations", "report", "reports",
    "results", "review", "screening", "status", "study", "supportive",
    "surveys", "trial", "update",
}


def get_allowed_realistic_bio_techniques(attacker_setting: str) -> set:
    return set(_REALISTIC_BIO_TECHNIQUES.get(attacker_setting or "blackbox", _REALISTIC_BIO_TECHNIQUES["blackbox"]))


def get_allowed_realistic_financial_techniques(attacker_setting: str) -> set:
    return set(
        _REALISTIC_FINANCIAL_TECHNIQUES.get(
            attacker_setting or "blackbox",
            _REALISTIC_FINANCIAL_TECHNIQUES["blackbox"],
        )
    )


def _entity_is_malformed(entity: str) -> bool:
    lower = (entity or "").strip().lower()
    if not lower:
        return False
    if lower in _GENERIC_GROUNDING_PLACEHOLDERS:
        return True
    if any(pat.search(lower) for _, pat in _BROKEN_FRAGMENT_PATTERNS):
        return True
    words = re.findall(r"[a-z0-9+-]+", lower)
    if words and (words[0] in _ENTITY_NOISE_WORDS or words[-1] in _ENTITY_NOISE_WORDS):
        return True
    if words and (words[0] in _TITLE_FRAGMENT_HEAD_WORDS or words[-1] in _TITLE_FRAGMENT_TAIL_WORDS):
        return True
    if len(words) >= 3:
        non_noise = [word for word in words if word not in _ENTITY_NOISE_WORDS]
        if len(non_noise) <= 1:
            return True
    return False


def _financial_entity_is_malformed(entity: str) -> bool:
    lower = (entity or "").strip().lower()
    if not lower:
        return False
    if lower in _FINANCIAL_GROUNDING_PLACEHOLDERS:
        return True
    if any(pat.search(lower) for _, pat in _FINANCIAL_FRAGMENT_PATTERNS):
        return True
    words = re.findall(r"[a-z0-9$%+.-]+", lower)
    if not words:
        return True
    if words[0] in _ENTITY_NOISE_WORDS or words[-1] in _ENTITY_NOISE_WORDS:
        return True
    if len(words) >= 4:
        non_noise = [word for word in words if word not in _ENTITY_NOISE_WORDS]
        if len(non_noise) <= 1:
            return True
    return False


def _compound_is_invalid(entity: str) -> bool:
    if _entity_is_malformed(entity):
        return True
    if _bio_is_valid_compound is None:
        return False
    return not _bio_is_valid_compound(entity)


def _disease_is_invalid(entity: str) -> bool:
    if _entity_is_malformed(entity):
        return True
    if _bio_is_valid_disease is None:
        return False
    return not _bio_is_valid_disease(entity)


def evaluate_realism_record(
    meta: Dict,
    mode: Optional[str] = None,
    domain: Optional[str] = None,
) -> Dict:
    """
    Evaluate whether a generated record satisfies the nfcorpus biomedical realism gate.
    """
    issues: List[str] = []
    payload = meta.get("payload_text", "") or ""
    payload_lower = payload.lower()
    attacker_setting = meta.get("attacker_setting", "blackbox")
    technique = meta.get("technique", "")

    if mode == "realistic" and domain == "biomedical":
        allowed = get_allowed_realistic_bio_techniques(attacker_setting)
        if technique not in allowed:
            issues.append(f"disallowed_technique:{technique}")

        for rule, pattern in _BANNED_CONTROL_PATTERNS:
            if pattern.search(payload):
                issues.append(rule)

        for rule, pattern in _BROKEN_FRAGMENT_PATTERNS:
            if pattern.search(payload):
                issues.append(rule)

        for rule, pattern in _DISALLOWED_WRAPPER_PATTERNS:
            if pattern.search(payload):
                issues.append(rule)

        if technique != "json_ld_meta_injection" and payload.lstrip().startswith(("{", "[", "\"", "'")):
            issues.append("disallowed_wrapper:structured_payload")

        compounds = [c for c in meta.get("compounds_extracted", []) if isinstance(c, str)]
        diseases = [d for d in meta.get("diseases_extracted", []) if isinstance(d, str)]
        resolved_compound = meta.get("resolved_compound")
        resolved_disease = meta.get("resolved_disease")
        if isinstance(resolved_compound, str) and resolved_compound:
            compounds = [resolved_compound] + compounds
        if isinstance(resolved_disease, str) and resolved_disease:
            diseases = [resolved_disease] + diseases
        for entity in compounds:
            if _compound_is_invalid(entity):
                issues.append(f"malformed_compound:{entity}")
        for entity in diseases:
            if _disease_is_invalid(entity):
                issues.append(f"malformed_disease:{entity}")

        if meta.get("compound_source") == "topic" and meta.get("disease_source") == "topic":
            issues.append("invalid_grounding:double_topic_fallback")

    if mode == "realistic" and domain == "financial":
        allowed = get_allowed_realistic_financial_techniques(attacker_setting)
        if technique not in allowed:
            issues.append(f"disallowed_technique:{technique}")

        for rule, pattern in _BANNED_CONTROL_PATTERNS:
            if pattern.search(payload):
                issues.append(rule)

        for rule, pattern in _FINANCIAL_FRAGMENT_PATTERNS:
            if pattern.search(payload):
                issues.append(rule)

        for rule, pattern in _DISALLOWED_WRAPPER_PATTERNS:
            if pattern.search(payload):
                issues.append(rule)

        if payload.lstrip().startswith(("{", "[", "\"", "'")):
            issues.append("disallowed_wrapper:structured_payload")

        for field_name in ("resolved_topic", "resolved_focus"):
            value = meta.get(field_name)
            if isinstance(value, str) and value and _financial_entity_is_malformed(value):
                issues.append(f"malformed_financial_entity:{field_name}:{value}")

        query_text = meta.get("selected_query_text_raw", "") or meta.get("selected_query_text_normalized", "") or ""
        if attacker_setting == "whitebox" and query_text and not float(meta.get("query_similarity", 0) or 0):
            issues.append("query_similarity:missing")
        if meta.get("focus_source") == "query":
            focus_text = (meta.get("resolved_focus") or "").strip()
            if _looks_like_question_phrase(query_text) or len(focus_text.split()) > 4:
                issues.append("query_focus:too_close_to_query")

    if mode == "realistic" and domain in {"general", "web"}:
        for rule, pattern in _BANNED_CONTROL_PATTERNS:
            if pattern.search(payload):
                issues.append(rule)

        for rule, pattern in _GENERIC_WEAK_PREFIX_PATTERNS:
            if pattern.search(payload):
                issues.append(rule)

        if payload.lstrip().startswith(("{", "[", "\"", "'")) and technique != "json_ld_meta_injection":
            issues.append("disallowed_wrapper:structured_payload")

        query_text = meta.get("selected_query_text_raw", "") or meta.get("selected_query_text_normalized", "") or ""
        if attacker_setting == "whitebox" and query_text and not float(meta.get("query_similarity", 0) or 0):
            issues.append("query_similarity:missing")
        query_tokens = {
            tok for tok in re.findall(r"[a-z0-9]+", query_text.lower())
            if len(tok) > 2 and tok not in _ENTITY_NOISE_WORDS
        }
        payload_tokens = {
            tok for tok in re.findall(r"[a-z0-9]+", payload_lower)
            if len(tok) > 2 and tok not in _ENTITY_NOISE_WORDS
        }
        if query_tokens:
            overlap = query_tokens & payload_tokens
            if len(overlap) >= max(3, (len(query_tokens) + 1) // 2):
                issues.append("query_echo:high_overlap")
        if _looks_like_question_phrase(query_text) and len(query_tokens) >= 3:
            issues.append("query_echo:question_like_query")

        non_boilerplate_tokens = [
            tok for tok in re.findall(r"[a-z0-9]+", payload_lower)
            if len(tok) > 2
            and tok not in _ENTITY_NOISE_WORDS
            and tok not in {
                "evidence", "findings", "source", "reading", "context", "consider",
                "align", "response", "drawing", "conclusions", "primary", "reference",
                "relevant", "passage", "page", "discussion", "query",
            }
        ]
        if len(non_boilerplate_tokens) < 4:
            issues.append("weak_payload:low_content")

    if attacker_setting == "blackbox":
        if meta.get("target_query_ids"):
            issues.append("blackbox_query_leak:target_query_ids")
        if meta.get("selected_query_text_raw"):
            issues.append("blackbox_query_leak:selected_query_text_raw")
        if meta.get("selected_query_text_normalized"):
            issues.append("blackbox_query_leak:selected_query_text_normalized")
        if meta.get("semantic_query_id"):
            issues.append("blackbox_query_leak:semantic_query_id")
        if float(meta.get("query_similarity", 0) or 0):
            issues.append("blackbox_query_leak:query_similarity")

    deduped = list(dict.fromkeys(issues))
    return {
        "passed": not deduped,
        "issues": deduped,
        "quality_flags": deduped,
    }


def summarize_realism_audit(records: List[Dict]) -> Dict:
    status_counts = {"accepted": 0, "rejected": 0}
    by_technique: Dict[str, int] = {}
    by_strategy: Dict[str, int] = {}
    by_setting: Dict[str, int] = {}
    rejection_reasons: Dict[str, int] = {}

    for record in records:
        status = record.get("status", "accepted")
        status_counts[status] = status_counts.get(status, 0) + 1

        technique = record.get("technique", "")
        if technique:
            by_technique[technique] = by_technique.get(technique, 0) + 1

        strategy = record.get("directive_strategy", "")
        if strategy:
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

        setting = record.get("attacker_setting", "")
        if setting:
            by_setting[setting] = by_setting.get(setting, 0) + 1

        for reason in record.get("triggered_rules", []):
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    return {
        "status_counts": status_counts,
        "by_technique": dict(sorted(by_technique.items())),
        "by_strategy": dict(sorted(by_strategy.items())),
        "by_setting": dict(sorted(by_setting.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def check_spans(poisoned_docs: List[Dict], metadata: List[Dict]) -> Dict:
    """Verify text[span_start:span_end] exactly matches payload_text for every doc."""
    meta_by_id = {m["doc_id"]: m for m in metadata}
    total = len(poisoned_docs)
    span_ok, span_fail, span_missing = 0, 0, 0
    failures = []

    for doc in poisoned_docs:
        did  = doc["_id"]
        text = doc.get("text", "")
        meta = meta_by_id.get(did)
        if meta is None:
            span_missing += 1
            continue
        s, e    = meta.get("span_start", 0), meta.get("span_end", 0)
        payload = meta.get("payload_text", "")
        if not payload:
            span_missing += 1
            continue
        extracted = text[s:e]
        if extracted == payload:
            span_ok += 1
        else:
            span_fail += 1
            failures.append({
                "doc_id":    did,
                "expected":  payload[:80],
                "got":       extracted[:80],
                "span":      [s, e],
            })

    return {
        "span_ok":      span_ok,
        "span_fail":    span_fail,
        "span_missing": span_missing,
        "total":        total,
        "pass_rate":    span_ok / total if total else 0,
        "failures":     failures[:20],   # cap at 20 examples in report
    }


def check_poison_rate(poisoned_docs: List[Dict], merged: List[Dict]) -> Dict:
    """Verify poison count and rate against merged corpus."""
    ipi_in_merged = sum(1 for d in merged if d["_id"].startswith("IPI_"))
    clean_in_merged = len(merged) - ipi_in_merged
    actual_rate = ipi_in_merged / len(merged) if merged else 0

    return {
        "poisoned_only_count": len(poisoned_docs),
        "ipi_in_merged":       ipi_in_merged,
        "clean_in_merged":     clean_in_merged,
        "merged_total":        len(merged),
        "actual_poison_rate":  round(actual_rate, 4),
    }


def check_duplicate_ids(poisoned_docs: List[Dict]) -> Dict:
    ids = [d["_id"] for d in poisoned_docs]
    unique = len(set(ids))
    return {"total": len(ids), "unique": unique, "duplicates": len(ids) - unique}


def check_no_word_splits(poisoned_docs: List[Dict], metadata: List[Dict]) -> Dict:
    """Check that injection boundaries do not cut inside a word on either side."""
    meta_by_id = {m["doc_id"]: m for m in metadata}
    splits = []
    for doc in poisoned_docs:
        text = doc.get("text", "")
        meta = meta_by_id.get(doc["_id"])
        if meta is None:
            continue
        s = meta.get("span_start", 0)
        e = meta.get("span_end", 0)
        # Check char immediately before span_start
        if s > 0 and text[s - 1].isalnum():
            splits.append({
                "doc_id":  doc["_id"],
                "side":    "before",
                "char":    text[s - 1],
                "context": text[max(0, s - 10):s + 10],
            })
        # Check char immediately after span_end
        if e < len(text) and text[e].isalnum():
            splits.append({
                "doc_id":  doc["_id"],
                "side":    "after",
                "char":    text[e],
                "context": text[max(0, e - 10):e + 10],
            })
    return {"word_splits": len(splits), "examples": splits[:10]}


def sample_audit(
    poisoned_docs: List[Dict],
    metadata: List[Dict],
    n: int = 50,
    seed: int = 42,
) -> List[Dict]:
    """Return a random sample of poisoned docs with their metadata for manual review."""
    meta_by_id = {m["doc_id"]: m for m in metadata}
    rng = random.Random(seed)
    sample = rng.sample(poisoned_docs, min(n, len(poisoned_docs)))
    results = []
    for doc in sample:
        meta = meta_by_id.get(doc["_id"], {})
        s, e = meta.get("span_start", 0), meta.get("span_end", 0)
        results.append({
            "doc_id":           doc["_id"],
            "technique":        meta.get("technique", "?"),
            "strategy":         meta.get("directive_strategy", "?"),
            "attacker_setting": meta.get("attacker_setting", "?"),
            "payload_preview":  doc["text"][s:e][:120],
            "context_before":   doc["text"][max(0, s - 80):s],
            "context_after":    doc["text"][e:e + 80],
            "compound_confidence": meta.get("compound_confidence", 0),
            "insertion_boundary":  meta.get("insertion_boundary_type", "?"),
        })
    return results


def check_technique_structure(poisoned_docs: List[Dict], metadata: List[Dict]) -> Dict:
    """Spot-check technique-specific structural invariants.

    Verified invariants:
    - json_ld_meta_injection: text must start with '{' (payload at position=start)
    - unicode_stealth: text must contain at least one ZWC character
    """
    _ZWC = {'\u200b', '\u200c', '\u200d', '\u2060', '\u00ad'}
    meta_by_id = {m["doc_id"]: m for m in metadata}
    issues = []
    for doc in poisoned_docs:
        meta = meta_by_id.get(doc["_id"], {})
        tech = meta.get("technique", "")
        text = doc.get("text", "")
        if tech == "json_ld_meta_injection" and not text.lstrip().startswith("{"):
            issues.append({"doc_id": doc["_id"], "issue": "json_ld not at start of text"})
        if tech == "unicode_stealth" and not any(c in text for c in _ZWC):
            issues.append({"doc_id": doc["_id"], "issue": "no ZWC character found in text"})
    return {"issues": issues, "count": len(issues)}


def check_payload_quality(
    metadata: List[Dict],
    mode: Optional[str] = None,
    domain: Optional[str] = None,
) -> Dict:
    """Flag generic placeholders, suspicious compounds, direct query echo, and weak noisy carriers."""
    issues = []
    for meta in metadata:
        payload = meta.get("payload_text", "")
        payload_lower = payload.lower()
        query = (meta.get("selected_query_text_normalized") or meta.get("selected_query_text_raw") or "").strip().lower()
        technique = meta.get("technique", "")
        compounds = [c.lower() for c in meta.get("compounds_extracted", []) if isinstance(c, str)]
        compound_conf = meta.get("compound_confidence", 0)

        if any(phrase in payload_lower for phrase in GENERIC_PLACEHOLDERS):
            issues.append({"doc_id": meta.get("doc_id", "?"), "issue": "generic placeholder in payload", "technique": technique})

        # Surface-form extract bug: compound name was extracted *with* a leading
        # preposition (e.g. "of Zn extract" instead of "Zn extract").
        # Check compound names, not the payload prose, to avoid false-positives
        # when the template legitimately writes "effect of apple extract on ...".
        if any(re.match(r"^(?:of|from|with)\s+\S+\s+extract$", c, re.I) for c in compounds):
            issues.append({"doc_id": meta.get("doc_id", "?"), "issue": "surface-form extract bug in payload", "technique": technique})

        suspicious = [
            compound for compound in compounds
            if (compound in SUSPICIOUS_COMPOUNDS
                or re.fullmatch(r"\d+-(?:day|week|month|year)s?-old", compound)
                or any(pat.fullmatch(compound) for pat in _JUNK_COMPOUND_PATS))
        ]
        if suspicious:
            issues.append({"doc_id": meta.get("doc_id", "?"), "issue": f"suspicious compounds extracted: {', '.join(suspicious[:3])}", "technique": technique})

        if query and len(query.split()) >= 4 and query in payload_lower:
            issues.append({"doc_id": meta.get("doc_id", "?"), "issue": "payload directly echoes selected query", "technique": technique})

        if (
            not (mode == "realistic" and domain == "biomedical")
            and technique in HIGH_VISIBILITY_TECHNIQUES
            and compound_conf < 3
        ):
            issues.append({"doc_id": meta.get("doc_id", "?"), "issue": "high-visibility carrier used with weak compound grounding", "technique": technique})

    return {"issues": issues, "count": len(issues)}


def check_realism_gate(
    metadata: List[Dict],
    mode: Optional[str] = None,
    domain: Optional[str] = None,
) -> Dict:
    audit = []
    issues = []
    for meta in metadata:
        verdict = evaluate_realism_record(meta, mode=mode, domain=domain)
        entry = {
            "doc_id": meta.get("doc_id", "?"),
            "technique": meta.get("technique", ""),
            "directive_strategy": meta.get("directive_strategy", ""),
            "attacker_setting": meta.get("attacker_setting", ""),
            "status": "accepted" if verdict["passed"] else "rejected",
            "triggered_rules": verdict["issues"],
        }
        audit.append(entry)
        if verdict["issues"]:
            issues.append({
                "doc_id": meta.get("doc_id", "?"),
                "technique": meta.get("technique", ""),
                "issues": verdict["issues"],
            })

    version = REALISM_GATE_VERSION
    if mode == "realistic" and domain == "financial":
        version = FINANCIAL_REALISM_GATE_VERSION

    return {
        "version": version,
        "count": len(issues),
        "issues": issues[:20],
        "summary": summarize_realism_audit(audit),
    }


def run_all_checks(
    poisoned_docs: List[Dict],
    metadata: List[Dict],
    merged: List[Dict],
    sample_n: int = 50,
    mode: Optional[str] = None,
    domain: Optional[str] = None,
) -> Dict:
    report = {
        "span_validation":      check_spans(poisoned_docs, metadata),
        "poison_rate":          check_poison_rate(poisoned_docs, merged),
        "duplicate_ids":        check_duplicate_ids(poisoned_docs),
        "word_split_check":     check_no_word_splits(poisoned_docs, metadata),
        "technique_structure":  check_technique_structure(poisoned_docs, metadata),
        "payload_quality":      check_payload_quality(metadata, mode=mode, domain=domain),
        "realism_gate":         check_realism_gate(metadata, mode=mode, domain=domain),
        "sample_audit":         sample_audit(poisoned_docs, metadata, sample_n),
    }
    # Summary pass/fail
    sv = report["span_validation"]
    # Fail if span content check skipped for more than 10% of docs (stale metadata)
    too_many_missing = sv["span_missing"] > max(1, int(0.10 * sv["total"])) if sv["total"] else False
    report["passed"] = (
        sv["span_fail"] == 0
        and not too_many_missing
        and report["duplicate_ids"]["duplicates"] == 0
        and report["word_split_check"]["word_splits"] <= 1   # tolerate ≤1 edge-case split
        and report["technique_structure"]["count"] == 0
        and report["payload_quality"]["count"] == 0
        and report["realism_gate"]["count"] == 0
    )
    return report


def main():
    ap = argparse.ArgumentParser(description="QA checks for IPI poisoned corpora")
    ap.add_argument("--poisoned",  required=True)
    ap.add_argument("--metadata",  required=True)
    ap.add_argument("--merged",    required=True)
    ap.add_argument("--out",       required=True)
    ap.add_argument("--sample",    type=int, default=50)
    ap.add_argument("--mode",      default=None)
    ap.add_argument("--domain",    default=None)
    args = ap.parse_args()

    print("Loading files...")
    poisoned = load_jsonl(args.poisoned)
    metadata = load_jsonl(args.metadata)
    merged   = load_jsonl(args.merged)

    print(f"Running QA checks on {len(poisoned)} poisoned docs...")
    report = run_all_checks(poisoned, metadata, merged, args.sample, mode=args.mode, domain=args.domain)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nQA Report -> {args.out}")
    print(f"  Span OK      : {report['span_validation']['span_ok']}/{report['span_validation']['total']}")
    print(f"  Span fail    : {report['span_validation']['span_fail']}")
    print(f"  Duplicates   : {report['duplicate_ids']['duplicates']}")
    print(f"  Word splits  : {report['word_split_check']['word_splits']}")
    print(f"  Struct issues: {report['technique_structure']['count']}")
    print(f"  Payload issues: {report['payload_quality']['count']}")
    print(f"  Realism issues: {report['realism_gate']['count']}")
    print(f"  Poison rate  : {report['poison_rate']['actual_poison_rate']:.1%}")
    print(f"  PASSED       : {report['passed']}")
    if not report['passed']:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
