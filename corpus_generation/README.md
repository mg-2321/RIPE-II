# Corpus Generation

This folder contains the source code used to construct and audit RIPE-II-style
indirect prompt injection corpora. It should contain code only; generated
datasets, audits, checkpoints, logs, and backups should remain outside Git or
be released as separate artifacts.

## Main Corpus Builders

- `materialize_nq_main.py` builds the Natural Questions main benchmark from
  qrel-positive documents with visible and obfuscated attack techniques.
- `materialize_hotpotqa_main.py` builds the HotpotQA main benchmark using the
  same qrel-positive design as NQ.
- `materialize_msmarco_main.py` builds the MS MARCO main benchmark from BEIR
  qrel-positive documents.
- `materialize_fiqa_main_candidate.py` builds the canonical FIQA main candidate
  from query-aligned documents and vetted legacy attack pools.
- `materialize_nfcorpus_main.py` builds the NFCorpus main benchmark with fixed
  attack counts and mixed attack strengths.
- `materialize_scifact_main.py` builds the SciFact main benchmark, including
  visual-asset variants used for finalized benchmark analysis.

## Attack Generation and Variants

- `ipi_generator_v4_semantic_dense.py` is the generic semantic dense IPI
  generator for non-main corpora.
- `build_blackbox_rate_sweep.py` creates blackbox poisoning-rate variants while
  preserving the canonical query set.
- `build_security_strong_corpus.py` creates security-strong variants focused on
  instruction hierarchy override, task redirection, citation hijacking, report
  manipulation, obfuscation, and exfiltration-style objectives.
- `build_visual_ocr_blackbox_corpus.py` creates high-quality visual blackbox
  corpora with real rendered image assets, manifests, and audit files. It
  supports two separated modes: `ocr`, where OCR-extracted image text is
  inserted into the retrieved document, and `preview`, where the attack remains
  image-only for browser/link-preview or multimodal-agent evaluation.

## Quality Control and Shared Utilities

- `audit_security_strong_corpus.py` audits security-strong corpus metadata and
  attack-family coverage.
- `qa_checks.py` validates generated poisoned, metadata, and merged corpus
  files.
- `query_utils.py` provides query normalization, deduplication, and query-style
  helpers.
- `selection_strategies.py` implements blackbox, graybox, and whitebox document
  selection strategies.
- `biomedical_signals.py` provides biomedical signal extraction used by
  biomedical/NFCorpus-style attack generation.

## Git Hygiene

Do not commit generated corpora from `IPI_generators/`, `data/`, `results/`,
`_backups/`, or `_candidate_*_audit/`. Keep final datasets in the research
artifact directory and publish them separately from the source-code repository
when appropriate.

## Rebuilding Visual Blackbox Corpora

The DSN-count visual artifact set can be regenerated with:

```bash
python corpus_generation/build_visual_ocr_blackbox_corpus.py \
  --output-root /gscratch/uwb/gayat23/GuardRAG/IPI_generators \
  --profile dsn \
  --mode both \
  --selection-mode quality \
  --variant-suffix dsn_quality_v1 \
  --corpora nq hotpotqa msmarco fiqa nfcorpus scifact
```

This produces two separated artifact families per dataset:

- `*_visual_ocr_blackbox_merged.jsonl`
- `*_visual_ocr_blackbox_metadata.jsonl`
- `*_visual_ocr_blackbox_manifest.json`
- `*_visual_ocr_blackbox_audit.json`
- `assets/*.png`
- `*_image_preview_blackbox_episodes.jsonl`
- `*_image_preview_blackbox_metadata.jsonl`
- `*_image_preview_blackbox_manifest.json`
- `*_image_preview_blackbox_audit.json`
- `previews/*.html`

The builder uses seeded random document selection with no qrels or retriever
access, preserving the blackbox attacker setting. The default
`--selection-mode quality` still remains blackbox, but filters for readable,
content-rich, non-duplicate source documents and records source-quality metadata
in manifests and metadata rows. In `preview` mode, the attack payload is not
inserted into the text layer; it is only rendered inside the image asset and
stored in metadata for evaluation labels.
