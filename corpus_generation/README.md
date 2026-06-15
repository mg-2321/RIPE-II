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
