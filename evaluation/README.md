# Evaluation

This folder contains evaluation and analysis code for RIPE-II/GuardRAG-style
RAG security experiments. It should contain source code only; generated JSONL
outputs, plots, logs, and large benchmark artifacts should stay in `results/`,
`logs/`, or an external artifact release.

## Core Evaluation Scripts

- `live_judge_eval.py` runs live RAG attack evaluation with an LLM-as-judge and
  cosine-similarity ASR comparison.
- `run_component_study.py` records how poisoned documents propagate through
  retrieval, reranking, and generation stages.
- `run_selection_proof.py` performs counterfactual document-removal and
  clean-document substitution checks for causal evidence of poison selection.
- `summarize_live_judge_metrics.py` aggregates live-judge JSONL outputs into
  benchmark-level metrics.
- `summarize_retriever_stage_results.py` builds retriever-stage tables and
  figures for the RIPE-II analysis.

## Shared Helpers

- `judge_utils.py` centralizes judge prompt construction, verdict parsing, and
  judge-model helper logic.

## Figure/Plot Scripts

- `plot_blackbox_retriever_revised.py` renders blackbox retriever-risk plots.
- `plot_capability_vulnerability_inversion.py` renders the
  capability-vulnerability inversion figure.
- `plot_slack_ipi_diagram.py` recreates the Slack-style RAG/IPI diagram with
  corrected labels.
- `plot_tier_progression_fresh.py` renders poisoning-rate progression plots.

## Git Hygiene

Do not commit generated evaluation outputs. Keep source scripts in Git and keep
large result directories under ignored paths such as `results/` and `logs/`.
