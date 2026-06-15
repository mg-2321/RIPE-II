# RIPE-II: Retrieval In-Place Poisoning Evaluation with Indirect Prompt Injections

RIPE-II is a research framework for building poisoned RAG corpora and evaluating how indirect prompt injections change retrieval and generation behavior in end-to-end RAG systems.

This repository contains:

- poisoned-corpus generation
- retriever analysis across multiple retrieval configurations
- end-to-end live attack evaluation with clean counterfactual baselines
- document-selection proof / ablation analysis
- compact benchmark artifacts and summary tables

It does **not** include the separate phase-2 training repo.

## Scope

RIPE-II focuses on realistic and engineered-blackbox corpus poisoning for RAG. The main question is:

1. can a poisoned document be retrieved in a mixed corpus?
2. does the model actually use it?
3. does it change the answer relative to a clean baseline?
4. how does this vary across retrievers, models, and corpora?

## Main Corpora

- **NF-Corpus**
- **SciFact**
- **FiQA**
- **HotpotQA**
- **Natural Questions**
- **MS MARCO**

## Repo Layout

```text
configs/                  model and runtime config helpers
corpus_generation/        corpus materialization and query-building scripts
evaluation/               live judge eval, retriever study, selection proof
guards/                   lightweight guard implementations
IPI_generators/           compact poisoned benchmark corpora to keep in-repo
rag_pipeline_components/  RAG pipeline implementation
results/                  compact query files, tables, and summary artifacts
retrievers/               BM25, dense, hybrid, SPLADE retrievers
scripts/                  small utility scripts
tests/                    focused regression tests
```

## Main Entry Points

### 1. Build or materialize poisoned corpora

Examples:

```bash
python corpus_generation/materialize_nfcorpus_realistic_main_candidate_v4.py
python corpus_generation/materialize_scifact_realistic_main_candidate_v1.py
python corpus_generation/materialize_fiqa_realistic_engineered_blackbox_v2.py
python corpus_generation/materialize_nq_hotpotqa_realistic_engineered_blackbox_v1.py
```

### 2. Run end-to-end live evaluation

The main evaluator is [evaluation/live_judge_eval.py](/mmfs1/home/gayat23/projects/guardrag-thesis/evaluation/live_judge_eval.py).

```bash
python evaluation/live_judge_eval.py \
  --corpus nfcorpus \
  --tier main_candidate \
  --model llama-3.1-8b \
  --judge gpt-4o-mini \
  --retriever bm25 \
  --sample 20
```

This evaluator reports:

- poison exposure
- target retrieval
- answer generation on the poisoned corpus
- clean baseline generation
- cosine-style attack signal
- LLM-judge counterfactual verdict

### 3. Run retriever-only component analysis

Use [evaluation/run_component_study.py](/mmfs1/home/gayat23/projects/guardrag-thesis/evaluation/run_component_study.py) to compare retrieval setups such as:

- `bm25`
- `dense_e5`
- `hybrid`
- `splade`
- optional reranking

Example:

```bash
python evaluation/run_component_study.py \
  --corpus nfcorpus \
  --tier main_candidate \
  --retriever bm25 \
  --top-k 10 \
  --output results/nfcorpus_component_study_bm25.jsonl
```

### 4. Run selection proof / ablation

Use [evaluation/run_selection_proof.py](/mmfs1/home/gayat23/projects/guardrag-thesis/evaluation/run_selection_proof.py) to test whether the model actually relied on the poisoned document.

```bash
python evaluation/run_selection_proof.py \
  --results-file /path/to/live_judge_results.jsonl \
  --corpus nfcorpus \
  --tier main_candidate \
  --model llama-3.1-8b \
  --judge gpt-4o-mini \
  --output /path/to/selection_proof.jsonl
```

## Retrievers

Implemented retrievers:

- `bm25`
- `dense`
- `hybrid`
- `splade`

Optional reranking is supported through the pipeline and component-study path.

## Environment Notes

Typical runs expect:

- a Python environment with the repo dependencies installed
- Hugging Face model access for local embedding / generation models
- OpenAI API access for judge models or external-guard experiments when used



