# RAG Pipeline Components

This package contains modular components for building retrieval-augmented
generation pipelines used in GuardRAG/RIPE-II experiments.

## Modules

- `types.py` defines shared document/result dataclasses.
- `document_store.py` provides JSONL-backed document storage with poisoning
  metadata support.
- `chunking.py` contains document chunking utilities.
- `query_processing.py` contains query normalization and processing hooks.
- `rerankers.py` contains reranking interfaces and simple rerankers.
- `generator.py` wraps local and hosted model generation backends.
- `api_generator.py` provides API-backed generator wrappers for OpenAI and
  Anthropic-style evaluations.
- `pipeline.py` wires storage, retrieval, reranking, prompt construction, and
  generation into an end-to-end RAG pipeline.

## Design Note

The package is intentionally componentized so experiments can report which RAG
stage changed: corpus construction, retrieval, reranking, prompt formatting,
generation, or security reporting.

## Git Hygiene

Keep code in Git. Do not commit runtime caches, model outputs, generated
prompts, or large corpora.
