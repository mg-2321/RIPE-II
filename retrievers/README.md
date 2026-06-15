# Retrievers

This package contains retriever implementations used by the GuardRAG/RIPE-II
RAG pipeline.

## Modules

- `base.py` defines the common retriever interface and result type.
- `bm25.py` implements lexical BM25 retrieval with persistent index caching.
- `dense.py` implements dense retrieval over a persistent vector index.
- `vector_index.py` provides vector-index backends used by dense retrieval.
- `hybrid.py` combines lexical and dense retrieval scores.
- `splade.py` implements SPLADE-style sparse neural retrieval.
- `__init__.py` provides a small retriever registry.

## Artifact Policy

Retriever indexes can be very large. Keep generated indexes under ignored cache
paths, not in Git. Source code and lightweight configs should be tracked; FAISS,
BM25, SPLADE, SQLite, and pickle indexes should be regenerated or distributed as
separate artifacts.
