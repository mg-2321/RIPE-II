"""
Base retriever interfaces used by GuardRAG.

Author: Gayatri Malladi

Mirrors the RAG'n'Roll retriever contract: retrieve(query, top_k) ->
List[(doc, score)] with scores normalized to higher=better.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

from rag_pipeline_components.document_store import DocumentStore
from rag_pipeline_components.types import Document


RetrieverResult = Tuple[Document, float]


class BaseRetriever(ABC):
    """Abstract retriever definition."""

    name: str = "base"

    def __init__(self, store: DocumentStore):
        self.store = store

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrieverResult]:
        """Return a ranked list of documents for the query."""

    def _collect_documents(self, indices: Sequence[str], scores) -> List[RetrieverResult]:
        results: List[RetrieverResult] = []
        for idx, score in zip(indices, scores):
            doc = self.store.get(idx)
            if doc is not None:
                results.append((doc, float(score)))
        return results
