"""
Retriever package registry.

Author: Gayatri Malladi
"""

from __future__ import annotations

from importlib import import_module

from .base import BaseRetriever  # noqa: F401


_RETRIEVER_SPECS = {
    "dense": ("retrievers.dense", "DenseRetriever"),
    "hybrid": ("retrievers.hybrid", "HybridRetriever"),
    "splade": ("retrievers.splade", "SPLADERetriever"),
    "bm25": ("retrievers.bm25", "BM25Retriever"),
}


def list_retrievers():
    return list(_RETRIEVER_SPECS.keys())


def get_retriever(name: str):
    try:
        module_name, class_name = _RETRIEVER_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown retriever {name}. Available: {list_retrievers()}"
        ) from exc

    module = import_module(module_name)
    return getattr(module, class_name)
