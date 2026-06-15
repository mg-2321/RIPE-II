"""
Dense retriever backed by a persistent vector index.

Author: Gayatri Malladi
"""

from __future__ import annotations

from typing import List
from pathlib import Path
import hashlib
import os
import shutil
import time

from .base import BaseRetriever, RetrieverResult
from .vector_index import PersistentVectorIndex


def _resolve_device(device: str | None, require_gpu: bool) -> str:
    try:
        import torch
    except ImportError as exc:
        if require_gpu:
            raise RuntimeError(
                "GPU retrieval was requested, but torch is not installed."
            ) from exc
        return "cpu"

    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        if require_gpu:
            raise RuntimeError("GPU retrieval was requested, but CUDA is not available.")
        return "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Retriever requested device '{device}', but CUDA is not available."
        )
    return device


class DenseRetriever(BaseRetriever):
    name = "dense"

    @staticmethod
    def _index_dir(cache_dir: Path, model_name: str, corpus_hash: str, backend: str) -> Path:
        safe_model = model_name.replace("/", "_")
        return cache_dir / f"dense_{safe_model}_{corpus_hash}_{backend}"

    @classmethod
    def _wait_for_cache(
        cls,
        index_dir: Path,
        lock_path: Path,
        poll_s: float = 5.0,
        timeout_s: float = 4 * 3600.0,
        stale_after_s: float = 30 * 60.0,
    ):
        start = time.time()
        announced = False
        while True:
            if PersistentVectorIndex.exists(index_dir):
                print(f"Loading cached vector index from {index_dir}")
                return PersistentVectorIndex.load(index_dir)
            if not lock_path.exists():
                return None
            lock_age_s = time.time() - lock_path.stat().st_mtime
            if lock_age_s > stale_after_s:
                print(
                    "Removing stale dense-index lock after "
                    f"{int(lock_age_s)}s: {lock_path}"
                )
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                if index_dir.exists() and not PersistentVectorIndex.exists(index_dir):
                    shutil.rmtree(index_dir, ignore_errors=True)
                return None
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for vector index build: {index_dir}")
            if not announced:
                print(f"Waiting for vector index to be built by another job: {index_dir}")
                announced = True
            time.sleep(poll_s)

    def __init__(
        self,
        store,
        model_name: str = "intfloat/e5-large-v2",
        cache_dir: str = None,
        device: str | None = None,
        require_gpu: bool = False,
        index_backend: str = "auto",
    ):
        super().__init__(store)
        self.model_name = model_name
        self.device = _resolve_device(device, require_gpu)
        self.index_backend = PersistentVectorIndex.resolve_backend(index_backend)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for DenseRetriever. Install via pip."
            ) from exc
        print(f"Loading dense model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self._ids = [doc.doc_id for doc in self.store]
        
        # Prefix rules:
        #   E5 models (intfloat/e5-*)       : "passage: {text}" for docs, "query: {q}" for queries
        #   BGE models (BAAI/bge-*)         : no prefix for docs, instruction prefix for queries
        #   All others                      : no prefix
        is_e5  = "e5"  in model_name.lower()
        is_bge = "bge" in model_name.lower()
        corpus_texts = [
            f"passage: {doc.title} {doc.text}" if is_e5
            else f"{doc.title} {doc.text}"           # BGE and others: no doc prefix
            for doc in self.store
        ]
        
        # Try to load cached embeddings
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            corpus_hash = hashlib.md5("".join(corpus_texts).encode()).hexdigest()[:16]
            index_dir = self._index_dir(cache_dir, model_name, corpus_hash, self.index_backend)
            lock_path = Path(str(index_dir) + ".lock")

            if PersistentVectorIndex.exists(index_dir):
                print(f"Loading dense vector index from {index_dir}...")
                self._index = PersistentVectorIndex.load(index_dir)
            else:
                while True:
                    try:
                        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    except FileExistsError:
                        waited = self._wait_for_cache(index_dir, lock_path)
                        if waited is not None:
                            self._index = waited
                            break
                        continue

                    try:
                        with os.fdopen(fd, 'w') as lock_file:
                            lock_file.write(f"pid={os.getpid()}\n")
                        print(f"Encoding corpus with {model_name}...")
                        print("  Building persistent vector index...")
                        embeddings = self.model.encode(
                            corpus_texts,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        )
                        self._index = PersistentVectorIndex.build(
                            index_dir=index_dir,
                            ids=self._ids,
                            embeddings=embeddings,
                            backend=self.index_backend,
                            metric="cosine",
                        )
                        break
                    finally:
                        try:
                            lock_path.unlink()
                        except FileNotFoundError:
                            pass
        else:
            print(f"Encoding corpus with {model_name}...")
            print("  Building in-memory vector index...")
            embeddings = self.model.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=False)
            self._index = PersistentVectorIndex.build_in_memory(
                ids=self._ids,
                embeddings=embeddings,
                backend=self.index_backend,
                metric="cosine",
            )

        self._is_e5  = is_e5
        self._is_bge = is_bge

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrieverResult]:
        # E5:  "query: {q}"
        # BGE: "Represent this sentence for searching relevant passages: {q}"
        # Others: no prefix
        if self._is_e5:
            query_text = f"query: {query}"
        elif self._is_bge:
            query_text = f"Represent this sentence for searching relevant passages: {query}"
        else:
            query_text = query
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0]
        doc_ids, scores = self._index.search(query_embedding, top_k=top_k)
        return self._collect_documents(doc_ids, scores)
