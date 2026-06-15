"""
BM25 retriever implementation.

Author: Gayatri Malladi
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi as _ExternalBM25Okapi
except Exception:  # pragma: no cover
    _ExternalBM25Okapi = None

from .base import BaseRetriever, RetrieverResult


class _FallbackBM25Okapi:
    """Minimal BM25 implementation used when rank_bm25 is unavailable/broken."""

    def __init__(self, corpus_tokens, *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_freqs = []
        self.doc_len = []
        df = Counter()

        total_len = 0
        for tokens in corpus_tokens:
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            doc_len = len(tokens)
            self.doc_len.append(doc_len)
            total_len += doc_len
            for token in freqs:
                df[token] += 1

        self.avgdl = (total_len / self.corpus_size) if self.corpus_size else 0.0
        self.idf = {
            token: max(0.0, math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5)))
            for token, freq in df.items()
        }

    def get_scores(self, query_tokens):
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if not self.corpus_size:
            return scores
        avgdl = self.avgdl or 1.0
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for idx, freqs in enumerate(self.doc_freqs):
                freq = freqs.get(token, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_len[idx] / avgdl))
                scores[idx] += idf * ((freq * (self.k1 + 1.0)) / denom)
        return scores


BM25Okapi = _ExternalBM25Okapi or _FallbackBM25Okapi


class BM25Retriever(BaseRetriever):
    name = "bm25"

    @staticmethod
    def _cache_root() -> Path | None:
        root = os.environ.get("GUARDRAG_BM25_INDEX_DIR") or os.environ.get("GUARDRAG_DOCINDEX_DIR")
        if not root:
            return None
        path = Path(root) / "bm25"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _store_signature(store) -> str:
        path = getattr(store, "_path", None)
        if path is not None:
            p = Path(path)
            stat = p.stat()
            payload = f"{p.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"
            return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]

        ids = []
        for doc in store:
            ids.append(doc.doc_id)
            if len(ids) >= 1000:
                break
        payload = f"memory::{len(store)}::" + "::".join(ids)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]

    @classmethod
    def _cache_paths(cls, store, tokenizer_name: str) -> tuple[Path, Path] | None:
        root = cls._cache_root()
        if root is None:
            return None
        signature = cls._store_signature(store)
        safe_tokenizer = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in tokenizer_name)[:80]
        cache_path = root / f"bm25_{signature}_{safe_tokenizer}.pkl"
        lock_path = root / f"bm25_{signature}_{safe_tokenizer}.lock"
        return cache_path, lock_path

    @staticmethod
    def _wait_for_cache(cache_path: Path, lock_path: Path, *, poll_s: float = 10.0, timeout_s: float = 8 * 3600.0):
        start = time.time()
        announced = False
        while lock_path.exists():
            if cache_path.exists():
                return None
            age_s = time.time() - lock_path.stat().st_mtime
            if age_s > timeout_s:
                print(f"Removing stale BM25 lock after {int(age_s)}s: {lock_path}", flush=True)
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                return None
            if not announced:
                print(f"Waiting for BM25 index cache from another job: {cache_path}", flush=True)
                announced = True
            time.sleep(poll_s)
        return None

    @staticmethod
    def _load_cache(cache_path: Path):
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        return payload["index"], payload["ids"]

    @staticmethod
    def _save_cache(cache_path: Path, index, ids: list[str]) -> None:
        tmp = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
        with tmp.open("wb") as f:
            pickle.dump({"index": index, "ids": ids}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)

    def __init__(self, store, *, tokenizer=None):
        super().__init__(store)
        self._tokenizer = tokenizer or (lambda text: text.lower().split())
        tokenizer_name = getattr(self._tokenizer, "__name__", self._tokenizer.__class__.__name__)

        cache_paths = self._cache_paths(self.store, tokenizer_name)
        if cache_paths is not None:
            cache_path, lock_path = cache_paths
            if cache_path.exists():
                print(f"Loading BM25 index cache from {cache_path}", flush=True)
                self._index, self._ids = self._load_cache(cache_path)
                return
            while True:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    self._wait_for_cache(cache_path, lock_path)
                    if cache_path.exists():
                        print(f"Loading BM25 index cache from {cache_path}", flush=True)
                        self._index, self._ids = self._load_cache(cache_path)
                        return
                    continue
                else:
                    with os.fdopen(fd, "w") as lock_file:
                        lock_file.write(f"pid={os.getpid()}\n")
                    break
        else:
            cache_path = None
            lock_path = None

        # Tokenize corpus with progress for large corpora
        try:
            store_size = len(self.store)
            if store_size > 1000:
                print(f"Building BM25 index for {store_size} documents...", flush=True)
            corpus_tokens = []
            ids = []
            for i, doc in enumerate(self.store):
                corpus_tokens.append(self._tokenizer(f"{doc.title} {doc.text}"))
                ids.append(doc.doc_id)
                if store_size > 1000 and (i + 1) % 1000 == 0:
                    print(f"  Tokenized {i + 1}/{store_size} documents...", flush=True)
            if store_size > 1000:
                print("Creating BM25 index...", flush=True)
            self._index = BM25Okapi(corpus_tokens)
            self._ids = ids
            if cache_path is not None:
                print(f"Saving BM25 index cache to {cache_path}", flush=True)
                self._save_cache(cache_path, self._index, self._ids)
            if store_size > 1000:
                print("BM25 index built!", flush=True)
        finally:
            if lock_path is not None:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrieverResult]:
        query_tokens = self._tokenizer(query)
        scores = self._index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        doc_ids = [self._ids[i] for i in top_indices]
        doc_scores = [scores[i] for i in top_indices]
        return self._collect_documents(doc_ids, doc_scores)
