"""
Document store abstraction for GuardRAG.

Author: Gayatri Malladi

This module follows the component layout suggested by RAG'n'Roll:
documents live in a unified store, regardless of the downstream
retriever.  Stores can be backed by JSONL files, in-memory dictionaries,
or vector databases in future iterations.  For now we support a simple
JSONL-backed store with optional poisoning metadata.
"""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .chunking import Chunker
from .types import Document


class DocumentStore:
    """
    Lightweight document store that can be wrapped by different retrieval
    components.  Designed to mirror the 'DocumentStore' block from the
    RAG'n'Roll blueprint.
    """

    def __init__(
        self,
        documents: Optional[Iterable[Document]] = None,
        *,
        path: Optional[Path] = None,
        index_db_path: Optional[Path] = None,
    ):
        self._documents: Optional[Dict[str, Document]] = None
        self._path: Optional[Path] = None
        self._index_db_path: Optional[Path] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._cache: Dict[str, Document] = {}
        self._count: int = 0

        if documents is not None:
            self._documents = {}
            for doc in documents:
                if doc.doc_id in self._documents:
                    raise ValueError(f"Duplicate document id detected: {doc.doc_id}")
                self._documents[doc.doc_id] = doc
            self._count = len(self._documents)
            return

        if path is None or index_db_path is None:
            raise ValueError("DocumentStore requires either documents or a lazy path/index pair.")
        self._path = Path(path)
        self._index_db_path = Path(index_db_path)
        self._conn = sqlite3.connect(str(self._index_db_path))
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'count'").fetchone()
        self._count = int(row[0]) if row else 0

    @staticmethod
    def _doc_from_data(data: Dict) -> Document:
        metadata = data.get("metadata", {}).copy()
        for key, value in data.items():
            if key not in {"_id", "title", "text", "metadata"}:
                metadata[key] = value
        return Document(
            doc_id=data["_id"],
            title=data.get("title", ""),
            text=data.get("text", ""),
            metadata=metadata,
        )

    @staticmethod
    def _index_db_for(path: Path) -> Path:
        index_root = os.environ.get("GUARDRAG_DOCINDEX_DIR")
        if index_root:
            root = Path(index_root)
            root.mkdir(parents=True, exist_ok=True)
            resolved = str(path.resolve())
            digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
            safe_name = f"{path.stem}.{digest}.docindex.sqlite"
            return root / safe_name
        return path.with_suffix(path.suffix + ".docindex.sqlite")

    @classmethod
    def _ensure_sqlite_index(cls, path: Path) -> Path:
        index_db = cls._index_db_for(path)
        stat = path.stat()
        source_size = str(stat.st_size)
        source_mtime = str(stat.st_mtime_ns)
        rebuild = True

        if index_db.exists():
            try:
                conn = sqlite3.connect(str(index_db))
                size_row = conn.execute("SELECT value FROM meta WHERE key = 'source_size'").fetchone()
                mtime_row = conn.execute("SELECT value FROM meta WHERE key = 'source_mtime_ns'").fetchone()
                conn.close()
                rebuild = not (
                    size_row and mtime_row
                    and size_row[0] == source_size
                    and mtime_row[0] == source_mtime
                )
            except sqlite3.DatabaseError:
                rebuild = True

        if not rebuild:
            return index_db

        index_db.parent.mkdir(parents=True, exist_ok=True)
        tmp_db = index_db.with_suffix(index_db.suffix + f".{os.getpid()}.tmp")
        if tmp_db.exists():
            tmp_db.unlink()
        conn = sqlite3.connect(str(tmp_db))
        conn.execute("CREATE TABLE docs (doc_id TEXT PRIMARY KEY, offset INTEGER NOT NULL)")
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        count = 0
        bad_lines = 0
        batch = []
        with path.open("r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                doc_id = str(data.get("_id", ""))
                if not doc_id:
                    continue
                batch.append((doc_id, offset))
                count += 1
                if len(batch) >= 5000:
                    conn.executemany("INSERT OR REPLACE INTO docs(doc_id, offset) VALUES (?, ?)", batch)
                    batch.clear()
        if batch:
            conn.executemany("INSERT OR REPLACE INTO docs(doc_id, offset) VALUES (?, ?)", batch)
        conn.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            [
                ("source_size", source_size),
                ("source_mtime_ns", source_mtime),
                ("count", str(count)),
                ("bad_lines", str(bad_lines)),
            ],
        )
        conn.commit()
        conn.close()
        os.replace(tmp_db, index_db)
        return index_db

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        chunker: Optional[Chunker] = None,
        lazy: bool = False,
    ) -> "DocumentStore":
        path = Path(path)
        if lazy:
            if chunker is not None:
                raise ValueError("Lazy DocumentStore does not support chunking.")
            index_db = cls._ensure_sqlite_index(path)
            return cls(path=path, index_db_path=index_db)

        documents: List[Document] = []

        bad_lines = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                doc = cls._doc_from_data(data)
                if chunker:
                    documents.extend(chunker.chunk(doc))
                else:
                    documents.append(doc)

        if bad_lines:
            print(f"Warning: skipped {bad_lines} invalid JSON lines in {path}", file=sys.stderr)

        return cls(documents)

    def __len__(self) -> int:
        if self._documents is not None:
            return len(self._documents)
        return self._count

    def __iter__(self) -> Iterator[Document]:
        if self._documents is not None:
            for doc in self._documents.values():
                yield doc
            return
        assert self._path is not None
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield self._doc_from_data(data)

    def get(self, doc_id: str) -> Optional[Document]:
        if self._documents is not None:
            return self._documents.get(doc_id)
        if doc_id in self._cache:
            return self._cache[doc_id]
        assert self._conn is not None and self._path is not None
        row = self._conn.execute("SELECT offset FROM docs WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        with self._path.open("r", encoding="utf-8") as f:
            f.seek(int(row[0]))
            line = f.readline()
        if not line:
            return None
        data = json.loads(line)
        doc = self._doc_from_data(data)
        self._cache[doc_id] = doc
        return doc

    def filter(self, *, poisoned: Optional[bool] = None) -> List[Document]:
        """
        Filter documents by poisoning status.
        """
        if poisoned is None:
            return list(self)
        return [doc for doc in self if doc.is_poisoned == poisoned]

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for doc in self:
                f.write(json.dumps(doc.to_dict()) + "\n")

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
