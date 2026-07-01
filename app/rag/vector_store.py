"""Vector stores behind a common interface.

- ``FaissStore`` uses FAISS when installed and transparently falls back to a
  NumPy brute-force cosine index otherwise, so semantic search always works.
- ``PineconeStore`` targets a managed Pinecone index for horizontal scale.

Both expose the same ``add`` / ``search`` / ``persist`` / ``load`` surface, so the
pipeline is agnostic to which one is active.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class FaissStore:
    """FAISS inner-product index over L2-normalised vectors (= cosine similarity).

    Falls back to a NumPy brute-force search if faiss is not installed.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._ids: set[str] = set()
        self._faiss = None
        self._index = None
        self._matrix: np.ndarray | None = None  # used only in numpy fallback
        try:
            import faiss  # type: ignore

            self._faiss = faiss
            self._index = faiss.IndexFlatIP(dim)
        except Exception:
            self._faiss = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        # Skip chunks already indexed (ids are content hashes), so re-ingesting the
        # same document is idempotent rather than duplicating vectors.
        new_chunks, new_vectors = [], []
        for chunk, vec in zip(chunks, vectors):
            if chunk.id in self._ids:
                continue
            self._ids.add(chunk.id)
            new_chunks.append(chunk)
            new_vectors.append(vec)
        if not new_chunks:
            return

        arr = np.asarray(new_vectors, dtype="float32")
        if arr.shape[1] != self.dim:
            raise ValueError(
                f"vector dim {arr.shape[1]} != index dim {self.dim}"
            )
        self._chunks.extend(new_chunks)
        if self._faiss is not None:
            self._index.add(arr)
        else:
            self._matrix = (
                arr if self._matrix is None else np.vstack([self._matrix, arr])
            )

    def search(self, query_vector: list[float], top_k: int) -> list[SearchHit]:
        if self.size == 0:
            return []
        q = np.asarray([query_vector], dtype="float32")
        k = min(top_k, self.size)
        if self._faiss is not None:
            scores, idxs = self._index.search(q, k)
            pairs = zip(idxs[0].tolist(), scores[0].tolist())
        else:
            sims = (self._matrix @ q[0]).tolist()
            order = np.argsort(sims)[::-1][:k]
            pairs = ((int(i), float(sims[i])) for i in order)
        return [
            SearchHit(chunk=self._chunks[i], score=float(s))
            for i, s in pairs
            if i >= 0
        ]

    def persist(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path, "chunks.json"), "w", encoding="utf-8") as fh:
            json.dump([asdict(c) for c in self._chunks], fh, ensure_ascii=False)
        if self._faiss is not None:
            self._faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        elif self._matrix is not None:
            np.save(os.path.join(path, "matrix.npy"), self._matrix)

    def load(self, path: str) -> bool:
        chunks_file = os.path.join(path, "chunks.json")
        if not os.path.exists(chunks_file):
            return False
        with open(chunks_file, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self._chunks = [Chunk(**c) for c in raw]
        self._ids = {c.id for c in self._chunks}
        if self._faiss is not None and os.path.exists(
            os.path.join(path, "index.faiss")
        ):
            self._index = self._faiss.read_index(os.path.join(path, "index.faiss"))
        elif os.path.exists(os.path.join(path, "matrix.npy")):
            self._matrix = np.load(os.path.join(path, "matrix.npy"))
        return True


class PineconeStore:
    """Managed Pinecone index. Chunk text/metadata is stored in the vector metadata
    so search results can be reconstructed without a separate document store."""

    def __init__(self, api_key: str, index_name: str, dim: int) -> None:
        from pinecone import Pinecone, ServerlessSpec

        self.dim = dim
        self._pc = Pinecone(api_key=api_key)
        existing = {i["name"] for i in self._pc.list_indexes()}
        if index_name not in existing:
            self._pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = self._pc.Index(index_name)

    @property
    def size(self) -> int:
        try:
            return int(self._index.describe_index_stats()["total_vector_count"])
        except Exception:
            return 0

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        payload = [
            {
                "id": chunk.id,
                "values": vec,
                "metadata": {
                    "text": chunk.text,
                    "source": chunk.source,
                    **chunk.metadata,
                },
            }
            for chunk, vec in zip(chunks, vectors)
        ]
        for start in range(0, len(payload), 100):
            self._index.upsert(vectors=payload[start : start + 100])

    def search(self, query_vector: list[float], top_k: int) -> list[SearchHit]:
        res = self._index.query(
            vector=query_vector, top_k=top_k, include_metadata=True
        )
        hits: list[SearchHit] = []
        for match in res.get("matches", []):
            meta = match.get("metadata", {}) or {}
            text = meta.pop("text", "")
            source = meta.pop("source", "unknown")
            hits.append(
                SearchHit(
                    chunk=Chunk(
                        id=match["id"], text=text, source=source, metadata=meta
                    ),
                    score=float(match.get("score", 0.0)),
                )
            )
        return hits

    def persist(self, path: str) -> None:  # managed service — nothing local to persist
        return None

    def load(self, path: str) -> bool:
        return True


def build_vector_store(settings):
    if settings.vector_store.lower() == "pinecone" and settings.pinecone_api_key:
        return PineconeStore(
            settings.pinecone_api_key, settings.pinecone_index, settings.embedding_dim
        )
    return FaissStore(settings.embedding_dim)
