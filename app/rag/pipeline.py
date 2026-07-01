"""The RAG pipeline: ingest → embed → store, and query → retrieve → generate.

This class wires the three provider interfaces together and is the single object
the API layer talks to. It is deliberately provider-agnostic.
"""

from __future__ import annotations

from ..config import Settings
from ..schemas import Citation
from .chunking import chunk_text
from .embeddings import build_embedding_provider
from .llm import build_llm_provider
from .vector_store import SearchHit, build_vector_store


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embeddings = build_embedding_provider(settings)
        self.store = build_vector_store(settings)
        self.llm = build_llm_provider(settings)
        # Keep the store dim aligned with whatever the embedder actually produces.
        self._sync_dim()

    def _sync_dim(self) -> None:
        # The active embedder is the source of truth for dimensionality; make the
        # (still empty) store match it so the two never disagree.
        dim = getattr(self.embeddings, "dim", self.settings.embedding_dim)
        if hasattr(self.store, "dim") and self.store.dim != dim:
            self.store.dim = dim

    # ---- ingestion -------------------------------------------------------
    def ingest_documents(self, documents) -> tuple[int, int]:
        """Return (num_documents, num_chunks) added."""
        all_chunks = []
        for doc in documents:
            source = getattr(doc, "source", None) or "inline"
            meta = dict(getattr(doc, "metadata", {}) or {})
            if getattr(doc, "id", None):
                meta.setdefault("doc_id", doc.id)
            chunks = chunk_text(
                doc.text,
                source=source,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
                base_metadata=meta,
            )
            all_chunks.extend(chunks)

        if all_chunks:
            vectors = self.embeddings.embed([c.text for c in all_chunks])
            self.store.add(all_chunks, vectors)
        return len(documents), len(all_chunks)

    def persist(self) -> None:
        self.store.persist(self.settings.index_path)

    def load(self) -> bool:
        return self.store.load(self.settings.index_path)

    @property
    def size(self) -> int:
        return self.store.size

    # ---- query -----------------------------------------------------------
    def retrieve(self, question: str, top_k: int) -> list[SearchHit]:
        q_vec = self.embeddings.embed_one(question)
        return self.store.search(q_vec, top_k)

    def answer(self, question: str, top_k: int) -> tuple[str, list[Citation]]:
        hits = self.retrieve(question, top_k)
        if not hits:
            return (
                "No documents have been ingested yet, so I can't answer that.",
                [],
            )

        context_blocks: list[str] = []
        citations: list[Citation] = []
        for i, hit in enumerate(hits, start=1):
            marker = f"[{i}]"
            context_blocks.append(f"{marker} {hit.chunk.text}")
            snippet = hit.chunk.text
            citations.append(
                Citation(
                    marker=marker,
                    source=hit.chunk.source,
                    chunk_id=hit.chunk.id,
                    score=round(hit.score, 4),
                    snippet=snippet[:280] + ("…" if len(snippet) > 280 else ""),
                )
            )

        answer_text = self.llm.generate(question, context_blocks)
        return answer_text, citations
