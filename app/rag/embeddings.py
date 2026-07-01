"""Embedding providers behind a single interface.

- ``OpenAIEmbeddings`` and ``HuggingFaceEmbeddings`` for production quality.
- ``HashEmbeddings`` is a deterministic, dependency-free fallback so the whole
  system runs offline (tests, CI, demos) with no API key. It is a real
  hashing-vectorizer with L2 normalisation — not random — so semantically
  overlapping text still lands near each other, which is enough to exercise and
  evaluate the full pipeline.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        ...


def _l2_normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class HashEmbeddings:
    """Offline hashing embedder (bag-of-words → fixed-dim, normalised)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN.findall(text.lower())
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 1) % 2 == 0 else -1.0
            vec[idx] += sign
        return _l2_normalise(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._embed_one(text)


class OpenAIEmbeddings:
    """OpenAI embeddings (requires ``openai`` and an API key)."""

    def __init__(self, api_key: str, model: str, dim: int) -> None:
        from openai import OpenAI  # local import keeps the dep optional

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HuggingFaceEmbeddings:
    """Local sentence-transformers embeddings (no external API calls)."""

    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def build_embedding_provider(settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.lower()
    if provider == "openai" and settings.openai_api_key:
        return OpenAIEmbeddings(
            settings.openai_api_key, settings.openai_embed_model, settings.embedding_dim
        )
    if provider == "huggingface":
        return HuggingFaceEmbeddings(settings.hf_embed_model)
    # Default / fallback: fully offline.
    return HashEmbeddings(settings.embedding_dim)
