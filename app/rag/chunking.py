"""Split documents into overlapping, sentence-aware chunks.

Chunking keeps each retrieved unit small enough to embed well while preserving
enough context for the LLM to answer. We split on paragraph/sentence boundaries
first and only hard-split when a single sentence exceeds the window.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


def _stable_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_text(
    text: str,
    source: str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    base_metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    """Return overlapping chunks of at most ``chunk_size`` characters."""

    base_metadata = base_metadata or {}
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    current = ""

    def flush(buffer: str) -> None:
        buffer = buffer.strip()
        if not buffer:
            return
        idx = len(chunks)
        chunks.append(
            Chunk(
                id=_stable_id(source, idx, buffer),
                text=buffer,
                source=source,
                metadata=dict(base_metadata),
            )
        )

    for sentence in sentences:
        # Hard-split a single oversized sentence.
        while len(sentence) > chunk_size:
            head, sentence = sentence[:chunk_size], sentence[chunk_size:]
            if current:
                flush(current)
                current = ""
            flush(head)

        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            flush(current)
            # Start the next chunk with a tail overlap of the previous one for context.
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sentence}".strip()

    flush(current)
    return chunks
