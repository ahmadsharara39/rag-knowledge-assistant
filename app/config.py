"""Application configuration.

All settings are read from environment variables (or a local ``.env`` file) so the
same image can run offline for tests/CI and against real providers in production
without any code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env if present (optional dependency; safe if missing).
try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


DATA_DIR = Path(os.getenv("DATA_DIR", "storage")).resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable, process-wide configuration."""

    # Providers
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "extractive"))
    embedding_provider: str = field(
        default_factory=lambda: _get("EMBEDDING_PROVIDER", "hash")
    )
    vector_store: str = field(default_factory=lambda: _get("VECTOR_STORE", "faiss"))

    # Model names
    openai_chat_model: str = field(
        default_factory=lambda: _get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    )
    openai_embed_model: str = field(
        default_factory=lambda: _get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    )
    anthropic_model: str = field(
        default_factory=lambda: _get("ANTHROPIC_MODEL", "claude-sonnet-5")
    )
    hf_embed_model: str = field(
        default_factory=lambda: _get(
            "HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # Keys
    openai_api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY"))
    anthropic_api_key: str = field(default_factory=lambda: _get("ANTHROPIC_API_KEY"))
    pinecone_api_key: str = field(default_factory=lambda: _get("PINECONE_API_KEY"))
    pinecone_index: str = field(
        default_factory=lambda: _get("PINECONE_INDEX", "rag-knowledge")
    )

    # Retrieval / chunking
    embedding_dim: int = field(default_factory=lambda: _get_int("EMBEDDING_DIM", 384))
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 120))
    default_top_k: int = field(default_factory=lambda: _get_int("DEFAULT_TOP_K", 4))

    # Security
    api_keys: list[str] = field(
        default_factory=lambda: _get_list("API_KEYS", "dev-local-key")
    )
    rate_limit_per_minute: int = field(
        default_factory=lambda: _get_int("RATE_LIMIT_PER_MINUTE", 60)
    )

    # Storage
    index_path: str = field(
        default_factory=lambda: str(DATA_DIR / "faiss_index")
    )

    def active_providers(self) -> dict[str, str]:
        return {
            "llm": self.llm_provider,
            "embeddings": self.embedding_provider,
            "vector_store": self.vector_store,
        }


# Single shared instance.
settings = Settings()
