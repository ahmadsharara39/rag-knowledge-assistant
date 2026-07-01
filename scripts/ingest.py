"""Ingest local documents (.txt / .md) into the vector store and persist it.

Usage:
    python -m scripts.ingest --path data/sample_docs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import settings
from app.rag.pipeline import RAGPipeline
from app.schemas import Document

SUPPORTED = {".txt", ".md"}


def load_documents(path: Path) -> list[Document]:
    docs: list[Document] = []
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for file in files:
        if file.suffix.lower() not in SUPPORTED:
            continue
        text = file.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(Document(id=file.stem, text=text, source=file.name))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG store.")
    parser.add_argument("--path", default="data/sample_docs", help="File or directory.")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Path not found: {path}")

    docs = load_documents(path)
    if not docs:
        raise SystemExit(f"No .txt/.md documents found under {path}")

    pipeline = RAGPipeline(settings)
    pipeline.load()  # append to any existing index
    num_docs, num_chunks = pipeline.ingest_documents(docs)
    pipeline.persist()

    print(f"Ingested {num_docs} document(s) -> {num_chunks} chunk(s).")
    print(f"Vector store now holds {pipeline.size} vector(s).")
    print(f"Providers: {settings.active_providers()}")


if __name__ == "__main__":
    main()
