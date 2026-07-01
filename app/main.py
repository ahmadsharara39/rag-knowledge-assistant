"""FastAPI application exposing the RAG pipeline.

Routes:
  GET  /health  — liveness + active providers (public)
  POST /ingest  — add documents (auth + rate limit)
  POST /query   — ask a grounded question (auth + rate limit)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request

from . import __version__
from .auth import enforce_rate_limit, require_api_key
from .config import settings
from .rag.pipeline import RAGPipeline
from .schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

# One pipeline per process; load any persisted index at startup.
pipeline = RAGPipeline(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        pipeline.load()
    except Exception:
        # A missing/incompatible index is fine — start empty.
        pass
    yield


app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    version=__version__,
    description=(
        "Retrieval-augmented Q&A over your documents with grounded, cited answers. "
        "Providers (LLM, embeddings, vector store) are pluggable via environment "
        "variables."
    ),
    lifespan=lifespan,
)


def _auth_and_limit(request: Request, api_key: str = Depends(require_api_key)) -> str:
    enforce_rate_limit(request, api_key)
    return api_key


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        providers=settings.active_providers(),
        vectors_in_store=pipeline.size,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    body: IngestRequest, _: str = Depends(_auth_and_limit)
) -> IngestResponse:
    num_docs, num_chunks = pipeline.ingest_documents(body.documents)
    try:
        pipeline.persist()
    except Exception:
        pass
    return IngestResponse(
        ingested_documents=num_docs,
        total_chunks=num_chunks,
        vectors_in_store=pipeline.size,
    )


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, _: str = Depends(_auth_and_limit)) -> QueryResponse:
    answer, citations = pipeline.answer(body.question, body.top_k)
    return QueryResponse(
        answer=answer,
        citations=citations,
        provider=settings.active_providers(),
    )
