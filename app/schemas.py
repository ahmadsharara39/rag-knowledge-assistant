"""Request/response models. Validation happens here so bad input never reaches
the RAG pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str | None = Field(
        default=None, description="Optional stable id; generated if omitted."
    )
    text: str = Field(..., min_length=1, description="Raw document text.")
    source: str = Field(default="inline", description="Human-readable source label.")
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[Document] = Field(..., min_length=1, max_length=500)


class IngestResponse(BaseModel):
    ingested_documents: int
    total_chunks: int
    vectors_in_store: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)


class Citation(BaseModel):
    marker: str = Field(..., description="Inline marker used in the answer, e.g. [1].")
    source: str
    chunk_id: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    provider: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    version: str
    providers: dict[str, str]
    vectors_in_store: int
