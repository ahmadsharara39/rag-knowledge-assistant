from app.config import Settings
from app.rag.pipeline import RAGPipeline
from app.schemas import Document


def _pipeline():
    # Force offline providers regardless of the developer's environment.
    settings = Settings(
        llm_provider="extractive",
        embedding_provider="hash",
        vector_store="faiss",
    )
    return RAGPipeline(settings)


def test_ingest_and_retrieve_relevant_chunk():
    pipe = _pipeline()
    pipe.ingest_documents(
        [
            Document(text="FAISS and Pinecone are supported vector stores.", source="vs.md"),
            Document(text="Bananas are a yellow fruit rich in potassium.", source="fruit.md"),
        ]
    )
    hits = pipe.retrieve("Which vector databases are supported?", top_k=1)
    assert hits, "expected at least one hit"
    assert hits[0].chunk.source == "vs.md"


def test_answer_includes_citations():
    pipe = _pipeline()
    pipe.ingest_documents(
        [Document(text="The service is rate limited per API key.", source="sec.md")]
    )
    answer, citations = pipe.answer("How is the service rate limited?", top_k=2)
    assert citations
    assert citations[0].marker == "[1]"
    assert "[" in answer  # extractive LLM tags sentences with source markers


def test_empty_store_returns_graceful_message():
    pipe = _pipeline()
    answer, citations = pipe.answer("Anything?", top_k=3)
    assert citations == []
    assert "ingest" in answer.lower() or "can't" in answer.lower()
