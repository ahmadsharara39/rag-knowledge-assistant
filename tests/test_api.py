from fastapi.testclient import TestClient

from app.auth import reset_rate_limits
from app.main import app

client = TestClient(app)
HEADERS = {"x-api-key": "dev-local-key"}


def setup_function():
    reset_rate_limits()


def test_ui_is_served_at_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "RAG Knowledge Assistant" in resp.text


def test_health_is_public():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "llm" in body["providers"]


def test_query_requires_api_key():
    resp = client.post("/query", json={"question": "hello there", "top_k": 3})
    assert resp.status_code == 401


def test_ingest_then_query_flow():
    ingest = client.post(
        "/ingest",
        headers=HEADERS,
        json={
            "documents": [
                {
                    "text": "The assistant supports FAISS and Pinecone vector stores.",
                    "source": "vs.md",
                }
            ]
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["total_chunks"] >= 1

    query = client.post(
        "/query",
        headers=HEADERS,
        json={"question": "Which vector stores are supported?", "top_k": 3},
    )
    assert query.status_code == 200
    data = query.json()
    assert data["citations"], "expected citations"
    assert data["citations"][0]["source"] == "vs.md"


def test_query_validation_rejects_short_question():
    resp = client.post("/query", headers=HEADERS, json={"question": "hi"})
    assert resp.status_code == 422  # min_length violation
