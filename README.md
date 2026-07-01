# Enterprise RAG Knowledge Assistant

A production-style **Retrieval-Augmented Generation (RAG)** service that answers
questions over your own documents and returns **grounded answers with citations**.

Built to demonstrate the full deployment path an enterprise AI-integration role
cares about: document ingestion → chunking → **embeddings** → **vector search**
(FAISS / Pinecone) → **LLM** answer synthesis (OpenAI / Anthropic) → a **secured
FastAPI** service with API-key auth, request validation, and rate limiting →
**evaluation** (retrieval precision@k and answer faithfulness).

> Runs **fully offline** out of the box using deterministic fallback providers
> (no API key required), and switches to real OpenAI/Anthropic + Pinecone when
> you set environment variables. This makes it easy to demo, test in CI, and
> then flip to production.

---

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │                 FastAPI app                  │
  HTTP  ───────► │  /health  /ingest  /query   (API-key auth,   │
                 │             rate limiting, validation)       │
                 └───────────────┬──────────────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │        RAG pipeline       │
                    │  1. chunk documents       │
                    │  2. embed chunks/query    │◄── EmbeddingProvider
                    │  3. semantic search top-k │◄── VectorStore (FAISS/Pinecone)
                    │  4. build grounded prompt │
                    │  5. LLM answer + citations │◄── LLMProvider (OpenAI/Anthropic)
                    └───────────────────────────┘
```

Every external dependency is behind an interface with a **local fallback**, so
the same code path runs offline for tests/CI and in production with real APIs.

| Concern      | Interface            | Providers                                            |
|--------------|----------------------|------------------------------------------------------|
| Embeddings   | `EmbeddingProvider`  | OpenAI · Hugging Face (sentence-transformers) · Hash (offline) |
| Vector store | `VectorStore`        | FAISS (local) · Pinecone (managed)                   |
| LLM          | `LLMProvider`        | OpenAI · Anthropic · Extractive (offline)            |

---

## Quick start

```bash
# 1. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2. Install (core deps only — runs offline)
pip install -r requirements.txt

# 3. Copy env template (optional — leave keys blank to run offline)
cp .env.example .env

# 4. Ingest the sample documents into the vector store
python -m scripts.ingest --path data/sample_docs

# 5. Run the API
uvicorn app.main:app --reload

# 6. Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-local-key" \
  -d "{\"question\": \"What vector databases does the assistant support?\", \"top_k\": 3}"
```

Interactive API docs (Swagger/OpenAPI) are served at `http://localhost:8000/docs`.

---

## Going to production (real providers)

Set these in `.env` and the service upgrades automatically — no code changes:

```env
# --- LLM ---
LLM_PROVIDER=openai            # openai | anthropic | extractive
OPENAI_API_KEY=sk-...
# or
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# --- Embeddings ---
EMBEDDING_PROVIDER=openai      # openai | huggingface | hash
OPENAI_API_KEY=sk-...

# --- Vector store ---
VECTOR_STORE=faiss             # faiss | pinecone
PINECONE_API_KEY=...
PINECONE_INDEX=rag-knowledge

# --- Security ---
API_KEYS=your-strong-key-1,your-strong-key-2
RATE_LIMIT_PER_MINUTE=60
```

Install the optional provider SDKs only when you need them:

```bash
pip install -r requirements-optional.txt   # openai, anthropic, pinecone, sentence-transformers, faiss-cpu
```

---

## Evaluation

Retrieval quality and answer faithfulness are measured with a small labelled set
in `eval/qa_pairs.json`:

```bash
python -m scripts.evaluate
```

Reports **precision@k**, **recall@k**, **MRR** for retrieval, and a
**faithfulness / citation-coverage** score for generated answers.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests run fully offline (hash embeddings + extractive LLM), so they need no keys
and are safe for CI.

---

## Endpoints

| Method | Path      | Description                                        |
|--------|-----------|----------------------------------------------------|
| GET    | `/health` | Liveness + which providers are active              |
| POST   | `/ingest` | Add documents (text) to the vector store           |
| POST   | `/query`  | Ask a question; returns answer + cited sources     |

All non-health routes require the `x-api-key` header.

---

## Tech

Python · FastAPI · Pydantic · FAISS · Pinecone · OpenAI API · Anthropic API ·
Hugging Face · NumPy · pytest

## License

MIT — see [LICENSE](LICENSE).
