# Security and Deployment

The service is designed for secure, enterprise environments. Every non-health
endpoint requires an API key supplied in the `x-api-key` header, validated against
a configured allow-list. Requests are rate limited per API key using a sliding
sixty-second window; when the limit is exceeded the service returns HTTP 429 with a
Retry-After header. All request bodies are validated with strict Pydantic schemas,
so malformed or oversized input is rejected before it reaches the pipeline.

Secrets such as API keys for OpenAI, Anthropic, and Pinecone are read from
environment variables and never hard-coded, which keeps them out of source control
and lets each deployment environment inject its own credentials.

Providers are pluggable through environment variables. The LLM can be OpenAI or
Anthropic; embeddings can come from OpenAI or a local Hugging Face
sentence-transformers model; the vector store can be FAISS or Pinecone. With no
keys set, the service runs fully offline using deterministic fallback providers,
which makes it safe to test in continuous integration without external calls.
