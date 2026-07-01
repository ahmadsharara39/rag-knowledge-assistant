# Vector Databases and Semantic Search

Semantic search finds information by meaning rather than exact keyword matches.
Text is converted into embeddings — dense numeric vectors — and similar meanings
end up close together in vector space, measured by cosine similarity.

A vector database stores these embeddings and answers nearest-neighbour queries
efficiently. This Enterprise RAG Knowledge Assistant supports two vector stores:
FAISS for fast local, in-process indexing, and Pinecone as a managed, horizontally
scalable service. FAISS uses an inner-product index over L2-normalised vectors,
which is mathematically equivalent to cosine similarity. Pinecone stores each
chunk's text and metadata alongside its vector so results can be reconstructed
without a separate document store.

Other popular vector databases include Weaviate and Milvus. The choice depends on
scale, latency, and operational preferences: FAISS is ideal for embedded and
single-node use, while Pinecone and Weaviate suit large multi-tenant deployments.
