from app.rag.chunking import chunk_text


def test_chunks_respect_max_size():
    text = " ".join(f"Sentence number {i} about retrieval." for i in range(200))
    chunks = chunk_text(text, source="doc.md", chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_chunk_ids_are_stable_and_unique():
    text = "First sentence. Second sentence. Third sentence."
    a = chunk_text(text, source="doc.md", chunk_size=20, overlap=5)
    b = chunk_text(text, source="doc.md", chunk_size=20, overlap=5)
    assert [c.id for c in a] == [c.id for c in b]  # deterministic
    assert len({c.id for c in a}) == len(a)  # unique within a doc


def test_oversized_single_sentence_is_hard_split():
    text = "x" * 1000  # no sentence boundaries
    chunks = chunk_text(text, source="doc.md", chunk_size=100, overlap=0)
    assert len(chunks) >= 10
    assert all(len(c.text) <= 100 for c in chunks)


def test_metadata_is_carried_through():
    chunks = chunk_text(
        "Hello world.", source="doc.md", base_metadata={"doc_id": "42"}
    )
    assert chunks[0].metadata["doc_id"] == "42"
