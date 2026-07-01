from app.rag.chunking import Chunk
from app.rag.vector_store import FaissStore


def _vec(seed: float, dim: int = 8):
    v = [0.0] * dim
    v[int(seed) % dim] = 1.0
    return v


def test_add_is_idempotent_by_chunk_id():
    store = FaissStore(dim=8)
    chunks = [Chunk(id="a", text="alpha", source="s"), Chunk(id="b", text="beta", source="s")]
    vectors = [_vec(0), _vec(1)]
    store.add(chunks, vectors)
    store.add(chunks, vectors)  # same ids again
    assert store.size == 2  # no duplicates


def test_search_returns_nearest_first():
    store = FaissStore(dim=8)
    store.add(
        [Chunk(id="a", text="alpha", source="s"), Chunk(id="b", text="beta", source="s")],
        [_vec(0), _vec(1)],
    )
    hits = store.search(_vec(1), top_k=2)
    assert hits[0].chunk.id == "b"


def test_persist_and_load_roundtrip(tmp_path):
    store = FaissStore(dim=8)
    store.add([Chunk(id="a", text="alpha", source="s")], [_vec(0)])
    store.persist(str(tmp_path))

    reloaded = FaissStore(dim=8)
    assert reloaded.load(str(tmp_path))
    assert reloaded.size == 1
    # ids restored, so re-adding the same chunk is still idempotent
    reloaded.add([Chunk(id="a", text="alpha", source="s")], [_vec(0)])
    assert reloaded.size == 1
