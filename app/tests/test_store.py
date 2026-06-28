"""Tests for docaware.rag.store.VectorStore (NumPy cosine index)."""

from docaware.rag.chunk import Chunk
from docaware.rag.store import VectorStore


def _chunk(txt: str, i: int) -> Chunk:
    return Chunk(text=txt, source="doc", index=i)


def test_add_and_search_ranks_by_similarity():
    store = VectorStore(dim=3)
    store.add(
        [_chunk("east", 0), _chunk("north", 1), _chunk("up", 2)],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    results = store.search([0.9, 0.1, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0][0].text == "east"  # closest to the query vector
    assert results[0][1] >= results[1][1]  # descending score order


def test_search_empty_store_returns_empty():
    assert VectorStore(dim=3).search([1.0, 0.0, 0.0]) == []


def test_top_k_capped_to_size():
    store = VectorStore(dim=2)
    store.add([_chunk("a", 0)], [[1.0, 0.0]])
    assert len(store.search([1.0, 0.0], top_k=10)) == 1


def test_save_and_load_roundtrip(tmp_path):
    store = VectorStore(dim=2)
    store.add([_chunk("a", 0), _chunk("b", 1)], [[1.0, 0.0], [0.0, 1.0]])
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 2
    top = loaded.search([1.0, 0.0], top_k=1)
    assert top[0][0].text == "a"


def test_load_missing_index_is_empty():
    assert len(VectorStore.load("does/not/exist")) == 0
