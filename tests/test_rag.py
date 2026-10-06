"""
RAG pipeline tests. Uses a deterministic word-overlap stub embedding
function (not the real sentence-transformers model) so these run without
network access or a torch install -- CI-friendly. This exercises the real
mechanics: Chroma collection creation with explicit cosine space
configuration, real add/query round-trips, and real threshold filtering
logic in retriever.py.

This does NOT prove the real sentence-transformers embeddings + the
default 0.35 threshold produce good results on real FAQ text -- that
still needs the manual check in WINDOWS_SETUP_CHECKLIST.md step 10d. What
this DOES prove: the cosine-space bug is fixed, and retrieve_faq()
correctly accepts/rejects chunks based on distance vs threshold.
"""
import shutil
import tempfile
import pytest
from chromadb import EmbeddingFunction, Documents, Embeddings

from app.rag.ingestion import build_index, COLLECTION_NAME
from app.rag.retriever import retrieve_faq

VOCAB = [
    "shipping", "ship", "return", "returns", "refund", "refunds", "cancel",
    "cancellation", "policy", "day", "days", "order", "orders", "delivery",
    "deliver", "how", "long", "take", "takes", "back", "money", "laptop",
    "warranty", "battery", "bluetooth", "unused", "packaging",
]


class StubEmbeddingFunction(EmbeddingFunction):
    """Deterministic word-overlap embedding -- no ML model, no network.
    Good enough to prove retrieval mechanics; not a stand-in for real
    semantic quality."""
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            lowered = text.lower()
            vec = [1.0 if word in lowered else 0.0 for word in VOCAB]
            out.append(vec)
        return out

    def name(self) -> str:
        return "stub-word-overlap"


@pytest.fixture
def temp_chroma_dir():
    d = tempfile.mkdtemp(prefix="test_chroma_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def indexed_dir(temp_chroma_dir):
    count = build_index(persist_dir=temp_chroma_dir, embedding_fn=StubEmbeddingFunction())
    assert count > 0, "ingestion found no FAQ chunks -- check data/faq/ is populated"
    return temp_chroma_dir


def test_ingestion_indexes_all_faq_files(indexed_dir):
    import chromadb
    client = chromadb.PersistentClient(path=indexed_dir)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=StubEmbeddingFunction())
    assert collection.count() > 0


def test_collection_configured_for_cosine_space(indexed_dir):
    """Regression test for the P0 bug: without explicit cosine
    configuration, Chroma defaults to L2 and the threshold is meaningless."""
    import chromadb
    client = chromadb.PersistentClient(path=indexed_dir)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=StubEmbeddingFunction())
    assert collection.metadata is not None
    assert collection.metadata.get("hnsw:space") == "cosine"


def test_known_match_query_is_found(indexed_dir):
    """A query that clearly overlaps with the shipping FAQ should be
    accepted at a reasonably generous threshold."""
    result = retrieve_faq(
        "how long does shipping take",
        persist_dir=indexed_dir,
        embedding_fn=StubEmbeddingFunction(),
        threshold=0.9,
    )
    assert result["found"] is True
    assert result["count"] >= 1
    assert any("3-5 business days" in c or "shipping" in c.lower() for c in result["chunks"])


def test_known_miss_query_is_rejected_at_strict_threshold(indexed_dir):
    """A query with no real vocabulary overlap against any FAQ document
    should be rejected once the threshold is strict enough -- this is
    what prevents a weak match from being treated as a real answer."""
    result = retrieve_faq(
        "laptop battery bluetooth pairing issue",
        persist_dir=indexed_dir,
        embedding_fn=StubEmbeddingFunction(),
        threshold=0.05,  # very strict -- only near-identical vectors pass
    )
    assert result["found"] is False
    assert result["count"] == 0


def test_missing_index_fails_closed(temp_chroma_dir):
    """No collection built yet -- must return not-found, not raise."""
    result = retrieve_faq(
        "anything",
        persist_dir=temp_chroma_dir,
        embedding_fn=StubEmbeddingFunction(),
    )
    assert result["found"] is False
    assert result["chunks"] == []
    assert result["count"] == 0
