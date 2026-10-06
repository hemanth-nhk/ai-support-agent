"""
Tests for Phase 4 — Document Retrieval & Isolation Verification.
Tests cover:
- Mandatory session_id enforcement (rejects empty/missing session_id)
- Same-session document retrieval succeeds
- Cross-session retrieval returns no chunks
- document_id parameter strictly filters to target document
- Multiple documents in one session remain distinguishable
- Deleted document cannot be retrieved
- Nonexistent document or empty collection returns empty list safely
- uploaded_documents retrieval never accesses faq_documents
- Deletion does not affect other documents, sessions, or faq_documents
- Empty/whitespace query returns empty list
- Complete verification of chunk metadata schema
"""
import io
import shutil
import tempfile
import pytest
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

from app.services import document_service
from app.services.document_service import (
    DocumentValidationError,
    DocumentNotFoundError,
    index_document,
    delete_document,
    retrieve_document_chunks,
    get_documents_collection,
    DOCUMENTS_COLLECTION_NAME,
)
from app.rag.ingestion import COLLECTION_NAME as FAQ_COLLECTION_NAME

VOCAB = [
    "shipping", "returns", "refund", "warranty", "policy", "session",
    "alpha", "beta", "document", "invoice", "terms", "agreement", "laptop",
]


class StubEmbeddingFunction(EmbeddingFunction):
    """Deterministic word-overlap stub embedding function for offline tests."""
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
        return "stub-retrieval-embedding"


@pytest.fixture
def temp_chroma_dir():
    d = tempfile.mkdtemp(prefix="test_chroma_retrieval_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def stub_emb():
    return StubEmbeddingFunction()


# ============================================================================
# 1. Validation & Safety Checks
# ============================================================================

def test_retrieval_requires_valid_session_id(temp_chroma_dir, stub_emb):
    """retrieve_document_chunks must reject missing or whitespace-only session_id."""
    with pytest.raises(DocumentValidationError) as exc1:
        retrieve_document_chunks(session_id="", query="terms", persist_dir=temp_chroma_dir, embedding_fn=stub_emb)
    assert "session_id is required" in str(exc1.value)

    with pytest.raises(DocumentValidationError) as exc2:
        retrieve_document_chunks(session_id="   ", query="terms", persist_dir=temp_chroma_dir, embedding_fn=stub_emb)
    assert "session_id is required" in str(exc2.value)

    with pytest.raises(DocumentValidationError) as exc3:
        retrieve_document_chunks(session_id=None, query="terms", persist_dir=temp_chroma_dir, embedding_fn=stub_emb)
    assert "session_id is required" in str(exc3.value)


def test_empty_or_whitespace_query_returns_empty(temp_chroma_dir, stub_emb):
    """Querying with blank text returns empty list without error."""
    index_document(
        session_id="sess_q",
        filename="notes.txt",
        content_bytes=b"Some document terms here",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    assert retrieve_document_chunks("sess_q", "", persist_dir=temp_chroma_dir, embedding_fn=stub_emb) == []
    assert retrieve_document_chunks("sess_q", "   ", persist_dir=temp_chroma_dir, embedding_fn=stub_emb) == []


def test_nonexistent_collection_or_empty_returns_safely(temp_chroma_dir, stub_emb):
    """When collection is empty, retrieval returns [] safely without Chroma error."""
    res = retrieve_document_chunks(
        session_id="nonexistent_sess",
        query="warranty",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    assert res == []


# ============================================================================
# 2. Strict Session & Document Isolation
# ============================================================================

def test_same_session_document_retrieval_succeeds(temp_chroma_dir, stub_emb):
    """Same session retrieval successfully retrieves document chunks with metadata."""
    doc_info = index_document(
        session_id="session_101",
        filename="warranty_agreement.txt",
        content_bytes=b"Comprehensive warranty policy and terms for customer hardware.",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )

    chunks = retrieve_document_chunks(
        session_id="session_101",
        query="warranty terms",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )

    assert len(chunks) >= 1
    chunk = chunks[0]
    assert "warranty" in chunk["text"].lower()
    meta = chunk["metadata"]
    assert meta["session_id"] == "session_101"
    assert meta["document_id"] == doc_info["document_id"]
    assert meta["filename"] == "warranty_agreement.txt"
    assert meta["chunk_index"] == 0
    assert "distance" in chunk


def test_cross_session_retrieval_returns_no_chunks(temp_chroma_dir, stub_emb):
    """Session B cannot retrieve chunks belonging to Session A."""
    index_document(
        session_id="session_A",
        filename="secret_doc_a.txt",
        content_bytes=b"Secret agreement for alpha session users only.",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )

    # Session B queries for the exact same terms
    chunks_b = retrieve_document_chunks(
        session_id="session_B",
        query="agreement alpha secret",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    assert chunks_b == []


def test_document_id_filtering_restricts_to_target_document(temp_chroma_dir, stub_emb):
    """Specifying document_id retrieves ONLY chunks from that document within the same session."""
    doc1 = index_document(
        session_id="session_multi",
        filename="shipping_guide.txt",
        content_bytes=b"Shipping policy: Deliveries take 3 to 5 business days.",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    doc2 = index_document(
        session_id="session_multi",
        filename="returns_guide.txt",
        content_bytes=b"Returns policy: Items can be returned within 30 days for refund.",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )

    # Query specifically for doc1
    res_doc1 = retrieve_document_chunks(
        session_id="session_multi",
        query="policy",
        document_id=doc1["document_id"],
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    assert len(res_doc1) >= 1
    assert all(c["metadata"]["document_id"] == doc1["document_id"] for c in res_doc1)
    assert all(c["metadata"]["filename"] == "shipping_guide.txt" for c in res_doc1)

    # Query specifically for doc2
    res_doc2 = retrieve_document_chunks(
        session_id="session_multi",
        query="policy",
        document_id=doc2["document_id"],
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    assert len(res_doc2) >= 1
    assert all(c["metadata"]["document_id"] == doc2["document_id"] for c in res_doc2)
    assert all(c["metadata"]["filename"] == "returns_guide.txt" for c in res_doc2)


def test_multiple_documents_in_same_session_remain_distinguishable(temp_chroma_dir, stub_emb):
    """When a session has multiple documents, querying without document_id returns chunks with distinct IDs."""
    doc1 = index_document(
        session_id="sess_both",
        filename="doc1.txt",
        content_bytes=b"First document with warranty terms.",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    doc2 = index_document(
        session_id="sess_both",
        filename="doc2.txt",
        content_bytes=b"Second document with refund policy.",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )

    # Query without document_id
    chunks = retrieve_document_chunks(
        session_id="sess_both",
        query="document policy warranty refund",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
        top_k=5,
    )

    assert len(chunks) == 2
    doc_ids = {c["metadata"]["document_id"] for c in chunks}
    assert doc_ids == {doc1["document_id"], doc2["document_id"]}


# ============================================================================
# 3. Collection Separation & Deletion Guarantees
# ============================================================================

def test_uploaded_documents_never_accesses_faq_documents(temp_chroma_dir, stub_emb):
    """
    Simulates a Chroma database with existing faq_documents.
    uploaded_documents queries must NEVER access or return faq_documents chunks.
    """
    client = chromadb.PersistentClient(path=temp_chroma_dir)

    # 1. Populate faq_documents
    faq_col = client.get_or_create_collection(
        name=FAQ_COLLECTION_NAME,
        embedding_function=stub_emb,
        metadata={"hnsw:space": "cosine"},
    )
    faq_col.add(
        ids=["faq_1", "faq_2"],
        documents=[
            "Store FAQ: Shipping is free on orders over $50.",
            "Store FAQ: Standard returns accepted within 30 days.",
        ],
        metadatas=[{"source": "faq_shipping"}, {"source": "faq_returns"}],
    )
    assert faq_col.count() == 2

    # 2. Query uploaded documents for a session that has NO uploads
    results = retrieve_document_chunks(
        session_id="empty_session",
        query="shipping returns",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    assert results == [], "uploaded_documents retrieval must never return chunks from faq_documents"


def test_deletion_removes_all_chunks_and_does_not_affect_others(temp_chroma_dir, stub_emb):
    """
    Deleting a document removes all of its chunks, and does NOT affect:
    - other documents in the same session
    - documents in other sessions
    - faq_documents collection
    """
    client = chromadb.PersistentClient(path=temp_chroma_dir)

    # Setup faq_documents
    faq_col = client.get_or_create_collection(
        name=FAQ_COLLECTION_NAME,
        embedding_function=stub_emb,
        metadata={"hnsw:space": "cosine"},
    )
    faq_col.add(
        ids=["faq_chunk_1"],
        documents=["Store policy content"],
        metadatas=[{"category": "policy"}],
    )

    # Ingest Doc A1 in Session A (multi-chunk)
    long_content = b"Paragraph one with warranty terms.\n\n" * 15
    doc_a1 = index_document(
        session_id="Session_A",
        filename="doc_a1.txt",
        content_bytes=long_content,
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    assert doc_a1["chunk_count"] > 1

    # Ingest Doc A2 in Session A
    doc_a2 = index_document(
        session_id="Session_A",
        filename="doc_a2.txt",
        content_bytes=b"Doc A2 with invoice details.",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )

    # Ingest Doc B in Session B
    doc_b = index_document(
        session_id="Session_B",
        filename="doc_b.txt",
        content_bytes=b"Doc B with agreement terms.",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )

    # Now delete Doc A1
    del_result = delete_document(
        document_id=doc_a1["document_id"],
        session_id="Session_A",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    assert del_result["status"] == "deleted"
    assert del_result["deleted_chunks"] == doc_a1["chunk_count"]

    # 1. Doc A1 cannot be retrieved
    res_a1 = retrieve_document_chunks(
        session_id="Session_A",
        query="warranty terms",
        document_id=doc_a1["document_id"],
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    assert res_a1 == []

    # 2. Doc A2 is completely intact
    res_a2 = retrieve_document_chunks(
        session_id="Session_A",
        query="invoice",
        document_id=doc_a2["document_id"],
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    assert len(res_a2) >= 1
    assert res_a2[0]["metadata"]["document_id"] == doc_a2["document_id"]

    # 3. Doc B is completely intact
    res_b = retrieve_document_chunks(
        session_id="Session_B",
        query="agreement",
        document_id=doc_b["document_id"],
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_emb,
    )
    assert len(res_b) >= 1
    assert res_b[0]["metadata"]["document_id"] == doc_b["document_id"]

    # 4. faq_documents is completely unaffected
    assert faq_col.count() == 1
    faq_get = faq_col.get(ids=["faq_chunk_1"])
    assert len(faq_get["ids"]) == 1
