"""
Tests for Phase 3 — File Assistance v1: Backend Foundation.
Tests cover:
- PDF extraction
- DOCX extraction
- TXT extraction (UTF-8 and Latin-1 fallback)
- Unsupported file types
- Empty document rejection
- Oversized file rejection
- Deterministic chunking
- Successful upload endpoint
- Document metadata validation in Chroma
- Session isolation
- Document deletion endpoint
- Deleting unknown document / session mismatch
- Malformed / invalid upload requests
"""
import io
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient
from chromadb import EmbeddingFunction, Documents, Embeddings

from app.main import app
from app.config import settings
from app.services import document_service
from app.services.document_service import (
    DocumentValidationError,
    DocumentNotFoundError,
    validate_file,
    extract_text,
    chunk_text,
    index_document,
    delete_document,
    retrieve_document_chunks,
    sanitize_filename,
    MAX_FILE_SIZE_BYTES,
)

VOCAB = [
    "pdf", "docx", "txt", "document", "contract", "terms", "policy", "refund",
    "invoice", "agreement", "hello", "world", "extraction", "testing", "section",
]


class StubEmbeddingFunction(EmbeddingFunction):
    """Deterministic word-overlap stub embedding function for offline testing."""
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
        return "stub-doc-embedding"


def make_sample_pdf(text: str = "Hello World PDF Extraction") -> bytes:
    """Generates a minimal valid PDF 1.4 byte payload containing the specified text."""
    # PDF stream with text command
    text_escaped = text.replace("(", "\\(").replace(")", "\\)")
    stream_content = f"BT\n/F1 12 Tf\n72 712 Td\n({text_escaped}) Tj\nET\n".encode("utf-8")
    stream_len = len(stream_content)

    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length " + str(stream_len).encode("ascii") + b" >>\nstream\n"
        + stream_content
        + b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000244 00000 n \n"
        b"0000000338 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n412\n%%EOF\n"
    )
    return pdf


def make_sample_docx(paragraphs: list[str]) -> bytes:
    """Generates a DOCX byte payload containing paragraphs."""
    import docx
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def temp_chroma_dir():
    d = tempfile.mkdtemp(prefix="test_chroma_docs_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def client_with_mock_chroma(temp_chroma_dir, monkeypatch):
    """Configures TestClient with temporary Chroma directory and stub embedding function."""
    stub_fn = StubEmbeddingFunction()
    monkeypatch.setattr(settings, "chroma_persist_dir", temp_chroma_dir)
    monkeypatch.setattr(document_service, "get_embedding_function", lambda: stub_fn)
    return TestClient(app), temp_chroma_dir, stub_fn


# ============================================================================
# 1. Extraction Tests (PDF, DOCX, TXT)
# ============================================================================

def test_pdf_extraction():
    pdf_bytes = make_sample_pdf("Customer Support Policy PDF Document")
    text = extract_text("policy.pdf", pdf_bytes)
    assert "Customer Support Policy PDF Document" in text


def test_docx_extraction():
    docx_bytes = make_sample_docx(["First paragraph of terms.", "Second paragraph of conditions."])
    text = extract_text("terms.docx", docx_bytes)
    assert "First paragraph of terms." in text
    assert "Second paragraph of conditions." in text


def test_txt_extraction_utf8():
    txt_bytes = "This is a simple UTF-8 text document with information.".encode("utf-8")
    text = extract_text("notes.txt", txt_bytes)
    assert text == "This is a simple UTF-8 text document with information."


def test_txt_extraction_latin1_fallback():
    # Latin-1 characters that would fail strict UTF-8 decoding
    txt_bytes = "Document with café and résumé in Latin-1".encode("latin-1")
    text = extract_text("latin.txt", txt_bytes)
    assert "café" in text or "caf" in text


# ============================================================================
# 2. Validation Tests (Unsupported type, Empty document, Oversized, Sanitization)
# ============================================================================

def test_unsupported_file_type():
    with pytest.raises(DocumentValidationError) as exc:
        validate_file("data.csv", b"col1,col2\nval1,val2")
    assert "Unsupported file type '.csv'" in str(exc.value)

    with pytest.raises(DocumentValidationError):
        validate_file("image.png", b"\x89PNG\r\n\x1a\n")


def test_empty_document_rejection():
    # 0-byte file
    with pytest.raises(DocumentValidationError) as exc:
        validate_file("empty.txt", b"")
    assert "empty" in str(exc.value).lower()

    # Whitespace-only text file
    with pytest.raises(DocumentValidationError):
        extract_text("blank.txt", b"   \n\t  ")


def test_oversized_file_rejection():
    oversized = b"x" * (MAX_FILE_SIZE_BYTES + 10)
    with pytest.raises(DocumentValidationError) as exc:
        validate_file("big.pdf", oversized)
    assert "exceeds maximum allowed size" in str(exc.value)


def test_filename_sanitization():
    assert sanitize_filename("../../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("contract (v2) [final]!.docx") == "contract__v2___final__.docx"
    assert sanitize_filename("") == "uploaded_document.txt"


# ============================================================================
# 3. Deterministic Chunking Tests
# ============================================================================

def test_deterministic_chunking():
    text = "Sentence one. " * 30  # ~420 characters
    chunks_1 = chunk_text(text, chunk_size=100, chunk_overlap=20)
    chunks_2 = chunk_text(text, chunk_size=100, chunk_overlap=20)

    # Must be deterministic and non-empty
    assert chunks_1 == chunks_2
    assert len(chunks_1) > 1

    # Short text within chunk size returns 1 chunk
    short_chunks = chunk_text("Small document", chunk_size=100, chunk_overlap=20)
    assert short_chunks == ["Small document"]

    # Empty text returns empty list
    assert chunk_text("") == []


# ============================================================================
# 4. Successful Upload & Metadata Tests
# ============================================================================

def test_successful_upload_txt(client_with_mock_chroma):
    client, persist_dir, stub_fn = client_with_mock_chroma
    txt_content = b"User agreement terms for return and refund policies."

    response = client.post(
        "/documents/upload",
        files={"file": ("agreement.txt", txt_content, "text/plain")},
        data={"session_id": "session-100"},
    )

    assert response.status_code == 201
    body = response.json()
    assert "document_id" in body
    assert body["filename"] == "agreement.txt"
    assert body["chunk_count"] >= 1

    # Verify metadata in ChromaDB
    chunks = retrieve_document_chunks(
        session_id="session-100",
        query="refund",
        persist_dir=persist_dir,
        embedding_fn=stub_fn,
    )
    assert len(chunks) >= 1
    meta = chunks[0]["metadata"]
    assert meta["session_id"] == "session-100"
    assert meta["document_id"] == body["document_id"]
    assert meta["filename"] == "agreement.txt"
    assert "chunk_index" in meta


def test_successful_upload_pdf(client_with_mock_chroma):
    client, persist_dir, stub_fn = client_with_mock_chroma
    pdf_content = make_sample_pdf("PDF contract details about order refund and warranty terms.")

    response = client.post(
        "/documents/upload",
        files={"file": ("contract.pdf", pdf_content, "application/pdf")},
        data={"session_id": "session-200"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "contract.pdf"
    assert body["chunk_count"] >= 1


def test_successful_upload_docx(client_with_mock_chroma):
    client, persist_dir, stub_fn = client_with_mock_chroma
    docx_content = make_sample_docx(["Company policy section on item replacement."])

    response = client.post(
        "/documents/upload",
        files={"file": ("policy.docx", docx_content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"session_id": "session-300"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "policy.docx"
    assert body["chunk_count"] >= 1


# ============================================================================
# 5. Session Isolation Tests
# ============================================================================

def test_session_isolation(temp_chroma_dir):
    stub_fn = StubEmbeddingFunction()

    # Index doc 1 for session A
    res_a = index_document(
        session_id="session_A",
        filename="docA.txt",
        content_bytes=b"Sensitive information belonging strictly to Session A customer.",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_fn,
    )

    # Index doc 2 for session B
    res_b = index_document(
        session_id="session_B",
        filename="docB.txt",
        content_bytes=b"Completely different information belonging to Session B user.",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_fn,
    )

    # Query with session A
    chunks_a = retrieve_document_chunks(
        session_id="session_A",
        query="information",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_fn,
    )
    assert len(chunks_a) >= 1
    assert all(c["metadata"]["session_id"] == "session_A" for c in chunks_a)
    assert all(c["metadata"]["document_id"] == res_a["document_id"] for c in chunks_a)
    assert not any(c["metadata"]["session_id"] == "session_B" for c in chunks_a)

    # Query with session B
    chunks_b = retrieve_document_chunks(
        session_id="session_B",
        query="information",
        persist_dir=temp_chroma_dir,
        embedding_fn=stub_fn,
    )
    assert len(chunks_b) >= 1
    assert all(c["metadata"]["session_id"] == "session_B" for c in chunks_b)
    assert all(c["metadata"]["document_id"] == res_b["document_id"] for c in chunks_b)
    assert not any(c["metadata"]["session_id"] == "session_A" for c in chunks_b)


# ============================================================================
# 6. Document Deletion Tests
# ============================================================================

def test_document_deletion_endpoint(client_with_mock_chroma):
    client, persist_dir, stub_fn = client_with_mock_chroma
    txt_content = b"Document to be uploaded and subsequently deleted."

    upload_res = client.post(
        "/documents/upload",
        files={"file": ("to_delete.txt", txt_content, "text/plain")},
        data={"session_id": "session-del"},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # Delete the document with matching session_id
    del_res = client.delete(f"/documents/{doc_id}?session_id=session-del")
    assert del_res.status_code == 200
    del_body = del_res.json()
    assert del_body["status"] == "deleted"
    assert del_body["document_id"] == doc_id
    assert del_body["deleted_chunks"] >= 1

    # Verify no chunks remain in Chroma
    remaining = retrieve_document_chunks(
        session_id="session-del",
        query="deleted",
        persist_dir=persist_dir,
        embedding_fn=stub_fn,
    )
    assert len(remaining) == 0


def test_delete_unknown_document_returns_404(client_with_mock_chroma):
    client, _, _ = client_with_mock_chroma
    res = client.delete("/documents/non-existent-doc-id")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_delete_with_session_mismatch_returns_404(client_with_mock_chroma):
    client, _, _ = client_with_mock_chroma
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("owner.txt", b"Owner data", "text/plain")},
        data={"session_id": "real-session"},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # Attempt delete with wrong session_id
    del_res = client.delete(f"/documents/{doc_id}?session_id=wrong-session")
    assert del_res.status_code == 404


# ============================================================================
# 7. Malformed / Invalid Upload Requests Tests
# ============================================================================

def test_upload_missing_session_id(client_with_mock_chroma):
    client, _, _ = client_with_mock_chroma
    res = client.post(
        "/documents/upload",
        files={"file": ("test.txt", b"Hello content", "text/plain")},
    )
    # FastAPI form validation failure
    assert res.status_code == 422


def test_upload_empty_session_id(client_with_mock_chroma):
    client, _, _ = client_with_mock_chroma
    res = client.post(
        "/documents/upload",
        files={"file": ("test.txt", b"Hello content", "text/plain")},
        data={"session_id": "   "},
    )
    assert res.status_code == 400
    assert "valid session_id is required" in res.json()["detail"].lower()


def test_upload_unsupported_file_extension(client_with_mock_chroma):
    client, _, _ = client_with_mock_chroma
    res = client.post(
        "/documents/upload",
        files={"file": ("spreadsheet.xlsx", b"dummy xlsx bytes", "application/octet-stream")},
        data={"session_id": "session-1"},
    )
    assert res.status_code == 400
    assert "unsupported file type" in res.json()["detail"].lower()


def test_upload_empty_file_returns_400(client_with_mock_chroma):
    client, _, _ = client_with_mock_chroma
    res = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
        data={"session_id": "session-1"},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


# ============================================================================
# 8. DOCX Table & Header/Footer Extraction Tests
# ============================================================================

def test_docx_table_only_extraction():
    """Verifies that facts contained strictly in DOCX table cells are extracted."""
    import docx
    doc = docx.Document()
    table = doc.add_table(rows=3, cols=2)
    table.cell(0, 0).text = "Customer ID"
    table.cell(0, 1).text = "C1024"
    table.cell(1, 0).text = "Refund period"
    table.cell(1, 1).text = "30 days"
    table.cell(2, 0).text = "Warranty"
    table.cell(2, 1).text = "12 months"
    buf = io.BytesIO()
    doc.save(buf)
    text = extract_text("table_only.docx", buf.getvalue())
    assert "C1024" in text
    assert "30 days" in text
    assert "12 months" in text


def test_docx_mixed_paragraph_and_table_extraction():
    """Verifies that facts split across body paragraphs and tables are extracted."""
    import docx
    doc = docx.Document()
    doc.add_paragraph("Customer ID: C1024")
    doc.add_paragraph("Refund period: 30 days")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Warranty"
    table.cell(0, 1).text = "12 months"
    table.cell(1, 0).text = "Approval limit"
    table.cell(1, 1).text = "50,000"
    buf = io.BytesIO()
    doc.save(buf)
    text = extract_text("mixed.docx", buf.getvalue())
    assert "C1024" in text
    assert "30 days" in text
    assert "12 months" in text
    assert "50,000" in text


def test_docx_header_footer_extraction():
    """Verifies that facts in headers, footers, and body paragraphs are extracted deterministically."""
    import docx
    doc = docx.Document()
    sec = doc.sections[0]
    sec.header.paragraphs[0].text = "Header Policy: Customer ID: C1024"
    sec.footer.paragraphs[0].text = "Footer Note: Approval limit: 50,000"
    doc.add_paragraph("Body: Refund period: 30 days. Warranty: 12 months.")
    buf = io.BytesIO()
    doc.save(buf)
    text = extract_text("header_footer.docx", buf.getvalue())
    assert "C1024" in text
    assert "50,000" in text
    assert "30 days" in text
    assert "12 months" in text
