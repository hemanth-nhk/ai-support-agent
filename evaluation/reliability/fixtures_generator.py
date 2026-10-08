"""
Deterministic fixture generator for the Reliability & Adversarial Benchmark.
Provides byte payloads and on-disk files for PDF, DOCX, TXT, CSV, XLSX, and JSON test cases.
"""
import io
import json
from pathlib import Path
import docx
import openpyxl
import pandas as pd


def build_raw_pdf(pages_text: list[str]) -> bytes:
    """
    Builds a standards-compliant PDF 1.4 byte payload with multiple pages, text streams,
    and valid xref/trailer tables.
    """
    num_pages = len(pages_text)
    font_obj_idx = 3 + 2 * num_pages
    kids_refs = " ".join(f"{3 + 2*i} 0 R" for i in range(num_pages))

    body = []
    body.append(b"%PDF-1.4\n")
    offsets = [0]

    def add_obj(obj_bytes: bytes):
        offsets.append(len(b"".join(body)))
        body.append(obj_bytes)

    add_obj(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    add_obj(f"2 0 obj\n<< /Type /Pages /Kids [{kids_refs}] /Count {num_pages} >>\nendobj\n".encode("ascii"))

    for i, p_text in enumerate(pages_text):
        page_idx = 3 + 2 * i
        content_idx = 4 + 2 * i
        add_obj(
            f"{page_idx} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_idx} 0 R /Resources << /Font << /F1 {font_obj_idx} 0 R >> >> >>\nendobj\n".encode("ascii")
        )

        lines = p_text.split("\n")
        stream_parts = ["BT", "/F1 12 Tf"]
        y = 720
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_parts.append(f"72 {y} Td")
            stream_parts.append(f"({escaped}) Tj")
            stream_parts.append(f"-72 -{y} Td")
            y -= 16
        stream_parts.append("ET\n")
        s_bytes = "\n".join(stream_parts).encode("utf-8")
        s_obj = (
            f"{content_idx} 0 obj\n<< /Length {len(s_bytes)} >>\nstream\n".encode("ascii")
            + s_bytes
            + b"endstream\nendobj\n"
        )
        add_obj(s_obj)

    add_obj(
        f"{font_obj_idx} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("ascii")
    )

    xref_offset = len(b"".join(body))
    total_objs = font_obj_idx + 1
    xref = [f"xref\n0 {total_objs}\n0000000000 65535 f \n".encode("ascii")]
    for o in offsets[1:]:
        xref.append(f"{o:010d} 00000 n \n".encode("ascii"))
    xref.append(f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))

    return b"".join(body) + b"".join(xref)


def build_scanned_pdf() -> bytes:
    """Builds a minimal valid PDF page with zero text content streams (simulating scanned/image-only PDF)."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n200\n%%EOF\n"
    )


def build_docx_with_paragraphs(paragraphs: list[str]) -> bytes:
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_docx_with_headings_and_lists(headings: list[str], items: list[str]) -> bytes:
    doc = docx.Document()
    for h in headings:
        doc.add_heading(h, level=1)
    for it in items:
        doc.add_paragraph(it, style="List Bullet")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_docx_with_table(headers: list[str], rows: list[list[str]]) -> bytes:
    doc = docx.Document()
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    for c_idx, h in enumerate(headers):
        table.cell(0, c_idx).text = str(h)
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            table.cell(r_idx + 1, c_idx).text = str(val)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_docx_mixed(paragraphs: list[str], table_rows: list[list[str]]) -> bytes:
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    table = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
    for r_idx, row in enumerate(table_rows):
        for c_idx, val in enumerate(row):
            table.cell(r_idx, c_idx).text = str(val)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_csv_bytes(data: list[dict]) -> bytes:
    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def build_xlsx_bytes(data: list[dict], sheet_name: str = "Sheet1") -> bytes:
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()


def build_multi_sheet_xlsx(sheets: dict[str, list[dict]]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, data in sheets.items():
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


def generate_needle_document(needle: str, total_words: int, position_pct: float) -> str:
    """
    Generates a controlled long text document with a target needle placed at a specific percentage.
    Filler words are standard neutral sentences.
    """
    filler_sentence = "The enterprise system operates continuously and logs telemetry for all operational nodes."
    filler_words = filler_sentence.split()

    target_needle_idx = int(total_words * position_pct)
    current_words = 0
    words = []

    while current_words < target_needle_idx:
        words.extend(filler_words)
        current_words += len(filler_words)

    words.append(f"\nCRITICAL POLICY RECORD: {needle}\n")
    current_words += len(needle.split()) + 4

    while current_words < total_words:
        words.extend(filler_words)
        current_words += len(filler_words)

    return " ".join(words)
