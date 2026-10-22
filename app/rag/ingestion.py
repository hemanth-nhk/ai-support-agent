"""
FAQ ingestion pipeline: load .txt files from data/faq/, chunk, embed with
sentence-transformers, and persist into a local Chroma collection.

Run this whenever the FAQ documents change:
    python -m app.rag.ingestion
"""
from pathlib import Path
import chromadb

from app.config import settings
from app.rag.embeddings import get_embedding_function

COLLECTION_NAME = "faq_documents"


def _load_faq_documents() -> list[dict]:
    """Loads each .txt file as one or more chunks. Simple paragraph-based
    chunking is sufficient for the small, well-structured FAQ files here."""
    docs = []
    faq_dir = Path(settings.faq_dir)
    for filepath in sorted(faq_dir.glob("*.txt")):
        text = filepath.read_text(encoding="utf-8")
        # Split on blank lines -> paragraph-level chunks; drop the
        # "DEMO DATA" disclaimer line and empty chunks.
        raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        chunks = [c for c in raw_chunks if not c.startswith("DEMO DATA")]
        if not chunks:
            continue
        # If the first chunk is a short title/header, attach it to subsequent content chunks
        # so standalone titles don't get indexed without their policy body (DOC-01).
        if len(chunks) > 1 and len(chunks[0].splitlines()) <= 2 and len(chunks[0]) < 60:
            header = chunks[0]
            content_chunks = [f"{header}:\n{c}" for c in chunks[1:]]
        else:
            content_chunks = chunks
        for i, chunk in enumerate(content_chunks):
            docs.append({
                "id": f"{filepath.stem}-{i}",
                "text": chunk,
                "source": filepath.name,
            })
    return docs


def build_index(persist_dir: str | None = None, embedding_fn=None) -> int:
    """Builds (or rebuilds) the persistent Chroma collection.
    Returns the number of chunks indexed.

    persist_dir / embedding_fn are injectable so tests can point at a
    throwaway directory and a deterministic stub embedding function
    instead of the real sentence-transformers model -- see
    tests/test_rag.py. Production code (see __main__ below) uses the
    real settings-driven values.
    """
    client = chromadb.PersistentClient(path=persist_dir or settings.chroma_persist_dir)

    # Rebuild cleanly each run so stale FAQ content never lingers.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection may not exist yet

    embedding_fn = embedding_fn or get_embedding_function()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        # Explicitly configure cosine distance. Chroma's default is L2
        # (squared Euclidean), which is on a completely different scale
        # from the cosine-distance threshold used in retriever.py. Without
        # this, RAG_SIMILARITY_THRESHOLD is being compared against the
        # wrong metric entirely -- this line is what makes that threshold
        # mean what its name says.
        metadata={"hnsw:space": "cosine"},
    )

    docs = _load_faq_documents()
    if not docs:
        print("No FAQ documents found -- nothing indexed.")
        return 0

    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs],
    )
    print(f"Indexed {len(docs)} FAQ chunks into Chroma collection '{COLLECTION_NAME}'.")
    return len(docs)


if __name__ == "__main__":
    build_index()
