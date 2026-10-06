"""
FAQ retrieval with an enforced similarity threshold. A retrieved chunk
that is too dissimilar from the query is treated as NO RESULT rather than
being handed to the LLM -- this is what prevents "confidently wrong"
answers on questions the FAQ doesn't actually cover.

The collection is explicitly configured for cosine distance at creation
time (see app/rag/ingestion.py -- metadata={"hnsw:space": "cosine"}).
This matters: Chroma's default is L2 (squared Euclidean), which is on a
different scale entirely, so the threshold below would be meaningless
without that explicit configuration.

Cosine distance ranges 0 (identical) to 2 (opposite).
settings.rag_similarity_threshold is a MAXIMUM allowed distance -- lower
is stricter. This needs empirical tuning per embedding model; the
default (0.35) is a reasonable starting point for all-MiniLM-L6-v2 on
short FAQ paragraphs and should be validated against real queries
during Phase 17 verification (see WINDOWS_SETUP_CHECKLIST.md step 10d).
"""
import chromadb

from app.config import settings
from app.rag.embeddings import get_embedding_function
from app.rag.ingestion import COLLECTION_NAME


def retrieve_faq(query: str, persist_dir: str | None = None, embedding_fn=None,
                  threshold: float | None = None) -> dict:
    """
    Returns {"found": bool, "chunks": [str, ...], "count": int}.
    "found" is False if no chunk clears the similarity threshold --
    callers must not fall back to the LLM's own knowledge in that case.

    persist_dir / embedding_fn / threshold are injectable so tests can
    exercise this against a deterministic stub instead of the real
    sentence-transformers model -- see tests/test_rag.py. Production
    calls (via app/agent/nodes.py) use the real settings-driven values.
    """
    client = chromadb.PersistentClient(path=persist_dir or settings.chroma_persist_dir)
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn or get_embedding_function(),
        )
    except Exception:
        # Index not built yet -- fail closed, not silently.
        return {"found": False, "chunks": [], "count": 0}

    results = collection.query(query_texts=[query], n_results=settings.rag_top_k)

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    effective_threshold = threshold if threshold is not None else settings.rag_similarity_threshold
    accepted = [
        doc for doc, dist in zip(documents, distances)
        if dist <= effective_threshold
    ]

    return {
        "found": len(accepted) > 0,
        "chunks": accepted,
        "count": len(accepted),
    }
