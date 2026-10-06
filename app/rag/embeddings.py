"""
Embedding function for Chroma, backed by sentence-transformers.
Runs fully locally -- no API key, no network call at query time (the
model itself is downloaded once from Hugging Face on first run).
"""
from chromadb.utils import embedding_functions
from app.config import settings


def get_embedding_function():
    """
    Uses Chroma's built-in SentenceTransformerEmbeddingFunction wrapper,
    which loads the model named in settings.embedding_model_name
    (default: all-MiniLM-L6-v2 -- small, fast, well-suited to a CPU-only
    local demo).
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model_name
    )
