"""
Centralized configuration. All values are overridable via .env — never hardcode
secrets or environment-specific paths elsewhere in the codebase.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database ---
    database_path: str = str(BASE_DIR / "data" / "support_agent.db")

    # --- Access control (demo-level only, see app/api/auth.py) ---
    support_api_key: str = ""

    # --- RAG / Chroma ---
    faq_dir: str = str(BASE_DIR / "data" / "faq")
    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma_store")
    embedding_model_name: str = "all-MiniLM-L6-v2"
    rag_similarity_threshold: float = 0.42  # cosine distance threshold calibrated for all-MiniLM-L6-v2
    rag_top_k: int = 3
    uploaded_doc_similarity_threshold: float | None = None  # None allows broad document questions; calibrated for empirical distance tuning

    # --- LLM (Ollama, local only) ---
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"

    # --- App ---
    app_name: str = "AI Customer Support Agent"
    max_message_length: int = 500

    # --- Tabular Analytics / Datasets ---
    datasets_dir: str = str(BASE_DIR / "data" / "datasets")
    charts_dir: str = str(BASE_DIR / "data" / "charts")
    max_dataset_file_size_mb: int = 50

    # --- Session Security ---
    session_secret_key: str = ""


settings = Settings()
