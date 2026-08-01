from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "repo_cache.sqlite3"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    github_repo_owner: str = "ESHyperscale"
    github_repo_name: str = "HyperscaleES"
    github_token: str = ""

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"

    # Groq is tried first if a key is set (fast, free, no local compute needed);
    # Ollama is the always-available fallback with zero external dependency.
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Files above this size (bytes) are treated as large binary/data assets:
    # metadata is cached, but content is not stored or indexed.
    large_file_threshold: int = 500_000


settings = Settings()
