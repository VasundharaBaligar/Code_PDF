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

    # The paper this repo implements ("Evolution Strategies at the Hyperscale").
    # We ingest arXiv's LaTeX source rather than the PDF: equations arrive as
    # real LaTeX instead of OCR guesswork, and \label{} anchors give precise,
    # named citation targets.
    arxiv_id: str = "2511.16652"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"

    # Groq is tried first if a key is set (fast, free, no local compute needed);
    # Ollama is the always-available fallback with zero external dependency.
    groq_api_key: str = ""
    # Preference order, best first. Groq retires models without notice -- it
    # dropped every Llama chat model mid-project -- so pinning a single one
    # means an upstream change silently demotes every answer to the local
    # fallback. The provider resolves this list against Groq's live catalogue
    # and uses the best entry that actually exists.
    groq_models: str = "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.6-27b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    @property
    def groq_model_preferences(self) -> list[str]:
        return [m.strip() for m in self.groq_models.split(",") if m.strip()]

    # Files above this size (bytes) are treated as large binary/data assets:
    # metadata is cached, but content is not stored or indexed.
    large_file_threshold: int = 500_000


settings = Settings()
