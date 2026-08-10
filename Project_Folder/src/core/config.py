"""Configuration-driven settings system backed by Pydantic v2 Settings and YAML.

Reads from:
1. config/settings.yaml (application defaults)
2. .env file (secrets, environment variables, credentials)
3. Dynamic runtime environment variables (highest priority)
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
import yaml

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseModel):
    name: str = "Enterprise-RAG-StarterKit"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    log_format: Literal["dev", "json"] = "dev"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = ["*"]
    api_key_secret: SecretStr = SecretStr("default_secret_key")


class LLMSettings(BaseModel):
    default_provider: str = "openai"
    default_model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 2048

    # Provider API Keys
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"


class EmbeddingSettings(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536


class StorageSettings(BaseModel):
    vector_db_type: str = "pgvector"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag_db"
    postgres_user: str = "rag_user"
    postgres_password: SecretStr = SecretStr("rag_secure_password")
    postgres_schema: str = "public"
    vector_collection_name: str = "enterprise_knowledge_base"

    # HNSW Index Parameters
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40

    # Redis Connection
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: SecretStr | None = None
    redis_db: int = 0

    @property
    def postgres_async_url(self) -> str:
        pwd = self.postgres_password.get_secret_value()
        return f"postgresql+asyncpg://{self.postgres_user}:{pwd}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def postgres_sync_url(self) -> str:
        pwd = self.postgres_password.get_secret_value()
        return f"postgresql+psycopg://{self.postgres_user}:{pwd}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        pwd = self.redis_password.get_secret_value() if self.redis_password else ""
        auth = f":{pwd}@" if pwd else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


class IngestionSettings(BaseModel):
    chunking_strategy: Literal["hierarchical", "semantic", "recursive"] = "hierarchical"
    parent_chunk_size: int = 1200
    child_chunk_size: int = 250
    chunk_overlap: int = 40
    extract_markdown_tables: bool = True
    enable_ocr: bool = True
    ocr_language: str = "eng"
    enable_sha256_deduplication: bool = True


class SemanticCacheSettings(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.96
    ttl_seconds: int = 86400


class AgentSettings(BaseModel):
    semantic_cache: SemanticCacheSettings = Field(default_factory=SemanticCacheSettings)
    max_sub_queries: int = 3
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_k: int = 60
    reranker_provider: str = "flashrank"
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2"
    reranker_top_n: int = 5
    reranker_min_threshold: float = 0.20
    cohere_api_key: SecretStr | None = None
    enable_hallucination_check: bool = True
    max_reflection_retries: int = 1


class ObservabilitySettings(BaseModel):
    backend: Literal["langsmith", "langfuse", "phoenix", "none"] = "langfuse"
    strict_remote_hub: bool = True
    remote_prompt_name: str = "rag-qa-enterprise"
    remote_prompt_version: str = "production"
    prompt_cache_ttl_seconds: int = 300

    # Langfuse
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "enterprise-rag-starter"

    # Phoenix
    phoenix_endpoint: str = "http://localhost:6006"


class SupervisorSettings(BaseModel):
    timeout_seconds: int = 60
    max_tokens_budget_per_run: int = 4000
    track_monetary_cost: bool = True


def load_yaml_config(yaml_path: Path) -> dict[str, Any]:
    """Load settings from config/settings.yaml if present."""
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    """Unified application settings combining YAML defaults and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    supervisor: SupervisorSettings = Field(default_factory=SupervisorSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    # Find settings.yaml relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    yaml_path = project_root / "config" / "settings.yaml"
    yaml_data = load_yaml_config(yaml_path)

    # Instantiate Settings, overlaying YAML defaults
    return Settings(
        app=AppSettings(**yaml_data.get("app", {})),
        ingestion=IngestionSettings(**yaml_data.get("ingestion", {})),
        agent=AgentSettings(
            semantic_cache=SemanticCacheSettings(**yaml_data.get("agent", {}).get("semantic_cache", {})),
            **{k: v for k, v in yaml_data.get("agent", {}).items() if k != "semantic_cache"},
        ),
        observability=ObservabilitySettings(**yaml_data.get("observability", {})),
        supervisor=SupervisorSettings(**yaml_data.get("supervisor", {})),
    )
