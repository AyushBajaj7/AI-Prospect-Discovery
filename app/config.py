"""
Application configuration loaded from environment variables / .env file.
All sensitive values (API keys) are Optional[str] with no defaults so they
are never accidentally baked into the image.
"""
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Never display secrets in repr / logs
        secrets_dir=None,
    )

    # ------------------------------------------------------------------ #
    # App metadata
    # ------------------------------------------------------------------ #
    APP_NAME: str = "AI-Powered Multi-Platform Prospect Discovery API"
    APP_VERSION: str = "1.0.0"

    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    PORT: int = Field(default=4000, ge=1, le=65535)
    HOST: str = "0.0.0.0"
    DEBUG: bool = False

    # Allowed CORS origins – defaults to localhost only for safety.
    # Set to a comma-separated list of origins in production, e.g.
    #   CORS_ORIGINS=https://app.example.com,https://admin.example.com
    CORS_ORIGINS: str = "http://localhost:4000,http://127.0.0.1:4000"

    # ------------------------------------------------------------------ #
    # Search providers
    # ------------------------------------------------------------------ #
    # Priority: serper > brave > ddg (free, no key required)
    SEARCH_PROVIDER: str = "auto"
    SERPER_API_KEY: Optional[str] = None
    BRAVE_API_KEY: Optional[str] = None

    # ------------------------------------------------------------------ #
    # LLM providers – auto-detects first available key
    # ------------------------------------------------------------------ #
    LLM_PROVIDER: str = "auto"

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-20241022"

    # ------------------------------------------------------------------ #
    # Discovery pipeline tuning
    # ------------------------------------------------------------------ #
    MAX_RESULTS_PER_PLATFORM: int = Field(default=5, ge=1, le=20)
    SEARCH_TIMEOUT_SECONDS: int = Field(default=15, ge=5, le=60)
    SEARCH_MAX_CONCURRENCY: int = Field(default=4, ge=1, le=10)
    # Minimum 0–100 score to include a prospect in the final results
    RELEVANCE_THRESHOLD: float = Field(default=35.0, ge=0.0, le=100.0)

    # ------------------------------------------------------------------ #
    # In-process cache
    # ------------------------------------------------------------------ #
    # TTL in seconds for query-intent & search caches (default: 1 hour)
    CACHE_TTL_SECONDS: int = Field(default=3600, ge=60)
    CACHE_MAX_SIZE: int = Field(default=1000, ge=50, le=50000)

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("SEARCH_PROVIDER")
    @classmethod
    def validate_search_provider(cls, v: str) -> str:
        allowed = {"auto", "serper", "brave", "ddg"}
        if v.lower() not in allowed:
            raise ValueError(f"SEARCH_PROVIDER must be one of {allowed}")
        return v.lower()

    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        allowed = {"auto", "gemini", "openai", "anthropic", "fallback"}
        if v.lower() not in allowed:
            raise ValueError(f"LLM_PROVIDER must be one of {allowed}")
        return v.lower()

    def allowed_origins(self) -> List[str]:
        """Returns the parsed list of allowed CORS origins."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
