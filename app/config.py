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
        secrets_dir=None,
    )

    # ------------------------------------------------------------------ #
    # App metadata
    # ------------------------------------------------------------------ #
    APP_NAME: str = "AI-Powered Multi-Platform Prospect Discovery API"
    APP_VERSION: str = "1.1.0"

    # ------------------------------------------------------------------ #
    # Server & Security
    # ------------------------------------------------------------------ #
    PORT: int = Field(default=4000, ge=1, le=65535)
    HOST: str = "0.0.0.0"
    DEBUG: bool = False

    # Allowed CORS origins – explicit origins only (never wildcard with credentials).
    CORS_ORIGINS: str = "http://localhost:4000,http://127.0.0.1:4000"

    # Per-IP sliding-window endpoint rate limiting (process-local; back with
    # Redis or API gateway rate-limiting in multi-worker deployments).
    RATE_LIMIT_REQUESTS: int = Field(default=30, ge=1, le=10000)
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60, ge=1, le=3600)
    RATE_LIMIT_MAX_CLIENTS: int = Field(default=10000, ge=100, le=500000)
    # Comma-separated trusted proxy IPs or CIDRs allowed to set X-Forwarded-For.
    # Empty (default) disables trusting X-Forwarded-For headers from callers.
    TRUSTED_PROXIES: str = ""

    # ------------------------------------------------------------------ #
    # Search providers
    # ------------------------------------------------------------------ #
    # Priority: serper -> brave -> ddg (free fallback, no key required)
    SEARCH_PROVIDER: str = "auto"
    SERPER_API_KEY: Optional[str] = None
    BRAVE_API_KEY: Optional[str] = None

    # ------------------------------------------------------------------ #
    # LLM providers – auto-detects and falls back across configured keys
    # ------------------------------------------------------------------ #
    LLM_PROVIDER: str = "auto"

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-6-luna"

    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-haiku-5-5"

    # ------------------------------------------------------------------ #
    # Discovery pipeline tuning
    # ------------------------------------------------------------------ #
    MAX_RESULTS_PER_PLATFORM: int = Field(default=5, ge=1, le=20)
    SEARCH_TIMEOUT_SECONDS: int = Field(default=15, ge=5, le=60)
    SEARCH_MAX_CONCURRENCY: int = Field(default=4, ge=1, le=10)
    # Minimum 0–100 score required to include a prospect in final results
    RELEVANCE_THRESHOLD: float = Field(default=35.0, ge=0.0, le=100.0)

    # ------------------------------------------------------------------ #
    # Bounded in-process TTL caches
    # ------------------------------------------------------------------ #
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

    def trusted_proxies(self) -> List[str]:
        """Returns the parsed list of trusted proxy IP/CIDR entries."""
        return [p.strip() for p in self.TRUSTED_PROXIES.split(",") if p.strip()]


settings = Settings()
