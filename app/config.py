from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "AI-Powered Multi-Platform Prospect Discovery API"
    APP_VERSION: str = "1.0.0"
    PORT: int = Field(default=4000, description="Server port, default 4000 per assessment spec")
    HOST: str = "0.0.0.0"
    DEBUG: bool = False

    # Search Provider: 'serper', 'brave', or 'ddg'
    SEARCH_PROVIDER: str = "auto"
    SERPER_API_KEY: Optional[str] = None
    BRAVE_API_KEY: Optional[str] = None

    # LLM Provider: 'openai', 'gemini', 'anthropic', or 'auto'
    LLM_PROVIDER: str = "auto"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-20241022"

    # Search and Discovery Limits
    MAX_RESULTS_PER_PLATFORM: int = 5
    SEARCH_TIMEOUT_SECONDS: int = 15
    RELEVANCE_THRESHOLD: int = 40  # Minimum score (out of 100) to qualify as a relevant prospect

    # Cache Settings
    CACHE_TTL_SECONDS: int = 3600  # 1 hour in-memory cache for query parsing and discovery

settings = Settings()
