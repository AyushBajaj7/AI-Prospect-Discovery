"""
Pydantic v2 request / response schemas.

All public-facing models are strictly typed and have descriptive Field metadata
so that the auto-generated OpenAPI documentation is accurate and complete.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SearchStatus(str, Enum):
    """Explicit outcome classification for each upstream platform search call."""

    SUCCESS = "SUCCESS"
    NO_RESULTS = "NO_RESULTS"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    PROVIDER_ERROR = "PROVIDER_ERROR"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ProspectQueryRequest(BaseModel):
    """Body accepted by POST /prospect-discovery."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description=(
            "Natural-language prospect discovery query, "
            "e.g. 'Football lovers in Washington DC in USA'."
        ),
        examples=["Startup founders in London UK"],
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Strip control chars, collapse internal whitespace, require alpha chars."""
        v = re.sub(r"[\x00-\x1f\x7f]+", " ", v).strip()
        v = re.sub(r"\s+", " ", v)
        if not re.search(r"[a-zA-Z]", v):
            raise ValueError("Query must contain at least one letter.")
        return v


# ---------------------------------------------------------------------------
# Internal intent model (produced by LLM / heuristic parser)
# ---------------------------------------------------------------------------

class LocationDetails(BaseModel):
    city: Optional[str] = Field(default=None, max_length=80)
    state: Optional[str] = Field(default=None, max_length=80)
    country: Optional[str] = Field(default=None, max_length=80)

    def formatted(self) -> str:
        """Human-readable location string; returns empty string when unset."""
        parts = [p for p in [self.city, self.state, self.country] if p]
        return ", ".join(parts)


class StructuredIntent(BaseModel):
    interest: Optional[str] = Field(
        None,
        max_length=100,
        description="Primary topic or interest, e.g. Football, Cricket, Startups.",
    )
    audience: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Target audience / profession, e.g. 'Football lovers'.",
    )
    location: LocationDetails = Field(default_factory=LocationDetails)
    target_type: Literal["individual", "business"] = Field(
        default="individual",
        description="Whether targeting individual persons/owners or organization entities.",
    )
    keywords: List[str] = Field(
        default_factory=list,
        max_length=12,
        description="Supplemental keywords for platform-level search queries.",
    )


# ---------------------------------------------------------------------------
# Public response models
# ---------------------------------------------------------------------------

class SocialLink(BaseModel):
    """A single verified public social-media profile or contact link."""

    platform: str = Field(
        ...,
        description=(
            "Platform or contact channel: LinkedIn, X, YouTube, Instagram, Facebook, "
            "Threads, WhatsApp, Email, Phone, or Google."
        ),
    )
    link: str = Field(
        ...,
        description=(
            "Verified public URL or contact URI "
            "(e.g. https://linkedin.com/in/xyz, mailto:a@b.com, "
            "tel:+1234567890, https://wa.me/1234567890)."
        ),
    )


class Prospect(BaseModel):
    """A single consolidated prospect record."""

    name: str = Field(
        ...,
        description="Prospect's full name or verified business name.",
    )
    location: Optional[str] = Field(
        None,
        description="Geographical location string when confirmed in evidence, e.g. 'London, UK'.",
    )
    social_link: List[SocialLink] = Field(
        default_factory=list,
        description="All consolidated public social profiles and contact links.",
    )


class ProviderHealthSummary(BaseModel):
    """Observability summary distinguishing empty results from upstream errors."""

    llm_provider: str = "heuristic_fallback"
    platforms_succeeded: int = 0
    platforms_no_results: int = 0
    platforms_rate_limited: int = 0
    platforms_timed_out: int = 0
    platforms_errored: int = 0
    platform_outcomes: Dict[str, SearchStatus] = Field(default_factory=dict)


class ProspectDiscoveryResponse(BaseModel):
    """Successful response returned by POST /prospect-discovery."""

    success: bool = True
    query: str = Field(..., description="The original query as submitted.")
    total_prospects: int = Field(
        ...,
        ge=0,
        description="Total number of unique prospect records returned.",
    )
    prospects: List[Prospect] = Field(
        default_factory=list,
        description="Ordered list of discovered prospect records.",
    )
    provider_health: Optional[ProviderHealthSummary] = Field(
        default=None,
        description="Per-platform search status and LLM provider telemetry.",
    )


# ---------------------------------------------------------------------------
# Internal pipeline model (never serialised to the API caller)
# ---------------------------------------------------------------------------

class RawCandidate(BaseModel):
    """
    Intermediate model holding a single discovery hit from one platform adapter
    before deduplication and relevance filtering.
    """

    platform: str
    name: str
    location: Optional[str] = None
    profile_url: str
    title: str = ""
    snippet: str = ""
    emails: List[str] = Field(default_factory=list)
    phones: List[str] = Field(default_factory=list)
    whatsapp_links: List[str] = Field(default_factory=list)
    social_links: List[SocialLink] = Field(default_factory=list)
    relevance_score: float = 0.0
    verification_status: Literal["verified_profile", "public_web", "partial"] = "public_web"
    metadata: Dict[str, Any] = Field(default_factory=dict)
