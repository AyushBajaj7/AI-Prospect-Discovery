"""
Pydantic v2 request / response schemas.

All public-facing models are strictly typed and have descriptive Field metadata
so that the auto-generated OpenAPI documentation is accurate and complete.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


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
        """Strip leading/trailing whitespace and collapse internal whitespace."""
        v = v.strip()
        # Collapse multiple spaces / tabs / newlines to a single space
        v = re.sub(r"\s+", " ", v)
        # Reject queries that are purely punctuation or numbers
        if not re.search(r"[a-zA-Z]", v):
            raise ValueError("Query must contain at least one letter.")
        return v


# ---------------------------------------------------------------------------
# Internal intent model (produced by LLM / heuristic parser)
# ---------------------------------------------------------------------------

class LocationDetails(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None

    def formatted(self) -> str:
        """Human-readable location string; returns None-safe empty string."""
        parts = [p for p in [self.city, self.state, self.country] if p]
        return ", ".join(parts)


class StructuredIntent(BaseModel):
    interest: Optional[str] = Field(
        None,
        description="Primary topic or interest, e.g. Football, Cricket, Tech.",
    )
    audience: str = Field(
        ...,
        description="Target audience / profession, e.g. 'Football lovers'.",
    )
    location: LocationDetails = Field(default_factory=LocationDetails)
    target_type: Literal["individual", "business"] = Field(
        default="individual",
        description="Whether the target is an individual person or a business entity.",
    )
    keywords: List[str] = Field(
        default_factory=list,
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
            "Platform name: LinkedIn, X, YouTube, Instagram, Facebook, "
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
        description="Geographical location string, e.g. 'London, UK'.",
    )
    social_link: List[SocialLink] = Field(
        default_factory=list,
        description="All consolidated public social profiles and contact links.",
    )


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
    # Arbitrary per-adapter metadata (e.g. extracted handle, score breakdown)
    metadata: Dict[str, Any] = Field(default_factory=dict)
