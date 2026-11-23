from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, HttpUrl

class ProspectQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Natural-language prospect discovery query, e.g. 'Football lovers in Washington DC in USA'",
        examples=["Startup founders in London UK"]
    )

class LocationDetails(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None

    def formatted(self) -> str:
        parts = [p for p in [self.city, self.state, self.country] if p]
        return ", ".join(parts) if parts else "Unknown"

class StructuredIntent(BaseModel):
    interest: Optional[str] = Field(None, description="Extracted interest, e.g. Football, Cricket, Tech")
    audience: str = Field(..., description="Target audience or profession, e.g. Football lovers, Startup founders")
    location: LocationDetails = Field(default_factory=LocationDetails)
    target_type: Literal["individual", "business"] = Field(
        default="individual",
        description="Type of target prospect: individual or business"
    )
    keywords: List[str] = Field(default_factory=list, description="Supplemental keywords for platform search")

class SocialLink(BaseModel):
    platform: str = Field(
        ...,
        description="Platform name: Google, LinkedIn, X, YouTube, Instagram, WhatsApp, Facebook, Email, Threads, or Phone"
    )
    link: str = Field(
        ...,
        description="Verified public profile URL or contact URI (e.g. mailto:..., tel:..., https://wa.me/...)"
    )

class Prospect(BaseModel):
    name: str = Field(..., description="Prospect's full name or verified business name")
    location: Optional[str] = Field(None, description="Geographical location, e.g. 'London, UK'")
    social_link: List[SocialLink] = Field(
        default_factory=list,
        description="Consolidated verified public social profile and contact links"
    )

class ProspectDiscoveryResponse(BaseModel):
    success: bool = True
    query: str
    total_prospects: int
    prospects: List[Prospect]

class RawCandidate(BaseModel):
    """Internal model for raw prospect candidates collected by platform adapters before deduplication."""
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
    metadata: Dict[str, Any] = Field(default_factory=dict)
