"""
Base platform adapter – common discovery logic shared by all platform adapters.

Each concrete adapter overrides:
  - `platform_name`  (class attribute)
  - `build_query(intent)`  (returns the search query string)
  - Optionally `validate_url(url)` for custom URL logic.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from app.platforms.contacts import extract_all_contacts
from app.schemas import RawCandidate, SocialLink, StructuredIntent
from app.services.search_service import search_service
from app.utils.normalizers import (
    LOCATION_ALIASES,
    clean_name,
    extract_handle,
    location_matches_text,
    normalize_location,
)
from app.utils.validators import clean_url, is_valid_profile_url

logger = logging.getLogger(__name__)

# Compact reverse lookup for detecting explicit non-query cities/countries in text
_EXPLICIT_CITIES: Tuple[Tuple[str, str], ...] = (
    ("washington dc", "Washington DC, USA"),
    ("washington, d.c.", "Washington DC, USA"),
    ("new york city", "New York, USA"),
    ("new york", "New York, USA"),
    ("san francisco", "San Francisco, USA"),
    ("los angeles", "Los Angeles, USA"),
    ("chicago", "Chicago, USA"),
    ("toronto", "Toronto, Canada"),
    ("london", "London, UK"),
    ("manchester", "Manchester, UK"),
    ("berlin", "Berlin, Germany"),
    ("paris", "Paris, France"),
    ("ahmedabad", "Ahmedabad, India"),
    ("bengaluru", "Bengaluru, India"),
    ("bangalore", "Bengaluru, India"),
    ("mumbai", "Mumbai, India"),
    ("delhi", "Delhi, India"),
    ("dubai", "Dubai, UAE"),
    ("abu dhabi", "Abu Dhabi, UAE"),
    ("singapore", "Singapore"),
    ("sydney", "Sydney, Australia"),
    ("melbourne", "Melbourne, Australia"),
    ("tokyo", "Tokyo, Japan"),
)


class BasePlatformAdapter(ABC):
    platform_name: str = "Base"

    # ------------------------------------------------------------------
    # Shared helpers for query building & evidence-based extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _location_query_term(intent: StructuredIntent) -> str:
        """Return a clean search-query location clause."""
        if intent.location.city:
            return f'"{intent.location.city}"'
        if intent.location.country:
            return f'"{intent.location.country}"'
        return ""

    @staticmethod
    def _audience_query_term(intent: StructuredIntent) -> str:
        """
        Return a balanced audience search clause that captures singular/plural
        variations (e.g. "Startup founders" OR "Startup founder") rather than
        over-constraining SERPs to plural strings rarely written in bios.
        """
        aud = intent.audience.strip()
        words = aud.split()
        # Singularize trailing plural role word ("founders" -> "founder")
        if words and words[-1].lower().endswith("s") and len(words[-1]) > 3:
            singular = " ".join(words[:-1] + [words[-1][:-1]])
            if len(words) <= 3:
                return f'("{aud}" OR "{singular}")'
            if intent.interest:
                return f'("{singular}" OR ("{intent.interest}" "{words[-1][:-1]}"))'
            return f'"{singular}"'
        return f'"{aud}"'

    def _infer_location(self, text: str, intent: StructuredIntent) -> Optional[str]:
        """
        Extract an evidence-backed location string strictly from `text`.

        Never fabricates or blindly copies the search query's location onto a
        candidate if the title/snippet contains zero geographical evidence.
        """
        text_lower = (text or "").lower()
        if not text_lower:
            return None

        # 1. Check whether the text confirms the requested query city/country
        city_hit = bool(
            intent.location.city and location_matches_text(intent.location.city, text_lower)
        )
        country_hit = bool(
            intent.location.country and location_matches_text(intent.location.country, text_lower)
        )

        if city_hit:
            parts = [intent.location.city]
            if intent.location.state and location_matches_text(intent.location.state, text_lower):
                parts.append(intent.location.state)
            if intent.location.country:
                parts.append(intent.location.country)
            return normalize_location(", ".join(p for p in parts if p))

        # 2. Check if text explicitly states another known city (avoid misattribution)
        for city_token, formatted_loc in _EXPLICIT_CITIES:
            if re.search(rf"\b{re.escape(city_token)}\b", text_lower):
                return normalize_location(formatted_loc)

        # 3. Confirmed country match alone
        if country_hit and intent.location.country:
            return normalize_location(intent.location.country)

        # 4. Other country alias evidence
        for canonical, aliases in (
            ("UK", LOCATION_ALIASES["uk"]),
            ("USA", LOCATION_ALIASES["usa"]),
            ("UAE", LOCATION_ALIASES["uae"]),
        ):
            for alias in aliases:
                pattern = (
                    rf"(?<!\w){re.escape(alias)}(?!\w)"
                    if len(alias) <= 4
                    else rf"\b{re.escape(alias)}\b"
                )
                if re.search(pattern, text_lower):
                    return canonical

        return None

    def validate_url(self, url: str) -> bool:
        return is_valid_profile_url(self.platform_name, url)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_query(self, intent: StructuredIntent) -> str:
        """Return the search engine query string for this platform."""

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def discover(
        self, intent: StructuredIntent, max_results: int = 5
    ) -> List[RawCandidate]:
        """
        1. Build platform-specific search query.
        2. Execute search (via search_service).
        3. Validate URLs.
        4. Extract clean name, verified location, and public contacts.
        5. Return list of RawCandidate objects.
        """
        query = self.build_query(intent)
        try:
            hits = await search_service.search(query, num_results=max_results)
        except Exception as exc:
            logger.warning(
                "Search failed on platform %s: %s", self.platform_name, type(exc).__name__
            )
            return []

        candidates: List[RawCandidate] = []

        for hit in hits:
            raw_url = hit.get("link", "")
            title = hit.get("title", "")
            snippet = hit.get("snippet", "")

            profile_url = clean_url(raw_url)
            if not self.validate_url(profile_url):
                continue

            name = clean_name(title, self.platform_name)
            if not name:
                continue

            combined_text = f"{title} {snippet}"
            location = self._infer_location(combined_text, intent)
            contacts = extract_all_contacts(combined_text)

            social_links: List[SocialLink] = [
                SocialLink(platform=self.platform_name, link=profile_url)
            ]
            for email in contacts["emails"]:
                social_links.append(SocialLink(platform="Email", link=f"mailto:{email}"))
            for phone in contacts["phones"]:
                social_links.append(SocialLink(platform="Phone", link=f"tel:{phone}"))
            for wa in contacts["whatsapp_links"]:
                social_links.append(SocialLink(platform="WhatsApp", link=clean_url(wa)))

            handle = extract_handle(profile_url, self.platform_name)

            candidates.append(
                RawCandidate(
                    platform=self.platform_name,
                    name=name,
                    location=location,
                    profile_url=profile_url,
                    title=title,
                    snippet=snippet,
                    emails=contacts["emails"],
                    phones=contacts["phones"],
                    whatsapp_links=contacts["whatsapp_links"],
                    social_links=social_links,
                    metadata={"handle": handle},
                )
            )

        return candidates
