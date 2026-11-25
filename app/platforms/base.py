"""
Base platform adapter – common discovery logic shared by all platform adapters.

Each concrete adapter overrides:
  - `platform_name`  (class attribute)
  - `build_query(intent)`  (returns the search query string)
  - Optionally `validate_url(url)` for custom URL logic.
"""
from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from app.platforms.contacts import extract_all_contacts
from app.schemas import RawCandidate, SearchStatus, SocialLink, StructuredIntent
from app.services.search_service import SearchHit, SearchService, search_service
from app.utils.normalizers import (
    LOCATION_ALIASES,
    clean_name,
    extract_handle,
    location_matches_text,
    normalize_location,
)
from app.utils.validators import (
    PLATFORM_PATTERNS,
    clean_url,
    is_valid_profile_url,
    normalize_whatsapp_url,
    sanitize_search_term,
)

logger = logging.getLogger(__name__)

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

    @staticmethod
    def _location_query_term(intent: StructuredIntent) -> str:
        """Return a sanitized search-query location clause."""
        city = sanitize_search_term(intent.location.city, max_len=60)
        if city:
            return f'"{city}"'
        country = sanitize_search_term(intent.location.country, max_len=60)
        if country:
            return f'"{country}"'
        return ""

    @staticmethod
    def _audience_query_term(intent: StructuredIntent) -> str:
        """
        Return a sanitized, balanced audience search clause capturing
        singular/plural variations without permitting search-operator injection.
        """
        aud = sanitize_search_term(intent.audience, max_len=80)
        interest = sanitize_search_term(intent.interest, max_len=60)
        if not aud:
            return '""'
        words = aud.split()
        if words and words[-1].lower().endswith("s") and len(words[-1]) > 3:
            singular = " ".join(words[:-1] + [words[-1][:-1]])
            if len(words) <= 3:
                return f'("{aud}" OR "{singular}")'
            if interest:
                return f'("{singular}" OR ("{interest}" "{words[-1][:-1]}"))'
            return f'"{singular}"'
        return f'"{aud}"'

    @staticmethod
    def _topic_query_term(intent: StructuredIntent) -> str:
        """Return a sanitized topic/interest term for search queries."""
        topic = sanitize_search_term(intent.interest or intent.audience, max_len=80)
        return f'"{topic}"' if topic else '""'

    async def _execute_search(
        self, query: str, max_results: int
    ) -> Tuple[List[SearchHit], SearchStatus]:
        """
        Execute web search via `search_service` and classify the outcome while
        remaining compatible with tests that monkeypatch `search_service.search`.
        """
        try:
            # Check underlying __func__ rather than __dict__ so pytest monkeypatch
            # restoration of the bound method doesn't shadow search_with_status.
            if getattr(search_service.search, "__func__", None) is not SearchService.search:
                hits = await search_service.search(query, num_results=max_results)
                status = SearchStatus.SUCCESS if hits else SearchStatus.NO_RESULTS
                return hits, status
            outcome = await search_service.search_with_status(query, num_results=max_results)
            return outcome.hits, outcome.status
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("Search timed out on platform %s", self.platform_name)
            return [], SearchStatus.TIMEOUT
        except Exception as exc:
            logger.warning(
                "Search failed on platform %s: %s", self.platform_name, type(exc).__name__
            )
            return [], SearchStatus.PROVIDER_ERROR

    @staticmethod
    def _record_outcome(platform_name: str, status: SearchStatus) -> None:
        from app.services.prospect_service import prospect_service

        prospect_service.record_platform_outcome(platform_name, status)

    def _infer_location(self, text: str, intent: StructuredIntent) -> Optional[str]:
        """
        Extract an evidence-backed location string strictly from `text`.
        Returns None if the snippet/title contains no geographical evidence.
        """
        text_lower = (text or "").lower()
        if not text_lower:
            return None

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

        for city_token, formatted_loc in _EXPLICIT_CITIES:
            if re.search(rf"\b{re.escape(city_token)}\b", text_lower):
                return normalize_location(formatted_loc)

        if country_hit and intent.location.country:
            return normalize_location(intent.location.country)

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

    @abstractmethod
    def build_query(self, intent: StructuredIntent) -> str:
        """Return the search engine query string for this platform."""

    async def discover(
        self, intent: StructuredIntent, max_results: int = 5
    ) -> List[RawCandidate]:
        query = self.build_query(intent)
        hits, search_status = await self._execute_search(query, max_results)
        self._record_outcome(self.platform_name, search_status)

        candidates: List[RawCandidate] = []
        is_verified_social = self.platform_name in PLATFORM_PATTERNS

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
                wa_norm = normalize_whatsapp_url(wa)
                if wa_norm:
                    social_links.append(SocialLink(platform="WhatsApp", link=wa_norm))

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
                    verification_status="verified_profile" if is_verified_social else "public_web",
                    metadata={"handle": handle},
                )
            )

        return candidates
