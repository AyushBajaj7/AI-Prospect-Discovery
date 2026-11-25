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
from urllib.parse import urlparse

from app.platforms.contacts import extract_all_contacts
from app.schemas import (
    PlatformPipelineMetrics,
    RawCandidate,
    SearchStatus,
    SocialLink,
    StructuredIntent,
)
from app.services.search_service import SearchHit, SearchService, search_service
from app.utils.normalizers import (
    LOCATION_ALIASES,
    clean_name,
    classify_candidate_entity,
    extract_handle,
    location_matches_text,
    normalize_location,
)
from app.utils.validators import (
    PLATFORM_PATTERNS,
    clean_url,
    format_tel_uri,
    is_valid_profile_url,
    normalize_whatsapp_url,
    sanitize_search_term,
)

logger = logging.getLogger(__name__)

# Unambiguous city tokens used strictly to extract city names confirmed in text;
# country is appended ONLY when country text is also present.
_EXPLICIT_CITY_NAMES: Tuple[Tuple[str, str], ...] = (
    ("washington dc", "Washington DC"),
    ("washington, d.c.", "Washington DC"),
    ("new york city", "New York"),
    ("new york", "New York"),
    ("san francisco", "San Francisco"),
    ("los angeles", "Los Angeles"),
    ("chicago", "Chicago"),
    ("toronto", "Toronto"),
    ("london", "London"),
    ("manchester", "Manchester"),
    ("berlin", "Berlin"),
    ("paris", "Paris"),
    ("ahmedabad", "Ahmedabad"),
    ("bengaluru", "Bengaluru"),
    ("bangalore", "Bengaluru"),
    ("mumbai", "Mumbai"),
    ("delhi", "Delhi"),
    ("dubai", "Dubai"),
    ("abu dhabi", "Abu Dhabi"),
    ("singapore", "Singapore"),
    ("sydney", "Sydney"),
    ("melbourne", "Melbourne"),
    ("tokyo", "Tokyo"),
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
    def _record_metrics(platform_name: str, metrics: PlatformPipelineMetrics) -> None:
        from app.services.prospect_service import prospect_service

        prospect_service.record_platform_metrics(platform_name, metrics)

    def _infer_location(self, text: str, intent: StructuredIntent) -> Optional[str]:
        """
        Strictly evidence-only location extractor: includes city, state, and
        country ONLY when each component itself is confirmed in `text`.
        """
        text_lower = (text or "").lower()
        if not text_lower:
            return None

        matched_city: Optional[str] = None
        matched_state: Optional[str] = None
        matched_country: Optional[str] = None

        if intent.location.city and location_matches_text(intent.location.city, text_lower):
            matched_city = intent.location.city
        else:
            for city_token, city_label in _EXPLICIT_CITY_NAMES:
                if re.search(rf"\b{re.escape(city_token)}\b", text_lower):
                    matched_city = city_label
                    break

        if intent.location.state and location_matches_text(intent.location.state, text_lower):
            matched_state = intent.location.state

        if intent.location.country and location_matches_text(intent.location.country, text_lower):
            matched_country = intent.location.country
        else:
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
                        matched_country = canonical
                        break
                if matched_country:
                    break

        parts = [p for p in (matched_city, matched_state, matched_country) if p]
        return normalize_location(", ".join(parts)) if parts else None

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
        metrics = PlatformPipelineMetrics(
            search_status=search_status,
            search_hits=len(hits),
        )

        candidates: List[RawCandidate] = []
        is_verified_social = self.platform_name in PLATFORM_PATTERNS

        for hit in hits:
            raw_url = hit.get("link", "")
            title = hit.get("title", "")
            snippet = hit.get("snippet", "")

            profile_url = clean_url(raw_url)
            if not self.validate_url(profile_url):
                metrics.invalid_urls += 1
                logger.info(
                    "Platform %s rejected hit [reason=invalid_url] url=%r",
                    self.platform_name,
                    raw_url[:120],
                )
                continue

            name = clean_name(title, self.platform_name)
            if not name:
                metrics.rejected_names += 1
                logger.info(
                    "Platform %s rejected hit [reason=rejected_name] title=%r",
                    self.platform_name,
                    title[:120],
                )
                continue

            combined_text = f"{title} {snippet}"
            location = self._infer_location(combined_text, intent)
            contacts = extract_all_contacts(combined_text)
            phone_ctx = (urlparse(profile_url).hostname or "").lower() or None

            social_links: List[SocialLink] = [
                SocialLink(platform=self.platform_name, link=profile_url)
            ]
            for email in contacts["emails"]:
                social_links.append(SocialLink(platform="Email", link=f"mailto:{email}"))
            for phone in contacts["phones"]:
                tel_uri = format_tel_uri(phone, phone_context=phone_ctx)
                if tel_uri:
                    social_links.append(SocialLink(platform="Phone", link=tel_uri))
            for wa in contacts["whatsapp_links"]:
                wa_norm = normalize_whatsapp_url(wa)
                if wa_norm:
                    social_links.append(SocialLink(platform="WhatsApp", link=wa_norm))

            handle = extract_handle(profile_url, self.platform_name)
            entity_type = classify_candidate_entity(
                name, self.platform_name, profile_url, title, snippet
            )

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
                    verification_status=(
                        "url_format_validated" if is_verified_social else "search_indexed_web"
                    ),
                    entity_type=entity_type,
                    metadata={"handle": handle},
                )
            )

        self._record_metrics(self.platform_name, metrics)
        return candidates
