"""
Base platform adapter – common discovery logic shared by all platform adapters.

Each concrete adapter overrides:
  - `platform_name`  (class attribute)
  - `build_query(intent)`  (returns the search query string)
  - Optionally `validate_url(url)` for custom URL logic.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from app.platforms.contacts import extract_all_contacts
from app.schemas import RawCandidate, SocialLink, StructuredIntent
from app.services.search_service import search_service
from app.utils.normalizers import clean_name, extract_handle, normalize_location
from app.utils.validators import clean_url, is_valid_profile_url

logger = logging.getLogger(__name__)


class BasePlatformAdapter(ABC):
    platform_name: str = "Base"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_location(self, text: str, intent: StructuredIntent) -> Optional[str]:
        """
        Infer a human-readable location string for a candidate.

        Prefers text-confirmed information; falls back to the query intent.
        """
        text_lower = (text or "").lower()

        if intent.location.city and intent.location.city.lower() in text_lower:
            parts = [intent.location.city]
            if intent.location.country:
                parts.append(intent.location.country)
            return normalize_location(", ".join(parts))

        if intent.location.country and intent.location.country.lower() in text_lower:
            return normalize_location(intent.location.country)

        # Fall back: use the intent location even if not confirmed in text
        formatted = intent.location.formatted()
        return normalize_location(formatted) if formatted else None

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
        4. Extract clean name, location, and public contacts.
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

            # --- URL validation ---
            profile_url = clean_url(raw_url)
            if not self.validate_url(profile_url):
                continue

            # --- Name extraction ---
            name = clean_name(title, self.platform_name)
            if not name:
                continue

            # --- Location inference ---
            combined_text = f"{title} {snippet}"
            location = self._infer_location(combined_text, intent)

            # --- Contact extraction from snippet ---
            contacts = extract_all_contacts(combined_text)

            # --- Build social links list ---
            social_links: List[SocialLink] = [
                SocialLink(platform=self.platform_name, link=profile_url)
            ]
            for email in contacts["emails"]:
                social_links.append(SocialLink(platform="Email", link=f"mailto:{email}"))
            for phone in contacts["phones"]:
                social_links.append(SocialLink(platform="Phone", link=f"tel:{phone}"))
            for wa in contacts["whatsapp_links"]:
                social_links.append(SocialLink(platform="WhatsApp", link=wa))

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
