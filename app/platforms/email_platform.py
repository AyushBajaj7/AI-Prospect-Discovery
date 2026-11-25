"""
Public Email & Contact adapter.

Discovers publicly published email addresses and contact points from public
web pages / bios matching the target audience without fabrication or scraping
behind authentication walls.
"""
from __future__ import annotations

import logging
from typing import List
from urllib.parse import urlparse

from app.platforms.base import BasePlatformAdapter
from app.platforms.contacts import extract_all_contacts
from app.schemas import RawCandidate, SocialLink, StructuredIntent
from app.services.search_service import search_service
from app.utils.normalizers import clean_name, extract_handle
from app.utils.validators import PLATFORM_PATTERNS, clean_url, is_valid_profile_url

logger = logging.getLogger(__name__)


def _classify_source_platform(url: str) -> str | None:
    """Return the verified platform name for a source URL, or None if invalid non-profile social URL."""
    hostname = (urlparse(url).hostname or "").lower()
    domain_map = {
        "linkedin.com": "LinkedIn",
        "x.com": "X",
        "twitter.com": "X",
        "youtube.com": "YouTube",
        "instagram.com": "Instagram",
        "facebook.com": "Facebook",
        "threads.net": "Threads",
    }
    for dom, plat in domain_map.items():
        if hostname == dom or hostname.endswith("." + dom):
            return plat if is_valid_profile_url(plat, url) else None
    return "Google" if is_valid_profile_url("Google", url) else None


class EmailAdapter(BasePlatformAdapter):
    platform_name = "Email"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = self._location_query_term(intent)
        audience = self._audience_query_term(intent)
        return (
            f'{audience} {loc} ("mailto:" OR "email" OR "@gmail.com" OR "contact") '
            f"-site:wikipedia.org -site:reddit.com -site:quora.com"
        ).strip()

    def validate_url(self, url: str) -> bool:
        return _classify_source_platform(url) is not None

    async def discover(
        self, intent: StructuredIntent, max_results: int = 5
    ) -> List[RawCandidate]:
        query = self.build_query(intent)
        try:
            hits = await search_service.search(query, num_results=max_results)
        except Exception as exc:
            logger.warning("Email discovery search error: %s", type(exc).__name__)
            return []

        candidates: List[RawCandidate] = []

        for hit in hits:
            title = hit.get("title", "")
            snippet = hit.get("snippet", "")
            page_url = clean_url(hit.get("link", ""))

            source_platform = _classify_source_platform(page_url)
            if not source_platform:
                continue

            combined = f"{title} {snippet}"
            contacts = extract_all_contacts(combined)
            emails = contacts["emails"]
            if not emails:
                continue

            name = clean_name(title, source_platform)
            if not name:
                continue

            location = self._infer_location(combined, intent)
            social_links: List[SocialLink] = [
                SocialLink(platform=source_platform, link=page_url),
            ]
            for email in emails:
                social_links.append(SocialLink(platform="Email", link=f"mailto:{email}"))
            for phone in contacts["phones"]:
                social_links.append(SocialLink(platform="Phone", link=f"tel:{phone}"))
            for wa in contacts["whatsapp_links"]:
                cleaned_wa = clean_url(wa)
                if PLATFORM_PATTERNS["WhatsApp"].match(cleaned_wa):
                    social_links.append(SocialLink(platform="WhatsApp", link=cleaned_wa))

            handle = extract_handle(page_url, source_platform)

            candidates.append(
                RawCandidate(
                    platform="Email",
                    name=name,
                    location=location,
                    profile_url=page_url,
                    title=title,
                    snippet=snippet,
                    emails=emails,
                    phones=contacts["phones"],
                    whatsapp_links=contacts["whatsapp_links"],
                    social_links=social_links,
                    metadata={"handle": handle},
                )
            )

        return candidates
