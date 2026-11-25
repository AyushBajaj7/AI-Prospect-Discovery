"""
Public Email & Contact adapter.

Discovers publicly published email addresses and contact points from public
web pages / bios matching the target audience without fabrication.
"""
from __future__ import annotations

from typing import List, Optional
from urllib.parse import urlparse

from app.platforms.base import BasePlatformAdapter
from app.platforms.contacts import extract_all_contacts
from app.schemas import RawCandidate, SocialLink, StructuredIntent
from app.utils.normalizers import clean_name, extract_handle
from app.utils.validators import clean_url, is_valid_profile_url, normalize_whatsapp_url


def _classify_source_platform(url: str) -> Optional[str]:
    """Return the verified platform name for a source URL, or None if invalid."""
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
        hits, search_status = await self._execute_search(query, max_results)
        self._record_outcome(self.platform_name, search_status)

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
                wa_norm = normalize_whatsapp_url(wa)
                if wa_norm:
                    social_links.append(SocialLink(platform="WhatsApp", link=wa_norm))

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
                    verification_status=(
                        "verified_profile" if source_platform != "Google" else "public_web"
                    ),
                    metadata={"handle": handle},
                )
            )

        return candidates
