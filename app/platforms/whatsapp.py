"""
WhatsApp public contact-link adapter.

Discovers publicly published wa.me links associated with individuals or
businesses that match the query. No private data is accessed.
"""
from __future__ import annotations

import logging
from typing import List

from app.platforms.base import BasePlatformAdapter
from app.platforms.contacts import extract_all_contacts
from app.schemas import RawCandidate, SocialLink, StructuredIntent
from app.services.search_service import search_service
from app.utils.normalizers import clean_name, normalize_location

logger = logging.getLogger(__name__)


class WhatsAppAdapter(BasePlatformAdapter):
    platform_name = "WhatsApp"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = ""
        if intent.location.city:
            loc = f'"{intent.location.city}"'
        elif intent.location.country:
            loc = f'"{intent.location.country}"'

        audience = f'"{intent.audience}"'
        return f'{audience} {loc} ("wa.me" OR "api.whatsapp.com/send")'.strip()

    def validate_url(self, url: str) -> bool:
        return bool(url) and (
            "wa.me/" in url.lower()
            or "api.whatsapp.com/send" in url.lower()
            or "chat.whatsapp.com/" in url.lower()
        )

    async def discover(
        self, intent: StructuredIntent, max_results: int = 5
    ) -> List[RawCandidate]:
        query = self.build_query(intent)
        try:
            hits = await search_service.search(query, num_results=max_results)
        except Exception as exc:
            logger.warning("WhatsApp search error: %s", type(exc).__name__)
            return []

        candidates: List[RawCandidate] = []

        for hit in hits:
            title = hit.get("title", "")
            snippet = hit.get("snippet", "")
            page_url = hit.get("link", "")
            combined = f"{title} {snippet} {page_url}"

            contacts = extract_all_contacts(combined)
            wa_links = contacts["whatsapp_links"]

            # Also check if the result URL itself is a wa.me link
            if self.validate_url(page_url) and page_url not in wa_links:
                wa_links.append(page_url)

            if not wa_links:
                continue

            name = clean_name(title, self.platform_name)
            if not name:
                continue

            location = self._infer_location(combined, intent)
            social_links: List[SocialLink] = [
                SocialLink(platform="WhatsApp", link=wa) for wa in wa_links
            ]
            for email in contacts["emails"]:
                social_links.append(SocialLink(platform="Email", link=f"mailto:{email}"))
            for phone in contacts["phones"]:
                social_links.append(SocialLink(platform="Phone", link=f"tel:{phone}"))

            candidates.append(
                RawCandidate(
                    platform="WhatsApp",
                    name=name,
                    location=location,
                    profile_url=wa_links[0],
                    title=title,
                    snippet=snippet,
                    emails=contacts["emails"],
                    phones=contacts["phones"],
                    whatsapp_links=wa_links,
                    social_links=social_links,
                )
            )

        return candidates
