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
from app.utils.normalizers import clean_name
from app.utils.validators import clean_url, is_valid_profile_url

logger = logging.getLogger(__name__)


class WhatsAppAdapter(BasePlatformAdapter):
    platform_name = "WhatsApp"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = self._location_query_term(intent)
        audience = self._audience_query_term(intent)
        return f'{audience} {loc} ("wa.me" OR "api.whatsapp.com/send")'.strip()

    def validate_url(self, url: str) -> bool:
        return is_valid_profile_url("WhatsApp", url)

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
            page_url = clean_url(hit.get("link", ""))
            combined = f"{title} {snippet} {page_url}"

            contacts = extract_all_contacts(combined)
            wa_links = [clean_url(wa) for wa in contacts["whatsapp_links"] if self.validate_url(clean_url(wa))]

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
