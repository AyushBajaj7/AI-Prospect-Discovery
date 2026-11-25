"""
WhatsApp public contact-link adapter.

Discovers publicly published wa.me / send links associated with matching
prospects and normalizes links to official digit-only `https://wa.me/<digits>`.
"""
from __future__ import annotations

from typing import List

from app.platforms.base import BasePlatformAdapter
from app.platforms.contacts import extract_all_contacts
from app.schemas import RawCandidate, SocialLink, StructuredIntent
from app.utils.normalizers import clean_name, classify_candidate_entity
from app.utils.validators import format_tel_uri, is_valid_profile_url, normalize_whatsapp_url


class WhatsAppAdapter(BasePlatformAdapter):
    platform_name = "WhatsApp"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = self._location_query_term(intent)
        audience = self._audience_query_term(intent)
        return f'{audience} {loc} ("wa.me" OR "api.whatsapp.com/send")'.strip()

    def validate_url(self, url: str) -> bool:
        norm = normalize_whatsapp_url(url)
        return bool(norm and is_valid_profile_url("WhatsApp", norm))

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
            page_url = hit.get("link", "")
            combined = f"{title} {snippet} {page_url}"

            contacts = extract_all_contacts(combined)
            wa_links = list(contacts["whatsapp_links"])

            page_wa = normalize_whatsapp_url(page_url)
            if page_wa and is_valid_profile_url("WhatsApp", page_wa) and page_wa not in wa_links:
                wa_links.append(page_wa)

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
                tel_uri = format_tel_uri(phone)
                if tel_uri:
                    social_links.append(SocialLink(platform="Phone", link=tel_uri))

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
                    verification_status="url_format_validated",
                    entity_type=classify_candidate_entity(
                        name, "WhatsApp", wa_links[0], title, snippet
                    ),
                )
            )

        return candidates
