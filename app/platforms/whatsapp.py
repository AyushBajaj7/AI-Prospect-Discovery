import re
from typing import List
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent, RawCandidate, SocialLink
from app.utils.normalizers import clean_name
from app.services.search_service import search_service
from app.platforms.contacts import extract_all_contacts

class WhatsAppAdapter(BasePlatformAdapter):
    platform_name: str = "WhatsApp"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = f'"{intent.location.city}"' if intent.location.city else (f'"{intent.location.country}"' if intent.location.country else "")
        return f'"{intent.audience}" {loc} ("wa.me" OR "api.whatsapp.com")'.strip()

    def validate_url(self, url: str) -> bool:
        return "wa.me" in url or "api.whatsapp.com" in url or "chat.whatsapp.com" in url

    async def discover(self, intent: StructuredIntent, max_results: int = 5) -> List[RawCandidate]:
        """Discovers WhatsApp contact links mentioned in search results."""
        query = self.build_query(intent)
        try:
            results = await search_service.search(query, num_results=max_results)
        except Exception:
            return []

        candidates: List[RawCandidate] = []
        for item in results:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            combined = f"{title} {snippet} {link}"
            contacts = extract_all_contacts(combined)
            
            # If the search result URL itself is a wa.me link or if snippet contains a wa.me link
            wa_links = contacts.get("whatsapp_links", [])
            if self.validate_url(link) and link not in wa_links:
                wa_links.append(link)

            if not wa_links:
                continue

            name = clean_name(title, self.platform_name)
            if not name or len(name) < 2:
                continue

            location = self.extract_location(combined, intent)
            social_links = [SocialLink(platform="WhatsApp", link=w) for w in wa_links]
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
                    social_links=social_links
                )
            )

        return candidates
