import re
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from app.schemas import StructuredIntent, RawCandidate, SocialLink
from app.services.search_service import search_service
from app.utils.validators import is_valid_profile_url, clean_url
from app.utils.normalizers import clean_name, extract_handle
from app.platforms.contacts import extract_all_contacts

logger = logging.getLogger(__name__)

class BasePlatformAdapter(ABC):
    platform_name: str = "Base"

    def __init__(self):
        pass

    def extract_location(self, text: str, intent: StructuredIntent) -> Optional[str]:
        """Infers candidate location from text and intent context."""
        if not text:
            return intent.location.formatted() if (intent.location.city or intent.location.country) else None

        text_lower = text.lower()
        if intent.location.city and intent.location.city.lower() in text_lower:
            parts = [intent.location.city]
            if intent.location.country:
                parts.append(intent.location.country)
            return ", ".join(parts)
        elif intent.location.country and intent.location.country.lower() in text_lower:
            return intent.location.country
        
        # Fallback to search query location
        if intent.location.city or intent.location.country:
            return intent.location.formatted()
        return None

    @abstractmethod
    def build_query(self, intent: StructuredIntent) -> str:
        """Constructs platform-specific search query."""
        pass

    def validate_url(self, url: str) -> bool:
        """Validates if the URL belongs to this platform and is a profile page."""
        return is_valid_profile_url(self.platform_name, url)

    async def discover(self, intent: StructuredIntent, max_results: int = 5) -> List[RawCandidate]:
        """Discovers prospects on this platform using search and extraction."""
        query = self.build_query(intent)
        try:
            results = await search_service.search(query, num_results=max_results)
        except Exception as e:
            logger.warning(f"Failed search on {self.platform_name}: {e}")
            return []

        candidates: List[RawCandidate] = []
        for item in results:
            raw_url = item.get("link", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            
            c_url = clean_url(raw_url)
            if not self.validate_url(c_url):
                continue

            name = clean_name(title, self.platform_name)
            if not name or len(name) < 2:
                continue

            combined_text = f"{title} {snippet}"
            location = self.extract_location(combined_text, intent)
            contacts = extract_all_contacts(combined_text)
            handle = extract_handle(c_url, self.platform_name)

            social_links = [SocialLink(platform=self.platform_name, link=c_url)]
            for email in contacts["emails"]:
                social_links.append(SocialLink(platform="Email", link=f"mailto:{email}"))
            for phone in contacts["phones"]:
                social_links.append(SocialLink(platform="Phone", link=f"tel:{phone}"))
            for wa in contacts["whatsapp_links"]:
                social_links.append(SocialLink(platform="WhatsApp", link=wa))

            candidates.append(
                RawCandidate(
                    platform=self.platform_name,
                    name=name,
                    location=location,
                    profile_url=c_url,
                    title=title,
                    snippet=snippet,
                    emails=contacts["emails"],
                    phones=contacts["phones"],
                    whatsapp_links=contacts["whatsapp_links"],
                    social_links=social_links,
                    metadata={"handle": handle}
                )
            )

        return candidates
