import re
from typing import List, Optional
from app.schemas import RawCandidate, StructuredIntent
from app.utils.normalizers import clean_name, normalize_location
from app.platforms.contacts import extract_all_contacts

class ProfileExtractor:
    """Refines and validates extracted profile information from discovery candidates."""

    def enrich_candidate(self, candidate: RawCandidate, intent: StructuredIntent) -> RawCandidate:
        # Clean candidate name further
        cleaned = clean_name(candidate.name, candidate.platform)
        if cleaned:
            candidate.name = cleaned

        # Standardize location
        candidate.location = normalize_location(candidate.location)

        # Re-check contacts across title and snippet
        extra_contacts = extract_all_contacts(f"{candidate.title} {candidate.snippet}")
        for email in extra_contacts["emails"]:
            if email not in candidate.emails:
                candidate.emails.append(email)
                candidate.social_links.append(
                    {"platform": "Email", "link": f"mailto:{email}"}
                )

        for phone in extra_contacts["phones"]:
            if phone not in candidate.phones:
                candidate.phones.append(phone)
                candidate.social_links.append(
                    {"platform": "Phone", "link": f"tel:{phone}"}
                )

        for wa in extra_contacts["whatsapp_links"]:
            if wa not in candidate.whatsapp_links:
                candidate.whatsapp_links.append(wa)
                candidate.social_links.append(
                    {"platform": "WhatsApp", "link": wa}
                )

        return candidate

profile_extractor = ProfileExtractor()
