"""
Profile extractor / enricher.

After the base adapter produces a RawCandidate, this service runs a second
pass to:
  - Re-clean the candidate name (in case the adapter missed a trailing suffix).
  - Normalise the location string.
  - Re-scan title + snippet for contact details that may have been missed.
  - De-duplicate social links within a single candidate.
"""
from __future__ import annotations

import logging
from typing import Set

from app.platforms.contacts import extract_all_contacts
from app.schemas import RawCandidate, SocialLink, StructuredIntent
from app.utils.normalizers import clean_name, classify_candidate_entity, normalize_location
from app.utils.validators import format_tel_uri

logger = logging.getLogger(__name__)


class ProfileExtractor:

    def enrich_candidate(
        self, candidate: RawCandidate, intent: StructuredIntent
    ) -> RawCandidate:
        # 1. Re-clean name
        better_name = clean_name(candidate.name, candidate.platform)
        if better_name:
            candidate.name = better_name

        # 2. Normalise location
        candidate.location = normalize_location(candidate.location)

        # 3. Re-scan for contacts in title + snippet
        extra = extract_all_contacts(f"{candidate.title} {candidate.snippet}")

        for email in extra["emails"]:
            if email not in candidate.emails:
                candidate.emails.append(email)
                candidate.social_links.append(
                    SocialLink(platform="Email", link=f"mailto:{email}")
                )

        for phone in extra["phones"]:
            if phone not in candidate.phones:
                candidate.phones.append(phone)
                tel_uri = format_tel_uri(phone)
                if tel_uri:
                    candidate.social_links.append(
                        SocialLink(platform="Phone", link=tel_uri)
                    )

        # Re-classify entity type after cleaning name
        candidate.entity_type = classify_candidate_entity(
            candidate.name,
            candidate.platform,
            candidate.profile_url,
            candidate.title,
            candidate.snippet,
        )

        for wa in extra["whatsapp_links"]:
            if wa not in candidate.whatsapp_links:
                candidate.whatsapp_links.append(wa)
                candidate.social_links.append(
                    SocialLink(platform="WhatsApp", link=wa)
                )

        # 4. De-duplicate social links within this single candidate
        seen: Set[str] = set()
        unique_links = []
        for sl in candidate.social_links:
            key = sl.link.lower().strip()
            if key not in seen:
                seen.add(key)
                unique_links.append(sl)
        candidate.social_links = unique_links

        return candidate


profile_extractor = ProfileExtractor()
