"""
Profile extractor / enricher.

Runs a second extraction and entity-classification pass after initial adapter
discovery:
  - Re-cleans candidate names
  - Normalises location strings
  - Re-scans title + snippet for emails, RFC 3966 tel: URIs, & WhatsApp links
  - Classifies entity_type (individual / business / unknown / parody_or_gimmick)
  - De-duplicates social links within a single candidate
"""
from __future__ import annotations

import logging
from typing import Set
from urllib.parse import urlparse

from app.platforms.contacts import extract_all_contacts
from app.schemas import RawCandidate, SocialLink, StructuredIntent
from app.utils.normalizers import clean_name, classify_candidate_entity, normalize_location
from app.utils.validators import format_tel_uri

logger = logging.getLogger(__name__)


class ProfileExtractor:

    def enrich_candidate(
        self, candidate: RawCandidate, intent: StructuredIntent
    ) -> RawCandidate:
        better_name = clean_name(candidate.name, candidate.platform)
        if better_name:
            candidate.name = better_name

        candidate.location = normalize_location(candidate.location)
        phone_ctx = (urlparse(candidate.profile_url).hostname or "").lower() or None

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
                tel_uri = format_tel_uri(phone, phone_context=phone_ctx)
                if tel_uri:
                    candidate.social_links.append(
                        SocialLink(platform="Phone", link=tel_uri)
                    )

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
