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

from app.platforms.contacts import extract_all_contacts, filter_attributed_emails
from app.schemas import RawCandidate, SocialLink, StructuredIntent
from app.utils.normalizers import clean_name, classify_candidate_entity, normalize_location
from app.utils.validators import derive_phone_context, format_tel_uri

logger = logging.getLogger(__name__)


class ProfileExtractor:

    def enrich_candidate(
        self, candidate: RawCandidate, intent: StructuredIntent
    ) -> RawCandidate:
        better_name = clean_name(candidate.name, candidate.platform)
        if better_name:
            candidate.name = better_name

        candidate.location = normalize_location(candidate.location)
        candidate.entity_type = classify_candidate_entity(
            candidate.name,
            candidate.platform,
            candidate.profile_url,
            candidate.title,
            candidate.snippet,
        )

        phone_ctx = derive_phone_context(candidate.profile_url, candidate.platform)
        extra = extract_all_contacts(f"{candidate.title} {candidate.snippet}")

        candidate.emails = filter_attributed_emails(
            candidate.emails, candidate.entity_type, candidate.profile_url
        )
        attributed_extra_emails = filter_attributed_emails(
            extra["emails"], candidate.entity_type, candidate.profile_url
        )
        allowed_emails = set(candidate.emails) | set(attributed_extra_emails)
        candidate.social_links = [
            sl
            for sl in candidate.social_links
            if sl.platform != "Email" or sl.link.removeprefix("mailto:").lower() in allowed_emails
        ]

        for email in attributed_extra_emails:
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
