"""
Deduplication service – consolidates multi-platform candidate profiles that
belong to the same real-world individual or business into a single Prospect.

Matching priority (from most to least reliable):
  1. Exact profile URL (normalized)
  2. Shared public email address
  3. Shared public phone number
  4. Shared platform handle (username) with compatible location
  5. Identical normalized full name with compatible location

Rule: never merge two candidates based on a single common given name alone.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

from app.schemas import Prospect, RawCandidate, SocialLink

logger = logging.getLogger(__name__)

# Handles that are too generic to reliably identify a person
_GENERIC_HANDLES: Set[str] = {
    "profile", "official", "page", "user", "admin", "info",
    "contact", "support", "help", "news", "home", "about",
}


class DeduplicationService:

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_name(self, name: str) -> str:
        """Lowercase, strip non-alphanumeric chars, collapse whitespace."""
        cleaned = re.sub(r"[^\w\s]", "", name.lower())
        return " ".join(cleaned.split())

    @staticmethod
    def _locations_compatible(loc1: Optional[str], loc2: Optional[str]) -> bool:
        """
        Return True when locations are not contradictory.

        Missing location is treated as "compatible with anything."
        """
        if not loc1 or not loc2:
            return True
        a, b = loc1.lower(), loc2.lower()
        return a in b or b in a

    def _should_merge(self, a: RawCandidate, b: RawCandidate) -> bool:
        # 1. Exact URL
        if a.profile_url.lower() == b.profile_url.lower():
            return True

        # 2. Shared email
        emails_a = {e.lower() for e in a.emails}
        emails_b = {e.lower() for e in b.emails}
        if emails_a and emails_b and emails_a & emails_b:
            return True

        # 3. Shared phone
        phones_a = set(a.phones)
        phones_b = set(b.phones)
        if phones_a and phones_b and phones_a & phones_b:
            return True

        # 4. Same platform handle (must be non-trivial and locations compatible)
        handle_a = (a.metadata.get("handle") or "").lower()
        handle_b = (b.metadata.get("handle") or "").lower()
        if (
            handle_a
            and handle_b
            and handle_a == handle_b
            and len(handle_a) >= 4
            and handle_a not in _GENERIC_HANDLES
            and self._locations_compatible(a.location, b.location)
        ):
            return True

        # 5. Exact full-name + compatible location (multi-word names only)
        norm_a = self._normalize_name(a.name)
        norm_b = self._normalize_name(b.name)
        if (
            norm_a
            and norm_b
            and norm_a == norm_b
            and len(norm_a.split()) >= 2
            and self._locations_compatible(a.location, b.location)
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate(self, candidates: List[RawCandidate]) -> List[Prospect]:
        """
        Cluster candidates into identity groups, then produce one Prospect
        per cluster with merged social links.
        """
        if not candidates:
            return []

        clusters: List[List[RawCandidate]] = []

        for cand in candidates:
            merged = False
            for cluster in clusters:
                if any(self._should_merge(existing, cand) for existing in cluster):
                    cluster.append(cand)
                    merged = True
                    break
            if not merged:
                clusters.append([cand])

        prospects: List[Prospect] = []
        for cluster in clusters:
            # Pick the most descriptive name (most words, then longest)
            primary_name = max(
                cluster,
                key=lambda c: (len(c.name.split()), len(c.name)),
            ).name

            # Pick the most specific (longest) location
            locations = [c.location for c in cluster if c.location]
            best_location = max(locations, key=len) if locations else None

            # Merge social links without duplicates (normalise URL case)
            merged_links: List[SocialLink] = []
            seen_links: Set[str] = set()
            # Sort so social-platform links come before email/phone/WhatsApp
            _PLATFORM_ORDER = {
                "LinkedIn": 0, "X": 1, "YouTube": 2, "Instagram": 3,
                "Facebook": 4, "Threads": 5, "Google": 6,
                "WhatsApp": 7, "Email": 8, "Phone": 9,
            }
            all_links = [sl for cand in cluster for sl in cand.social_links]
            all_links.sort(key=lambda sl: _PLATFORM_ORDER.get(sl.platform, 99))

            for sl in all_links:
                key = sl.link.lower().strip()
                if key not in seen_links:
                    seen_links.add(key)
                    merged_links.append(sl)

            prospects.append(
                Prospect(
                    name=primary_name,
                    location=best_location,
                    social_link=merged_links,
                )
            )

        # Sort prospects by number of social links descending (richer first)
        prospects.sort(key=lambda p: len(p.social_link), reverse=True)
        return prospects


deduplication_service = DeduplicationService()
