"""
Deduplication service – consolidates multi-platform candidate profiles that
belong to the same real-world individual or business into a single Prospect.

Matching priority (from most to least reliable):
  1. Exact profile URL (normalized)
  2. Shared personal/verified email address (respecting role-prefix & name compatibility)
  3. Shared public phone number (respecting name compatibility)
  4. Shared platform handle (username) with compatible location
  5. Identical normalized full name with BOTH locations present & compatible

Safety invariants:
  * Never merge two different profiles on the same 1-per-person social network
    (e.g. two distinct LinkedIn /in/ URLs are distinct people).
  * Never merge unrelated multi-word human names solely via a shared role email
    like cofounder@company.com or info@company.com found in a joint snippet.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Set

from app.schemas import Prospect, RawCandidate, SocialLink

logger = logging.getLogger(__name__)

# Handles that are too generic to reliably identify a person
_GENERIC_HANDLES: Set[str] = {
    "profile", "official", "page", "user", "admin", "info",
    "contact", "support", "help", "news", "home", "about",
}

# Social platforms where distinct profile URLs represent distinct accounts/people
_EXCLUSIVE_SOCIAL_PLATFORMS: Set[str] = {
    "LinkedIn", "X", "Instagram", "Threads", "YouTube", "Facebook",
}

# Generic role email local-parts that can be shared across a team or snippet
_GENERIC_EMAIL_PREFIXES: Set[str] = {
    "info", "hello", "contact", "support", "help", "admin", "team",
    "sales", "marketing", "office", "hi", "enquiries", "inquiries",
    "press", "media", "jobs", "careers", "founder", "cofounder", "ceo",
}


class DeduplicationService:

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Lowercase, strip non-alphanumeric chars, collapse whitespace."""
        cleaned = re.sub(r"[^\w\s]", "", name.lower())
        return " ".join(cleaned.split())

    @classmethod
    def _names_plausibly_same_person(cls, name_a: str, name_b: str) -> bool:
        """
        Return True if names share a meaningful token (e.g. surname 'patel' or
        first name 'alex'), or match via first-initial + surname ('S. Patel' vs 'Sam Patel').
        Rejects completely distinct names like 'Rahim Nathwani' vs 'Peter Elstrom'.
        """
        tokens_a = [t for t in cls._normalize_name(name_a).split() if len(t) >= 2]
        tokens_b = [t for t in cls._normalize_name(name_b).split() if len(t) >= 2]
        if not tokens_a or not tokens_b:
            return True  # insufficient name evidence to contradict
        return bool(set(tokens_a) & set(tokens_b))

    @staticmethod
    def _location_tokens(loc: str) -> Set[str]:
        return {tok.strip() for tok in re.split(r"[,/|]+", loc.lower()) if tok.strip()}

    @classmethod
    def _locations_compatible(cls, loc1: Optional[str], loc2: Optional[str]) -> bool:
        """
        Return True when locations share at least one whole token/segment
        or when one location is missing. Avoids naive substring collisions
        like 'India' matching 'Indiana, USA'.
        """
        if not loc1 or not loc2:
            return True
        t1 = cls._location_tokens(loc1)
        t2 = cls._location_tokens(loc2)
        if t1 & t2:
            return True
        a, b = loc1.lower().strip(), loc2.lower().strip()
        return bool(
            re.search(rf"\b{re.escape(a)}\b", b)
            or re.search(rf"\b{re.escape(b)}\b", a)
        )

    def _conflicts(self, a: RawCandidate, b: RawCandidate) -> bool:
        """Return True if two candidates demonstrably represent different accounts."""
        url_a = a.profile_url.lower()
        url_b = b.profile_url.lower()
        if url_a != url_b and a.platform == b.platform and a.platform in _EXCLUSIVE_SOCIAL_PLATFORMS:
            return True
        if a.location and b.location and not self._locations_compatible(a.location, b.location):
            return True
        return False

    def _should_merge(self, a: RawCandidate, b: RawCandidate) -> bool:
        url_a = a.profile_url.lower()
        url_b = b.profile_url.lower()

        # 1. Exact URL
        if url_a == url_b:
            return True

        # Guard: distinct URLs on the same exclusive social network belong to distinct accounts
        if self._conflicts(a, b):
            return False

        names_compatible = self._names_plausibly_same_person(a.name, b.name)

        # 2. Shared email
        emails_a = {e.lower() for e in a.emails}
        emails_b = {e.lower() for e in b.emails}
        shared_emails = emails_a & emails_b
        if shared_emails and self._locations_compatible(a.location, b.location):
            for em in shared_emails:
                local_part = em.split("@")[0]
                if local_part in _GENERIC_EMAIL_PREFIXES:
                    # Generic role email requires matching/compatible names
                    if names_compatible:
                        return True
                elif names_compatible or (
                    (a.metadata.get("handle") or "").lower()
                    and (a.metadata.get("handle") or "").lower()
                    == (b.metadata.get("handle") or "").lower()
                ):
                    return True

        # 3. Shared phone
        phones_a = {re.sub(r"\D", "", p) for p in a.phones if p}
        phones_b = {re.sub(r"\D", "", p) for p in b.phones if p}
        if phones_a and phones_b and (phones_a & phones_b) and names_compatible:
            return True

        # 4. Same platform handle (non-trivial and locations compatible)
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

        # 5. Exact full-name + BOTH locations verified and compatible
        norm_a = self._normalize_name(a.name)
        norm_b = self._normalize_name(b.name)
        if (
            norm_a
            and norm_b
            and norm_a == norm_b
            and len(norm_a.split()) >= 2
            and a.location
            and b.location
            and self._locations_compatible(a.location, b.location)
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate(self, candidates: List[RawCandidate]) -> List[Prospect]:
        """
        Cluster candidates with transitive closure into identity groups, then
        produce one Prospect per cluster with merged social links.
        """
        if not candidates:
            return []

        clusters: List[List[RawCandidate]] = []

        for cand in candidates:
            matching_indices = [
                idx
                for idx, cluster in enumerate(clusters)
                if any(self._should_merge(existing, cand) for existing in cluster)
                and not any(self._conflicts(existing, cand) for existing in cluster)
            ]
            if not matching_indices:
                clusters.append([cand])
            else:
                primary_idx = matching_indices[0]
                merged_other_indices = [
                    idx
                    for idx in matching_indices[1:]
                    if not any(
                        self._conflicts(c1, c2)
                        for c1 in clusters[primary_idx]
                        for c2 in clusters[idx]
                    )
                ]
                clusters[primary_idx].append(cand)
                for idx in reversed(merged_other_indices):
                    clusters[primary_idx].extend(clusters[idx])
                    del clusters[idx]

        _PLATFORM_ORDER = {
            "LinkedIn": 0, "X": 1, "YouTube": 2, "Instagram": 3,
            "Facebook": 4, "Threads": 5, "Google": 6,
            "WhatsApp": 7, "Email": 8, "Phone": 9,
        }

        prospects: List[Prospect] = []
        for cluster in clusters:
            primary_name = max(
                cluster,
                key=lambda c: (len(c.name.split()), len(c.name)),
            ).name

            locations = [c.location for c in cluster if c.location]
            best_location = max(locations, key=len) if locations else None

            merged_links: List[SocialLink] = []
            seen_links: Set[str] = set()
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

        prospects.sort(key=lambda p: len(p.social_link), reverse=True)
        return prospects


deduplication_service = DeduplicationService()
