"""
Deduplication service – consolidates multi-platform candidate profiles that
belong to the same real-world individual or business into a single Prospect.

Matching priority (from most to least reliable):
  1. Exact profile URL (normalized)
  2. Shared personal (non-generic) email address + full/surname-initial compatibility
  3. Shared public phone number + compatible full/surname-initial names
  4. Shared verified platform handle (username) with compatible location
  5. Identical normalized full name with BOTH locations present & compatible

Safety invariants:
  * Never merge two different accounts on the same exclusive social network.
  * Never merge unrelated individuals who only share a first name ("John Smith"
    vs "John Doe") or a generic company email (info@, contact@, cofounder@).
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Set

from app.schemas import Prospect, RawCandidate, SocialLink
from app.utils.validators import is_generic_role_email

logger = logging.getLogger(__name__)

_GENERIC_HANDLES: Set[str] = {
    "profile", "official", "page", "user", "admin", "info",
    "contact", "support", "help", "news", "home", "about",
}

_EXCLUSIVE_SOCIAL_PLATFORMS: Set[str] = {
    "LinkedIn", "X", "Instagram", "Threads", "YouTube", "Facebook",
}


class DeduplicationService:

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Lowercase, strip non-alphanumeric chars, collapse whitespace."""
        cleaned = re.sub(r"[^\w\s]", "", name.lower())
        return " ".join(cleaned.split())

    @classmethod
    def _names_plausibly_same_person(cls, name_a: str, name_b: str) -> bool:
        """
        Strict name-compatibility check:
          * Exact normalized names match ("john smith" == "john smith"), OR
          * Same surname (last token) AND compatible first name or initial
            ("Sam Patel" vs "S. Patel" -> True), while ("John Smith" vs "John Doe" -> False).
        """
        tokens_a = cls._normalize_name(name_a).split()
        tokens_b = cls._normalize_name(name_b).split()
        if not tokens_a or not tokens_b:
            return False
        if tokens_a == tokens_b:
            return True
        if len(tokens_a) >= 2 and len(tokens_b) >= 2:
            first_a, last_a = tokens_a[0], tokens_a[-1]
            first_b, last_b = tokens_b[0], tokens_b[-1]
            if last_a != last_b or len(last_a) < 2:
                return False
            # Match either exact first name or initial match ("s" vs "sam")
            return (
                first_a == first_b
                or (len(first_a) == 1 and first_b.startswith(first_a))
                or (len(first_b) == 1 and first_a.startswith(first_b))
            )
        # Single-word entity/brand matching a longer brand name containing it
        if len(tokens_a) == 1 and len(tokens_a[0]) >= 3:
            return tokens_a[0] in tokens_b
        if len(tokens_b) == 1 and len(tokens_b[0]) >= 3:
            return tokens_b[0] in tokens_a
        return False

    @staticmethod
    def _location_tokens(loc: str) -> Set[str]:
        return {tok.strip() for tok in re.split(r"[,/|]+", loc.lower()) if tok.strip()}

    @classmethod
    def _locations_compatible(cls, loc1: Optional[str], loc2: Optional[str]) -> bool:
        """
        Return True when locations share at least one whole token/segment
        or when one location is missing. Avoids substring collisions like
        'India' matching 'Indiana, USA'.
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
        """Return True if two candidates demonstrably represent different people/accounts."""
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

        # Conflict guard
        if self._conflicts(a, b):
            return False

        names_compatible = self._names_plausibly_same_person(a.name, b.name)
        handle_a = (a.metadata.get("handle") or "").lower()
        handle_b = (b.metadata.get("handle") or "").lower()
        same_handle = bool(
            handle_a
            and handle_b
            and handle_a == handle_b
            and len(handle_a) >= 4
            and handle_a not in _GENERIC_HANDLES
        )

        # 2. Shared email (generic company emails never establish personal identity alone)
        emails_a = {e.lower() for e in a.emails}
        emails_b = {e.lower() for e in b.emails}
        shared_emails = emails_a & emails_b
        if shared_emails:
            exact_full_name = (
                self._normalize_name(a.name) == self._normalize_name(b.name)
                and len(self._normalize_name(a.name).split()) >= 2
            )
            for em in shared_emails:
                if is_generic_role_email(em):
                    if (exact_full_name and self._locations_compatible(a.location, b.location)) or same_handle:
                        return True
                elif names_compatible or same_handle:
                    return True

        # 3. Shared phone (requires compatible name or verified handle)
        phones_a = {re.sub(r"\D", "", p) for p in a.phones if p}
        phones_b = {re.sub(r"\D", "", p) for p in b.phones if p}
        if phones_a and phones_b and (phones_a & phones_b):
            if names_compatible or same_handle:
                return True

        # 4. Same platform handle (non-trivial and locations compatible)
        if same_handle and self._locations_compatible(a.location, b.location):
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

    def consolidate(self, candidates: List[RawCandidate]) -> List[Prospect]:
        """Cluster candidates with transitive closure and conflict guards."""
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
