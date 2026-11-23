import logging
import re
from typing import List, Dict, Set, Optional
from app.schemas import RawCandidate, Prospect, SocialLink

logger = logging.getLogger(__name__)

class DeduplicationService:
    """Consolidates multi-platform candidate profiles belonging to the same entity/individual."""

    def _normalize_name(self, name: str) -> str:
        """Lowers and strips punctuation for comparison."""
        cleaned = re.sub(r"[^\w\s]", "", name.lower())
        return " ".join(cleaned.split())

    def _are_locations_compatible(self, loc1: Optional[str], loc2: Optional[str]) -> bool:
        """Checks if two locations are either identical or one contains the other."""
        if not loc1 or not loc2:
            return True  # Location missing on one doesn't forbid match if other strong signals exist
        l1, l2 = loc1.lower(), loc2.lower()
        return l1 in l2 or l2 in l1

    def _should_merge(self, a: RawCandidate, b: RawCandidate) -> bool:
        """Evaluates whether candidate A and candidate B represent the same individual."""
        # 1. Exact URL match
        if a.profile_url.lower() == b.profile_url.lower():
            return True

        # 2. Shared Contact Evidence (Email or Phone)
        if a.emails and b.emails:
            if set(e.lower() for e in a.emails).intersection(set(e.lower() for e in b.emails)):
                return True

        if a.phones and b.phones:
            if set(a.phones).intersection(set(b.phones)):
                return True

        # 3. Shared Social Handle across known platforms
        handle_a = a.metadata.get("handle")
        handle_b = b.metadata.get("handle")
        if handle_a and handle_b and len(handle_a) >= 3:
            # Avoid generic handles like "pages", "home", "channel"
            if handle_a == handle_b and handle_a not in ["profile", "official", "page", "user"]:
                # If handles match, ensure locations aren't completely contradictory
                if self._are_locations_compatible(a.location, b.location):
                    return True

        # 4. Normalized Full Name Match + Location Match
        norm_a = self._normalize_name(a.name)
        norm_b = self._normalize_name(b.name)
        if norm_a and norm_b and norm_a == norm_b:
            words = norm_a.split()
            # Only match by name if multi-word name (avoid single common names like "John")
            if len(words) >= 2:
                # Require compatible locations or identical platform/context
                if a.location and b.location:
                    if self._are_locations_compatible(a.location, b.location):
                        return True
                elif not a.location and not b.location:
                    return True

        return False

    def consolidate(self, candidates: List[RawCandidate]) -> List[Prospect]:
        """Groups matching candidates and produces unified Prospect records."""
        if not candidates:
            return []

        clusters: List[List[RawCandidate]] = []

        for cand in candidates:
            matched_cluster = None
            for cluster in clusters:
                if any(self._should_merge(existing, cand) for existing in cluster):
                    matched_cluster = cluster
                    break

            if matched_cluster is not None:
                matched_cluster.append(cand)
            else:
                clusters.append([cand])

        prospects: List[Prospect] = []
        for cluster in clusters:
            # Best representative name: multi-word or cleanest
            primary_name = cluster[0].name
            for item in cluster:
                if len(item.name.split()) > len(primary_name.split()):
                    primary_name = item.name

            # Best location: most specific (longest)
            locations = [c.location for c in cluster if c.location]
            best_location = max(locations, key=len) if locations else None

            # Consolidate social links without duplicates
            consolidated_links: List[SocialLink] = []
            seen_urls: Set[str] = set()

            for item in cluster:
                for sl in item.social_links:
                    url_norm = sl.link.lower().strip()
                    if url_norm not in seen_urls:
                        seen_urls.add(url_norm)
                        consolidated_links.append(sl)

            prospects.append(
                Prospect(
                    name=primary_name,
                    location=best_location,
                    social_link=consolidated_links
                )
            )

        return prospects

deduplication_service = DeduplicationService()
