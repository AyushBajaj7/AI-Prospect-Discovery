"""
Relevance scoring service.

Scores each RawCandidate against the parsed query intent on a 0–100 scale:

  Audience / profession match   40 pts
  Interest / topic match        20 pts
  Location match                25 pts
  Evidence quality              15 pts
  ─────────────────────────────────────
  Total                        100 pts

Candidates below settings.RELEVANCE_THRESHOLD are filtered out.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.schemas import PlatformPipelineMetrics, RawCandidate, StructuredIntent
from app.utils.normalizers import location_matches_text

_PERSONAL_ROLE_EVIDENCE_RE = re.compile(
    r"\b(?:founder|co-?founder|ceo|cto|cfo|owner|director|manager|engineer|"
    r"developer|designer|consultant|advisor|investor|coach|analyst|journalist|"
    r"writer|author|entrepreneur|lover|enthusiast|fan)\b",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)

# Semantic synonyms used for audience-match scoring when exact keywords miss
_AUDIENCE_SYNONYMS: Dict[str, List[str]] = {
    "founder": ["co-founder", "cofounder", "founding", "entrepreneur", "ceo", "chief executive"],
    "owner": ["proprietor", "managing director", "ceo", "operator", "principal"],
    "lover": ["fan", "enthusiast", "player", "coach", "supporter", "aficionado", "follower"],
    "agency": ["firm", "studio", "consultancy", "bureau", "group"],
}


def _stem(word: str) -> str:
    """Lightweight suffix strip (plurals / gerunds) for keyword matching."""
    for suffix in ("ing", "ers", "ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _keyword_in_text(keyword: str, text: str) -> bool:
    kw_lower = keyword.lower()
    kw_stem = _stem(kw_lower)
    if len(kw_lower) <= 4:
        return bool(re.search(rf"\b{re.escape(kw_lower)}\w*\b", text))
    return kw_lower in text or (
        len(kw_stem) >= 3 and bool(re.search(rf"\b{re.escape(kw_stem)}", text))
    )


class RelevanceService:

    def evaluate_candidate(
        self, candidate: RawCandidate, intent: StructuredIntent
    ) -> Tuple[float, Dict[str, float]]:
        """Return (total_score, breakdown_dict)."""

        searchable = (
            f"{candidate.name} {candidate.title} {candidate.snippet} {candidate.profile_url}"
        ).lower()

        breakdown: Dict[str, float] = {
            "audience_match": 0.0,
            "interest_match": 0.0,
            "location_match": 0.0,
            "evidence_quality": 0.0,
        }

        # ---------------------------------------------------------------- #
        # 1. Audience / profession match  (max 40)
        # ---------------------------------------------------------------- #
        stopwords = {"and", "the", "for", "with", "all", "are", "who", "that"}
        audience_kws = [
            w.lower()
            for w in re.findall(r"\b[A-Za-z][A-Za-z0-9]+\b", intent.audience)
            if len(w) > 2 and w.lower() not in stopwords
        ]

        if audience_kws:
            matched = sum(1 for kw in audience_kws if _keyword_in_text(kw, searchable))
            if matched == len(audience_kws):
                breakdown["audience_match"] = 40.0
            elif matched > 0:
                breakdown["audience_match"] = 40.0 * matched / len(audience_kws)
            else:
                for root, syns in _AUDIENCE_SYNONYMS.items():
                    if root in intent.audience.lower():
                        if any(_keyword_in_text(s, searchable) for s in syns):
                            breakdown["audience_match"] = 25.0
                            break
        else:
            breakdown["audience_match"] = 20.0

        # ---------------------------------------------------------------- #
        # 2. Interest / topic match  (max 20)
        # ---------------------------------------------------------------- #
        if intent.interest:
            interest_kws = [
                w.lower()
                for w in re.findall(r"\b[A-Za-z][A-Za-z0-9]+\b", intent.interest)
                if len(w) > 2
            ]
            if interest_kws:
                i_matched = sum(1 for kw in interest_kws if _keyword_in_text(kw, searchable))
                breakdown["interest_match"] = 20.0 if i_matched > 0 else 0.0
            else:
                breakdown["interest_match"] = 10.0
        else:
            breakdown["interest_match"] = 10.0

        # ---------------------------------------------------------------- #
        # 3. Location match  (max 25)
        # ---------------------------------------------------------------- #
        loc_score = 0.0
        no_location_in_query = not intent.location.city and not intent.location.country

        if no_location_in_query:
            loc_score = 25.0
        else:
            if intent.location.city and location_matches_text(intent.location.city, searchable):
                loc_score += 15.0
            if intent.location.state and location_matches_text(intent.location.state, searchable):
                loc_score += 5.0
            if intent.location.country and location_matches_text(intent.location.country, searchable):
                loc_score += 5.0

            # Fallback against stored candidate.location (only reliable when text didn't match)
            if loc_score == 0 and candidate.location:
                cand_loc_lower = candidate.location.lower()
                if intent.location.city and location_matches_text(intent.location.city, cand_loc_lower):
                    loc_score = 15.0
                elif intent.location.country and location_matches_text(intent.location.country, cand_loc_lower):
                    loc_score = 5.0

        breakdown["location_match"] = min(25.0, loc_score)

        # ---------------------------------------------------------------- #
        # 4. Evidence quality  (max 15)
        # ---------------------------------------------------------------- #
        evidence = 0.0
        if candidate.profile_url and len(candidate.profile_url) > 10:
            evidence += 5.0
        if candidate.name and len(candidate.name.split()) >= 2:
            evidence += 5.0
        if candidate.emails or candidate.phones or candidate.whatsapp_links:
            evidence += 5.0
        breakdown["evidence_quality"] = min(15.0, evidence)

        total = sum(breakdown.values())
        return total, breakdown

    @staticmethod
    def _passes_identity_gate(
        cand: RawCandidate,
        intent: StructuredIntent,
        breakdown: Dict[str, float],
    ) -> bool:
        """
        Separate identity-evidence gate preventing unverified short handles (e.g. 'AJ',
        'Raph') or ambiguous 'unknown' entities from passing solely on topical score.
        """
        alpha_only = re.sub(r"[^A-Za-z]", "", cand.name or "")
        name_words = [w for w in (cand.name or "").split() if any(ch.isalpha() for ch in w)]
        has_confirmed_location = bool(cand.location) or breakdown.get("location_match", 0.0) >= 15.0
        has_contact = bool(cand.emails or cand.phones or cand.whatsapp_links)
        is_linkedin_profile = cand.platform == "LinkedIn" and "/in/" in cand.profile_url.lower()

        if len(alpha_only) < 4:
            return False

        if intent.target_type == "individual":
            if cand.entity_type == "individual":
                # Confirmed individual: require >=2 words unless LinkedIn /in/ or verified loc+contact
                if len(name_words) >= 2 or is_linkedin_profile:
                    return True
                return has_confirmed_location and has_contact
            # "unknown" entity on individual-target query requires stronger corroboration
            if len(name_words) >= 2 and (has_confirmed_location or has_contact):
                return True
            if (
                len(name_words) >= 2
                and _PERSONAL_ROLE_EVIDENCE_RE.search(f"{cand.title} {cand.snippet}")
                and all(len(re.sub(r"[^A-Za-z]", "", w)) >= 2 for w in name_words)
            ):
                return True
            return False

        # target_type == "business"
        if cand.entity_type == "business":
            return len(alpha_only) >= 4
        return (len(name_words) >= 2 or len(alpha_only) >= 5) and (
            has_confirmed_location or has_contact
        )

    def filter_relevant_candidates(
        self,
        candidates: List[RawCandidate],
        intent: StructuredIntent,
        metrics_map: Optional[Dict[str, PlatformPipelineMetrics]] = None,
    ) -> List[RawCandidate]:
        relevant: List[RawCandidate] = []
        for cand in candidates:
            m = metrics_map.get(cand.platform) if metrics_map is not None else None

            # Exclude explicit parody/meme/gimmick accounts
            if cand.entity_type == "parody_or_gimmick":
                if m is not None:
                    m.parody_or_gimmick += 1
                logger.info(
                    "Platform %s rejected '%s' [reason=parody_or_gimmick]",
                    cand.platform,
                    cand.name,
                )
                continue

            # Enforce intent.target_type for confirmed mismatches while judging
            # ambiguous ("unknown") candidates with the stricter identity-evidence gate.
            if intent.target_type == "individual" and cand.entity_type == "business":
                if m is not None:
                    m.target_type_mismatches += 1
                logger.info(
                    "Platform %s rejected '%s' [reason=target_type_mismatch expected=individual got=business]",
                    cand.platform,
                    cand.name,
                )
                continue
            if intent.target_type == "business" and cand.entity_type == "individual":
                if m is not None:
                    m.target_type_mismatches += 1
                logger.info(
                    "Platform %s rejected '%s' [reason=target_type_mismatch expected=business got=individual]",
                    cand.platform,
                    cand.name,
                )
                continue

            score, breakdown = self.evaluate_candidate(cand, intent)
            cand.relevance_score = score
            cand.metadata["score_breakdown"] = breakdown

            if cand.location and (intent.location.city or intent.location.country):
                from app.services.deduplication_service import deduplication_service

                if not deduplication_service._locations_compatible(
                    cand.location, intent.location.formatted()
                ):
                    if m is not None:
                        m.below_relevance_threshold += 1
                    logger.info(
                        "Platform %s rejected '%s' [reason=location_mismatch cand=%r query=%r]",
                        cand.platform,
                        cand.name,
                        cand.location,
                        intent.location.formatted(),
                    )
                    continue

            if not self._passes_identity_gate(cand, intent, breakdown):
                if m is not None:
                    m.insufficient_identity_evidence += 1
                logger.info(
                    "Platform %s rejected '%s' [reason=insufficient_identity_evidence entity=%s]",
                    cand.platform,
                    cand.name,
                    cand.entity_type,
                )
                continue

            if score >= settings.RELEVANCE_THRESHOLD:
                if m is not None:
                    m.retained_candidates += 1
                relevant.append(cand)
            else:
                if m is not None:
                    m.below_relevance_threshold += 1
                logger.info(
                    "Platform %s rejected '%s' [reason=below_relevance_threshold score=%.1f threshold=%.1f]",
                    cand.platform,
                    cand.name,
                    score,
                    settings.RELEVANCE_THRESHOLD,
                )

        relevant.sort(key=lambda c: c.relevance_score, reverse=True)
        return relevant


relevance_service = RelevanceService()
