import re
import logging
from typing import List, Tuple, Dict
from app.schemas import RawCandidate, StructuredIntent
from app.config import settings

logger = logging.getLogger(__name__)

class RelevanceService:
    def evaluate_candidate(self, candidate: RawCandidate, intent: StructuredIntent) -> Tuple[float, Dict[str, float]]:
        """Calculates a 0-100 relevance score based on audience, interest, location, and evidence quality."""
        score = 0.0
        breakdown = {
            "audience_match": 0.0,
            "interest_match": 0.0,
            "location_match": 0.0,
            "evidence_quality": 0.0
        }

        searchable_text = f"{candidate.name} {candidate.title} {candidate.snippet} {candidate.profile_url}".lower()

        # 1. Audience / Profession Match (Max 40 points)
        audience_keywords = [
            w.lower() for w in re.findall(r"\b[A-Za-z0-9]+\b", intent.audience)
            if len(w) > 2 and w.lower() not in ["and", "the", "for", "with", "all"]
        ]
        
        def matches_keyword(word: str, text: str) -> bool:
            w_stem = word.rstrip("s")
            return word in text or (len(w_stem) >= 3 and w_stem in text)

        matched_count = sum(1 for kw in audience_keywords if matches_keyword(kw, searchable_text))
        if matched_count == len(audience_keywords) and len(audience_keywords) > 0:
            breakdown["audience_match"] = 40.0
        elif matched_count > 0:
            breakdown["audience_match"] = 40.0 * (matched_count / len(audience_keywords))
        else:
            # Check related semantic synonyms
            synonyms = {
                "founder": ["co-founder", "founding", "entrepreneur", "ceo", "partner"],
                "owner": ["proprietor", "founder", "managing director", "ceo", "partner"],
                "lover": ["fan", "enthusiast", "player", "coach", "supporter", "aficionado"]
            }
            for key, syns in synonyms.items():
                if key in intent.audience.lower() and any(s in searchable_text for s in syns):
                    breakdown["audience_match"] = 30.0
                    break

        # 2. Interest Match (Max 20 points)
        if intent.interest:
            interest_kw = [
                w.lower() for w in re.findall(r"\b[A-Za-z0-9]+\b", intent.interest)
                if len(w) > 2
            ]
            i_matches = sum(1 for kw in interest_kw if matches_keyword(kw, searchable_text))
            if i_matches > 0:
                breakdown["interest_match"] = 20.0
        else:
            breakdown["interest_match"] = 15.0  # neutral if no separate interest

        # 3. Location Match (Max 25 points)
        loc_score = 0.0
        if intent.location.city and intent.location.city.lower() in searchable_text:
            loc_score += 15.0
        if intent.location.state and intent.location.state.lower() in searchable_text:
            loc_score += 5.0
        if intent.location.country and intent.location.country.lower() in searchable_text:
            loc_score += 5.0
        elif not intent.location.country and loc_score > 0:
            loc_score += 10.0  # City matched when country not specified

        # If location not in query at all, give neutral score
        if not intent.location.city and not intent.location.country:
            loc_score = 25.0

        breakdown["location_match"] = min(25.0, loc_score)

        # 4. Profile & Evidence Quality (Max 15 points)
        evidence_score = 0.0
        if candidate.profile_url and len(candidate.profile_url) > 10:
            evidence_score += 5.0  # Verified URL
        if candidate.name and len(candidate.name.split()) >= 2:
            evidence_score += 5.0  # Realistic multi-word human or business name
        if candidate.emails or candidate.phones or candidate.whatsapp_links:
            evidence_score += 5.0  # Verified contact information available

        breakdown["evidence_quality"] = min(15.0, evidence_score)

        total_score = sum(breakdown.values())
        return total_score, breakdown

    def filter_relevant_candidates(
        self,
        candidates: List[RawCandidate],
        intent: StructuredIntent
    ) -> List[RawCandidate]:
        """Filters candidates below the minimum relevance threshold."""
        relevant = []
        for c in candidates:
            score, breakdown = self.evaluate_candidate(c, intent)
            c.relevance_score = score
            c.metadata["score_breakdown"] = breakdown
            if score >= settings.RELEVANCE_THRESHOLD:
                relevant.append(c)
            else:
                logger.debug(f"Filtered candidate '{c.name}' due to low relevance score: {score:.1f}")
        return relevant

relevance_service = RelevanceService()
