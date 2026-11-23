import asyncio
import logging
from typing import List
from app.schemas import (
    ProspectQueryRequest,
    ProspectDiscoveryResponse,
    Prospect,
    RawCandidate,
    StructuredIntent
)
from app.services.query_service import query_service
from app.platforms import ALL_PLATFORMS
from app.services.profile_extractor import profile_extractor
from app.services.relevance_service import relevance_service
from app.services.deduplication_service import deduplication_service
from app.config import settings

logger = logging.getLogger(__name__)

class ProspectService:
    async def discover_prospects(self, request: ProspectQueryRequest) -> ProspectDiscoveryResponse:
        query = request.query.strip()
        logger.info(f"Starting prospect discovery for query: '{query}'")

        # 1. Natural Language Query Understanding
        intent: StructuredIntent = await query_service.get_structured_intent(query)
        logger.info(f"Parsed intent: audience='{intent.audience}', interest='{intent.interest}', location={intent.location.formatted()}")

        # 2. Concurrently execute discovery across all supported platform adapters
        tasks = [
            adapter.discover(intent, max_results=settings.MAX_RESULTS_PER_PLATFORM)
            for adapter in ALL_PLATFORMS
        ]
        
        results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        raw_candidates: List[RawCandidate] = []
        for i, res in enumerate(results_nested):
            adapter_name = ALL_PLATFORMS[i].platform_name
            if isinstance(res, Exception):
                logger.error(f"Platform adapter '{adapter_name}' encountered an error: {res}")
            elif isinstance(res, list):
                logger.info(f"Platform adapter '{adapter_name}' discovered {len(res)} candidates.")
                raw_candidates.extend(res)

        # 3. Profile Extraction & Enrichment
        enriched_candidates = [
            profile_extractor.enrich_candidate(c, intent)
            for c in raw_candidates
        ]

        # 4. Relevance Scoring & Filtering
        relevant_candidates = relevance_service.filter_relevant_candidates(enriched_candidates, intent)
        logger.info(f"Filtered {len(relevant_candidates)} relevant candidates from {len(enriched_candidates)} raw candidates.")

        # If strict filtering leaves 0 candidates but raw candidates exist, gracefully take top scored candidates
        if not relevant_candidates and enriched_candidates:
            logger.info("Gracefully retaining top-ranked candidates to ensure non-empty results if available.")
            enriched_candidates.sort(key=lambda x: getattr(x, "relevance_score", 0.0), reverse=True)
            relevant_candidates = enriched_candidates[:5]

        # 5. Duplicate Detection & Entity Consolidation
        consolidated_prospects: List[Prospect] = deduplication_service.consolidate(relevant_candidates)
        logger.info(f"Consolidated into {len(consolidated_prospects)} unique prospect records.")

        return ProspectDiscoveryResponse(
            success=True,
            query=query,
            total_prospects=len(consolidated_prospects),
            prospects=consolidated_prospects
        )

prospect_service = ProspectService()
