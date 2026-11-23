"""
Prospect service – orchestrates the full discovery pipeline:

  1. Query understanding (LLM / heuristic)
  2. Concurrent multi-platform discovery
  3. Profile enrichment
  4. Relevance scoring & filtering
  5. Deduplication & entity consolidation
  6. Structured JSON response
"""
from __future__ import annotations

import asyncio
import logging
from typing import List

from app.config import settings
from app.platforms import ALL_PLATFORMS
from app.schemas import (
    Prospect,
    ProspectDiscoveryResponse,
    ProspectQueryRequest,
    RawCandidate,
    StructuredIntent,
)
from app.services.deduplication_service import deduplication_service
from app.services.profile_extractor import profile_extractor
from app.services.query_service import query_service
from app.services.relevance_service import relevance_service

logger = logging.getLogger(__name__)


class ProspectService:

    async def discover_prospects(
        self, request: ProspectQueryRequest
    ) -> ProspectDiscoveryResponse:
        query = request.query.strip()
        logger.info("Discovery started: query=%r", query)

        # --- Step 1: Understand the query ---
        intent: StructuredIntent = await query_service.get_structured_intent(query)
        logger.info(
            "Intent parsed: audience=%r interest=%r location=%r provider=%s",
            intent.audience,
            intent.interest,
            intent.location.formatted(),
            "see llm_service.stats",
        )

        # --- Step 2: Concurrent platform discovery ---
        tasks = [
            adapter.discover(intent, max_results=settings.MAX_RESULTS_PER_PLATFORM)
            for adapter in ALL_PLATFORMS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        raw_candidates: List[RawCandidate] = []
        for adapter, result in zip(ALL_PLATFORMS, results):
            if isinstance(result, Exception):
                logger.error(
                    "Platform adapter %s failed: %s",
                    adapter.platform_name,
                    type(result).__name__,
                )
            elif isinstance(result, list):
                logger.info(
                    "Platform %s: %d candidate(s) found.",
                    adapter.platform_name,
                    len(result),
                )
                raw_candidates.extend(result)

        # --- Step 3: Enrich candidates ---
        enriched: List[RawCandidate] = [
            profile_extractor.enrich_candidate(c, intent) for c in raw_candidates
        ]

        # --- Step 4: Relevance filtering ---
        relevant: List[RawCandidate] = relevance_service.filter_relevant_candidates(
            enriched, intent
        )
        logger.info(
            "Relevance filter: %d/%d candidates kept (threshold=%.0f).",
            len(relevant),
            len(enriched),
            settings.RELEVANCE_THRESHOLD,
        )

        # Graceful fallback: if strict threshold removes everything, keep
        # top-5 by score so the endpoint always returns *something* useful.
        if not relevant and enriched:
            logger.info("Threshold too strict; returning top-5 by score as fallback.")
            enriched.sort(key=lambda c: c.relevance_score, reverse=True)
            relevant = enriched[:5]

        # --- Step 5: Deduplication & consolidation ---
        consolidated: List[Prospect] = deduplication_service.consolidate(relevant)
        logger.info("Consolidated to %d unique prospect(s).", len(consolidated))

        return ProspectDiscoveryResponse(
            success=True,
            query=query,
            total_prospects=len(consolidated),
            prospects=consolidated,
        )


prospect_service = ProspectService()
