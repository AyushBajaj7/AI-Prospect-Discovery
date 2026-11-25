"""
Prospect service – orchestrates the full discovery pipeline:

  1. Query understanding (LLM fallback chain -> deterministic heuristic)
  2. Concurrent multi-platform discovery (collecting explicit SearchStatus)
  3. Profile enrichment
  4. Strict relevance scoring & threshold filtering (no sub-threshold leaks)
  5. Deduplication & entity consolidation
  6. Structured JSON response + observable provider health summary
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Dict, List, Optional

from app.config import settings
from app.platforms import ALL_PLATFORMS
from app.schemas import (
    Prospect,
    ProspectDiscoveryResponse,
    ProspectQueryRequest,
    ProviderHealthSummary,
    RawCandidate,
    SearchStatus,
    StructuredIntent,
)
from app.services.deduplication_service import deduplication_service
from app.services.llm_service import llm_service
from app.services.profile_extractor import profile_extractor
from app.services.query_service import query_service
from app.services.relevance_service import relevance_service

logger = logging.getLogger(__name__)

# Concurrency-safe per-request status map populated by adapters during gather()
_request_outcomes_var: contextvars.ContextVar[Optional[Dict[str, SearchStatus]]] = contextvars.ContextVar(
    "_request_outcomes_var", default=None
)


class ProspectService:

    @staticmethod
    def record_platform_outcome(platform_name: str, status: SearchStatus) -> None:
        outcomes = _request_outcomes_var.get()
        if outcomes is not None:
            outcomes[platform_name] = status

    async def discover_prospects(
        self, request: ProspectQueryRequest
    ) -> ProspectDiscoveryResponse:
        query = request.query.strip()
        logger.info("Discovery started: query=%r", query)

        # --- Step 1: Understand the query ---
        intent: StructuredIntent = await query_service.get_structured_intent(query)
        llm_provider = str(llm_service.stats.get("provider_used", "heuristic_fallback"))
        logger.info(
            "Intent parsed: audience=%r interest=%r location=%r provider=%s",
            intent.audience,
            intent.interest,
            intent.location.formatted(),
            llm_provider,
        )

        # --- Step 2: Concurrent platform discovery ---
        outcome_map: Dict[str, SearchStatus] = {}
        token = _request_outcomes_var.set(outcome_map)
        try:
            tasks = [
                adapter.discover(intent, max_results=settings.MAX_RESULTS_PER_PLATFORM)
                for adapter in ALL_PLATFORMS
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            _request_outcomes_var.reset(token)

        raw_candidates: List[RawCandidate] = []
        for adapter, result in zip(ALL_PLATFORMS, results):
            plat = adapter.platform_name
            if isinstance(result, BaseException):
                status = (
                    SearchStatus.TIMEOUT
                    if isinstance(result, (TimeoutError, asyncio.TimeoutError))
                    else SearchStatus.PROVIDER_ERROR
                )
                outcome_map[plat] = status
                logger.error("Platform adapter %s failed (%s)", plat, type(result).__name__)
            elif isinstance(result, list):
                if plat not in outcome_map:
                    outcome_map[plat] = SearchStatus.SUCCESS if result else SearchStatus.NO_RESULTS
                logger.info(
                    "Platform %s (%s): %d candidate(s) found.",
                    plat,
                    outcome_map[plat].value,
                    len(result),
                )
                raw_candidates.extend(result)

        # --- Step 3: Enrich candidates ---
        enriched: List[RawCandidate] = [
            profile_extractor.enrich_candidate(c, intent) for c in raw_candidates
        ]

        # --- Step 4: Strict relevance filtering (no low-confidence leaks) ---
        relevant: List[RawCandidate] = relevance_service.filter_relevant_candidates(
            enriched, intent
        )
        logger.info(
            "Relevance filter: %d/%d candidates kept (threshold=%.0f).",
            len(relevant),
            len(enriched),
            settings.RELEVANCE_THRESHOLD,
        )

        # --- Step 5: Deduplication & consolidation ---
        consolidated: List[Prospect] = deduplication_service.consolidate(relevant)
        logger.info("Consolidated to %d unique prospect(s).", len(consolidated))

        # --- Step 6: Build observable provider health summary ---
        counts = {s: 0 for s in SearchStatus}
        for s in outcome_map.values():
            counts[s] += 1

        health = ProviderHealthSummary(
            llm_provider=llm_provider,
            platforms_succeeded=counts[SearchStatus.SUCCESS],
            platforms_no_results=counts[SearchStatus.NO_RESULTS],
            platforms_rate_limited=counts[SearchStatus.RATE_LIMITED],
            platforms_timed_out=counts[SearchStatus.TIMEOUT],
            platforms_errored=counts[SearchStatus.PROVIDER_ERROR],
            platform_outcomes=outcome_map,
        )

        return ProspectDiscoveryResponse(
            success=True,
            query=query,
            total_prospects=len(consolidated),
            prospects=consolidated,
            provider_health=health,
        )


prospect_service = ProspectService()
