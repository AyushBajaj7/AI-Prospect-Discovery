"""
Prospect service – orchestrates the full discovery pipeline:

  1. Query understanding (cached with exact provider attribution)
  2. Concurrent multi-platform discovery (collecting per-platform SearchStatus + stage counters)
  3. Profile enrichment & entity-type classification
  4. Strict relevance, parody, & target_type filtering with per-platform rejection accounting
  5. Deduplication & conflict-aware entity consolidation
  6. Structured JSON response + ProviderHealthSummary telemetry
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.platforms import ALL_PLATFORMS
from app.schemas import (
    PlatformPipelineMetrics,
    Prospect,
    ProspectDiscoveryResponse,
    ProspectQueryRequest,
    ProviderHealthSummary,
    RawCandidate,
    SearchStatus,
)
from app.services.deduplication_service import deduplication_service
from app.services.profile_extractor import profile_extractor
from app.services.query_service import QueryService, query_service
from app.services.relevance_service import relevance_service

logger = logging.getLogger(__name__)

# Concurrency-safe per-request metrics map populated by adapters and relevance filter
_request_metrics_var: contextvars.ContextVar[Optional[Dict[str, PlatformPipelineMetrics]]] = (
    contextvars.ContextVar("_request_metrics_var", default=None)
)


class ProspectService:

    @staticmethod
    def record_platform_metrics(platform_name: str, metrics: PlatformPipelineMetrics) -> None:
        metrics_map = _request_metrics_var.get()
        if metrics_map is not None:
            metrics_map[platform_name] = metrics

    async def discover_prospects_with_health(
        self, request: ProspectQueryRequest
    ) -> Tuple[ProspectDiscoveryResponse, ProviderHealthSummary]:
        query = request.query.strip()
        logger.info("Discovery started: query=%r", query)

        # --- Step 1: Understand the query (cached together with exact provider) ---
        if (
            getattr(query_service.get_structured_intent, "__func__", None)
            is not QueryService.get_structured_intent
        ):
            intent = await query_service.get_structured_intent(query)
            llm_provider = "mocked"
        else:
            intent, llm_provider = await query_service.get_structured_intent_with_provider(query)

        logger.info(
            "Intent parsed: audience=%r interest=%r location=%r target_type=%s provider=%s",
            intent.audience,
            intent.interest,
            intent.location.formatted(),
            intent.target_type,
            llm_provider,
        )

        # --- Step 2: Concurrent platform discovery ---
        metrics_map: Dict[str, PlatformPipelineMetrics] = {}
        token = _request_metrics_var.set(metrics_map)
        try:
            tasks = [
                adapter.discover(intent, max_results=settings.MAX_RESULTS_PER_PLATFORM)
                for adapter in ALL_PLATFORMS
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            _request_metrics_var.reset(token)

        raw_candidates: List[RawCandidate] = []
        for adapter, result in zip(ALL_PLATFORMS, results):
            plat = adapter.platform_name
            if isinstance(result, BaseException):
                status = (
                    SearchStatus.TIMEOUT
                    if isinstance(result, (TimeoutError, asyncio.TimeoutError))
                    else SearchStatus.PROVIDER_ERROR
                )
                metrics_map[plat] = PlatformPipelineMetrics(search_status=status)
                logger.error("Platform adapter %s failed (%s)", plat, type(result).__name__)
            elif isinstance(result, list):
                if plat not in metrics_map:
                    metrics_map[plat] = PlatformPipelineMetrics(
                        search_status=SearchStatus.SUCCESS if result else SearchStatus.NO_RESULTS,
                        search_hits=len(result),
                    )
                raw_candidates.extend(result)

        # --- Step 3: Enrich & classify candidate entity types ---
        enriched: List[RawCandidate] = [
            profile_extractor.enrich_candidate(c, intent) for c in raw_candidates
        ]

        # --- Step 4: Strict relevance, parody & target_type filtering ---
        relevant: List[RawCandidate] = relevance_service.filter_relevant_candidates(
            enriched, intent, metrics_map=metrics_map
        )

        # Log full per-platform funnel at INFO level for transparent observability
        for plat, pm in metrics_map.items():
            logger.info(
                "Platform funnel [%s]: status=%s hits=%d invalid_url=%d bad_name=%d "
                "missing_contact=%d parody=%d target_mismatch=%d low_score=%d retained=%d",
                plat,
                pm.search_status.value,
                pm.search_hits,
                pm.invalid_urls,
                pm.rejected_names,
                pm.missing_required_contact,
                pm.parody_or_gimmick,
                pm.target_type_mismatches,
                pm.below_relevance_threshold,
                pm.retained_candidates,
            )

        # --- Step 5: Deduplication & consolidation ---
        consolidated: List[Prospect] = deduplication_service.consolidate(relevant)
        logger.info("Consolidated to %d unique prospect(s).", len(consolidated))

        # --- Step 6: Build provider health telemetry ---
        outcome_map = {plat: pm.search_status for plat, pm in metrics_map.items()}
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
            platform_metrics=metrics_map,
        )

        response = ProspectDiscoveryResponse(
            success=True,
            query=query,
            total_prospects=len(consolidated),
            prospects=consolidated,
            provider_health=None,
        )
        return response, health

    async def discover_prospects(
        self, request: ProspectQueryRequest
    ) -> ProspectDiscoveryResponse:
        response, _ = await self.discover_prospects_with_health(request)
        return response


prospect_service = ProspectService()
