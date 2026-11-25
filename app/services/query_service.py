"""
Query service – wraps LLM intent parsing with a bounded TTL in-process cache.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Tuple

from app.config import settings
from app.schemas import StructuredIntent
from app.services.llm_service import llm_service


class QueryService:

    def __init__(self) -> None:
        # {normalised_query: (cached_at_monotonic, StructuredIntent)}
        self._cache: OrderedDict[str, Tuple[float, StructuredIntent]] = OrderedDict()

    def _evict_if_needed(self, now: float) -> None:
        """Remove expired entries and enforce CACHE_MAX_SIZE bound (LRU)."""
        expired = [
            k for k, (ts, _) in self._cache.items()
            if now - ts >= settings.CACHE_TTL_SECONDS
        ]
        for k in expired:
            del self._cache[k]
        while len(self._cache) >= settings.CACHE_MAX_SIZE:
            self._cache.popitem(last=False)

    async def get_structured_intent(self, query: str) -> StructuredIntent:
        """
        Return the parsed StructuredIntent for `query`, using the bounded TTL
        in-process cache to avoid redundant LLM calls for identical queries.
        """
        cache_key = " ".join(query.strip().lower().split())
        now = time.monotonic()

        if cache_key in self._cache:
            cached_at, intent = self._cache[cache_key]
            if now - cached_at < settings.CACHE_TTL_SECONDS:
                self._cache.move_to_end(cache_key)
                return intent
            del self._cache[cache_key]

        intent = await llm_service.parse_query(query)
        self._evict_if_needed(now)
        self._cache[cache_key] = (now, intent)
        return intent


query_service = QueryService()
