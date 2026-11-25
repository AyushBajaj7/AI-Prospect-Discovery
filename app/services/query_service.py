"""
Query service – wraps LLM intent parsing with a bounded TTL+LRU cache.
Stores both the parsed StructuredIntent AND the provider name that generated
it so cached responses never report stale provider metadata from another request.
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
        # {normalised_query: (cached_at_monotonic, StructuredIntent, provider_used)}
        self._cache: OrderedDict[str, Tuple[float, StructuredIntent, str]] = OrderedDict()

    def clear_cache(self) -> None:
        """Clear cached intents (useful in test isolation)."""
        self._cache.clear()

    def _evict_if_needed(self, now: float) -> None:
        """Remove expired entries and enforce CACHE_MAX_SIZE bound (LRU)."""
        expired = [
            k for k, (ts, _, _) in self._cache.items()
            if now - ts >= settings.CACHE_TTL_SECONDS
        ]
        for k in expired:
            del self._cache[k]
        while len(self._cache) >= settings.CACHE_MAX_SIZE:
            self._cache.popitem(last=False)

    async def get_structured_intent_with_provider(
        self, query: str
    ) -> Tuple[StructuredIntent, str]:
        """
        Return `(StructuredIntent, provider_used)` for `query` from cache or LLM.
        """
        cache_key = " ".join(query.strip().lower().split())
        now = time.monotonic()

        if cache_key in self._cache:
            cached_at, intent, provider = self._cache[cache_key]
            if now - cached_at < settings.CACHE_TTL_SECONDS:
                self._cache.move_to_end(cache_key)
                return intent, provider
            del self._cache[cache_key]

        intent, provider = await llm_service.parse_query_with_provider(query)
        self._evict_if_needed(now)
        self._cache[cache_key] = (now, intent, provider)
        return intent, provider

    async def get_structured_intent(self, query: str) -> StructuredIntent:
        """Return only the StructuredIntent for `query`."""
        intent, _ = await self.get_structured_intent_with_provider(query)
        return intent


query_service = QueryService()
