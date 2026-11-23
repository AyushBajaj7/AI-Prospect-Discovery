"""
Query service – wraps LLM intent parsing with a time-based in-process cache.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

from app.config import settings
from app.schemas import StructuredIntent
from app.services.llm_service import llm_service


class QueryService:

    def __init__(self) -> None:
        # {normalised_query: (cached_at_timestamp, StructuredIntent)}
        self._cache: Dict[str, Tuple[float, StructuredIntent]] = {}

    async def get_structured_intent(self, query: str) -> StructuredIntent:
        """
        Return the parsed StructuredIntent for `query`, using the in-process
        cache to avoid redundant LLM calls for identical queries.
        """
        cache_key = query.strip().lower()
        now = time.monotonic()

        if cache_key in self._cache:
            cached_at, intent = self._cache[cache_key]
            if now - cached_at < settings.CACHE_TTL_SECONDS:
                return intent

        intent = await llm_service.parse_query(query)
        self._cache[cache_key] = (now, intent)
        return intent


query_service = QueryService()
