"""
Search service – multi-provider web search with bounded caching, concurrency
limiting, HTTP 429 retry handling, and graceful fallback chain.

Provider priority (auto mode): Serper → Brave → DuckDuckGo (free, no key).

Security notes
--------------
* API keys are read from settings; never logged or echoed.
* All results are sanitized dicts with str values – no code execution.
* Explicit timeouts and concurrency limits protect against hung connections
  and upstream rate-limit bursts.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

SearchHit = Dict[str, str]  # keys: title, link, snippet


class SearchService:

    def __init__(self) -> None:
        self._timeout = httpx.Timeout(settings.SEARCH_TIMEOUT_SECONDS)
        self._semaphore = asyncio.Semaphore(settings.SEARCH_MAX_CONCURRENCY)
        # {(normalized_query, num_results): (cached_at_monotonic, List[SearchHit])}
        self._cache: OrderedDict[Tuple[str, int], Tuple[float, List[SearchHit]]] = OrderedDict()

    def _get_cached(self, query: str, num_results: int) -> List[SearchHit] | None:
        key = (query.strip(), num_results)
        now = time.monotonic()
        if key in self._cache:
            cached_at, hits = self._cache[key]
            if now - cached_at < settings.CACHE_TTL_SECONDS:
                self._cache.move_to_end(key)
                return [dict(h) for h in hits]
            del self._cache[key]
        return None

    def _put_cache(self, query: str, num_results: int, hits: List[SearchHit]) -> None:
        now = time.monotonic()
        expired = [
            k for k, (ts, _) in self._cache.items()
            if now - ts >= settings.CACHE_TTL_SECONDS
        ]
        for k in expired:
            del self._cache[k]
        while len(self._cache) >= settings.CACHE_MAX_SIZE:
            self._cache.popitem(last=False)
        self._cache[(query.strip(), num_results)] = (now, [dict(h) for h in hits])

    async def search(self, query: str, num_results: int = 5) -> List[SearchHit]:
        """
        Execute a web search and return a list of {title, link, snippet} dicts.

        Uses TTL caching, bounded concurrency, and falls back down the provider
        chain cleanly on failure rather than raising.
        """
        if not query or not query.strip():
            return []

        cached = self._get_cached(query, num_results)
        if cached is not None:
            return cached

        provider = settings.SEARCH_PROVIDER
        results: List[SearchHit] = []

        async with self._semaphore:
            if provider == "auto":
                if settings.SERPER_API_KEY:
                    results = await self._search_serper(query, num_results)
                if not results and settings.BRAVE_API_KEY:
                    results = await self._search_brave(query, num_results)
                if not results:
                    results = await self._search_ddg(query, num_results)
            elif provider == "serper":
                if settings.SERPER_API_KEY:
                    results = await self._search_serper(query, num_results)
                if not results:
                    results = await self._search_ddg(query, num_results)
            elif provider == "brave":
                if settings.BRAVE_API_KEY:
                    results = await self._search_brave(query, num_results)
                if not results:
                    results = await self._search_ddg(query, num_results)
            else:
                results = await self._search_ddg(query, num_results)

        if results:
            self._put_cache(query, num_results, results)
        return results

    # ------------------------------------------------------------------
    # Provider implementations (with HTTP 429 retry / backoff)
    # ------------------------------------------------------------------

    async def _search_serper(self, query: str, num: int) -> List[SearchHit]:
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(
                        "https://google.serper.dev/search",
                        headers={
                            "X-API-KEY": settings.SERPER_API_KEY,  # type: ignore[arg-type]
                            "Content-Type": "application/json",
                        },
                        json={"q": query, "num": num},
                    )
                    if resp.status_code == 429 and attempt == 0:
                        await asyncio.sleep(0.8)
                        continue
                    resp.raise_for_status()
                    return [
                        {
                            "title": str(item.get("title", "")),
                            "link": str(item.get("link", "")),
                            "snippet": str(item.get("snippet", "")),
                        }
                        for item in resp.json().get("organic", [])
                    ]
            except httpx.HTTPStatusError as exc:
                logger.warning("Serper API HTTP %s.", exc.response.status_code)
                break
            except Exception as exc:
                logger.warning("Serper search error: %s", type(exc).__name__)
                if attempt == 0:
                    await asyncio.sleep(0.4)
        return []

    async def _search_brave(self, query: str, num: int) -> List[SearchHit]:
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        headers={
                            "X-Subscription-Token": settings.BRAVE_API_KEY,  # type: ignore[arg-type]
                            "Accept": "application/json",
                        },
                        params={"q": query, "count": num},
                    )
                    if resp.status_code == 429 and attempt == 0:
                        await asyncio.sleep(0.8)
                        continue
                    resp.raise_for_status()
                    return [
                        {
                            "title": str(item.get("title", "")),
                            "link": str(item.get("url", "")),
                            "snippet": str(item.get("description", "")),
                        }
                        for item in resp.json().get("web", {}).get("results", [])
                    ]
            except httpx.HTTPStatusError as exc:
                logger.warning("Brave API HTTP %s.", exc.response.status_code)
                break
            except Exception as exc:
                logger.warning("Brave search error: %s", type(exc).__name__)
                if attempt == 0:
                    await asyncio.sleep(0.4)
        return []

    async def _search_ddg(self, query: str, num: int) -> List[SearchHit]:
        """DuckDuckGo via `ddgs`, offloaded to thread with retry on throttling."""

        def _sync_search() -> List[SearchHit]:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS  # type: ignore[no-redef]
            with DDGS(timeout=settings.SEARCH_TIMEOUT_SECONDS) as ddgs:
                hits = list(ddgs.text(query, max_results=num))
            return [
                {
                    "title": str(item.get("title", "")),
                    "link": str(item.get("href", "")),
                    "snippet": str(item.get("body", "")),
                }
                for item in hits
            ]

        for attempt in range(2):
            try:
                return await asyncio.to_thread(_sync_search)
            except Exception as exc:
                logger.warning("DDG search error (attempt %d): %s", attempt + 1, type(exc).__name__)
                if attempt == 0:
                    await asyncio.sleep(0.6)
        return []


search_service = SearchService()
