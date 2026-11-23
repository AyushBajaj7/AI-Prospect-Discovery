"""
Search service – multi-provider web search with graceful fallback chain.

Provider priority (auto mode): Serper → Brave → DuckDuckGo (free, no key).

Security notes
--------------
* API keys are read from settings; never logged or echoed.
* All results are dicts with str values – no code execution.
* httpx is used with explicit timeouts to prevent hanging connections.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

SearchHit = Dict[str, str]  # keys: title, link, snippet


class SearchService:

    def __init__(self) -> None:
        self._timeout = httpx.Timeout(settings.SEARCH_TIMEOUT_SECONDS)

    async def search(self, query: str, num_results: int = 5) -> List[SearchHit]:
        """
        Execute a web search and return a list of {title, link, snippet} dicts.

        Falls back down the provider chain on failure rather than raising.
        """
        if not query or not query.strip():
            return []

        provider = settings.SEARCH_PROVIDER

        if provider == "auto":
            if settings.SERPER_API_KEY:
                provider = "serper"
            elif settings.BRAVE_API_KEY:
                provider = "brave"
            else:
                provider = "ddg"

        results: List[SearchHit] = []

        if provider == "serper" and settings.SERPER_API_KEY:
            results = await self._search_serper(query, num_results)

        elif provider == "brave" and settings.BRAVE_API_KEY:
            results = await self._search_brave(query, num_results)

        # Always fall through to DDG if the preferred provider returns nothing
        if not results:
            results = await self._search_ddg(query, num_results)

        return results

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    async def _search_serper(self, query: str, num: int) -> List[SearchHit]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    "https://google.serper.dev/search",
                    headers={
                        # Key is set via header, never in the URL
                        "X-API-KEY": settings.SERPER_API_KEY,  # type: ignore[arg-type]
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": num},
                )
                resp.raise_for_status()
                return [
                    {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    }
                    for item in resp.json().get("organic", [])
                ]
        except httpx.HTTPStatusError as exc:
            logger.warning("Serper API HTTP %s for query.", exc.response.status_code)
        except Exception as exc:
            logger.warning("Serper search error: %s", type(exc).__name__)
        return []

    async def _search_brave(self, query: str, num: int) -> List[SearchHit]:
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
                resp.raise_for_status()
                return [
                    {
                        "title": item.get("title", ""),
                        "link": item.get("url", ""),
                        "snippet": item.get("description", ""),
                    }
                    for item in resp.json().get("web", {}).get("results", [])
                ]
        except httpx.HTTPStatusError as exc:
            logger.warning("Brave API HTTP %s for query.", exc.response.status_code)
        except Exception as exc:
            logger.warning("Brave search error: %s", type(exc).__name__)
        return []

    async def _search_ddg(self, query: str, num: int) -> List[SearchHit]:
        """DuckDuckGo via the `ddgs` package, run in a thread to avoid blocking the event loop."""

        def _sync_search() -> List[SearchHit]:
            try:
                try:
                    from ddgs import DDGS
                except ImportError:
                    from duckduckgo_search import DDGS  # type: ignore[no-redef]
                with DDGS(timeout=settings.SEARCH_TIMEOUT_SECONDS) as ddgs:
                    hits = list(ddgs.text(query, max_results=num))
                return [
                    {
                        "title": item.get("title", ""),
                        "link": item.get("href", ""),
                        "snippet": item.get("body", ""),
                    }
                    for item in hits
                ]
            except Exception as exc:
                logger.warning("DDG search error: %s", type(exc).__name__)
                return []

        return await asyncio.to_thread(_sync_search)


search_service = SearchService()
