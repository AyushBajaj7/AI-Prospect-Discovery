"""
Search service – multi-provider web search with shared HTTPX lifespan client,
bounded LRU+TTL caching, concurrency semaphore, explicit outcome tracking,
and graceful provider fallback chain.

Provider priority (auto mode): Serper → Brave → DuckDuckGo (free, no key).
Only non-empty successful results are cached; rate-limit / timeout / error
outcomes are explicitly returned via `SearchStatus` so callers can distinguish
zero results from upstream failures.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx

from app.config import settings
from app.schemas import SearchStatus

logger = logging.getLogger(__name__)

SearchHit = Dict[str, str]  # keys: title, link, snippet


@dataclass
class SearchOutcome:
    hits: List[SearchHit]
    status: SearchStatus
    provider: str = "none"
    cached: bool = False
    http_status: Optional[int] = None
    exception_type: Optional[str] = None
    failure_reason: Optional[str] = None
    retry_count: int = 0


class SearchService:

    def __init__(self) -> None:
        self._timeout = httpx.Timeout(settings.SEARCH_TIMEOUT_SECONDS)
        self._limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(settings.SEARCH_MAX_CONCURRENCY)
        # Caches ONLY non-empty successful searches
        self._cache: OrderedDict[Tuple[str, int], Tuple[float, List[SearchHit], str]] = OrderedDict()

    async def startup(self) -> None:
        """Initialize the shared HTTPX connection pool during FastAPI lifespan startup."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout, limits=self._limits)

    async def shutdown(self) -> None:
        """Close the shared HTTPX connection pool cleanly on shutdown."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _get_cached(self, query: str, num_results: int) -> Optional[SearchOutcome]:
        key = (query.strip(), num_results)
        now = time.monotonic()
        if key in self._cache:
            cached_at, hits, provider = self._cache[key]
            if now - cached_at < settings.CACHE_TTL_SECONDS:
                self._cache.move_to_end(key)
                return SearchOutcome(
                    hits=[dict(h) for h in hits],
                    status=SearchStatus.SUCCESS,
                    provider=provider,
                    cached=True,
                )
            del self._cache[key]
        return None

    def _put_cache(
        self, query: str, num_results: int, hits: List[SearchHit], provider: str
    ) -> None:
        if not hits:
            return  # Never cache empty or failed results
        now = time.monotonic()
        expired = [
            k for k, (ts, _, _) in self._cache.items()
            if now - ts >= settings.CACHE_TTL_SECONDS
        ]
        for k in expired:
            del self._cache[k]
        while len(self._cache) >= settings.CACHE_MAX_SIZE:
            self._cache.popitem(last=False)
        self._cache[(query.strip(), num_results)] = (now, [dict(h) for h in hits], provider)

    async def _run_http_call(self, fn):
        """Use shared lifespan client if open, else scope a temporary client cleanly."""
        if self._client is not None and not self._client.is_closed:
            return await fn(self._client)
        async with httpx.AsyncClient(timeout=self._timeout, limits=self._limits) as temp_client:
            return await fn(temp_client)

    @staticmethod
    def _aggregate_failure_status(statuses: List[SearchStatus]) -> SearchStatus:
        """
        Aggregate status across attempted providers when all produce 0 hits so
        upstream failures (RATE_LIMITED / TIMEOUT / PROVIDER_ERROR) are never
        masked by a final fallback returning NO_RESULTS.
        """
        priority = (
            SearchStatus.RATE_LIMITED,
            SearchStatus.TIMEOUT,
            SearchStatus.PROVIDER_ERROR,
            SearchStatus.NO_RESULTS,
        )
        for st in priority:
            if st in statuses:
                return st
        return SearchStatus.NO_RESULTS

    async def search_with_status(
        self, query: str, num_results: int = 5
    ) -> SearchOutcome:
        """
        Execute a search and return `SearchOutcome` containing both `hits` and
        an aggregated `SearchStatus` across the entire fallback chain.
        """
        if not query or not query.strip():
            return SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="none")

        cached = self._get_cached(query, num_results)
        if cached is not None:
            return cached

        provider_cfg = settings.SEARCH_PROVIDER
        candidates: List[str] = []

        if provider_cfg == "auto":
            if settings.SERPER_API_KEY:
                candidates.append("serper")
            if settings.BRAVE_API_KEY:
                candidates.append("brave")
            candidates.append("ddg")
        elif provider_cfg == "serper":
            if settings.SERPER_API_KEY:
                candidates.append("serper")
            candidates.append("ddg")
        elif provider_cfg == "brave":
            if settings.BRAVE_API_KEY:
                candidates.append("brave")
            candidates.append("ddg")
        else:
            candidates.append("ddg")

        async with self._semaphore:
            attempted_outcomes: List[SearchOutcome] = []
            for prov in candidates:
                if prov == "serper":
                    outcome = await self._search_serper(query, num_results)
                elif prov == "brave":
                    outcome = await self._search_brave(query, num_results)
                else:
                    outcome = await self._search_ddg(query, num_results)

                if outcome.hits:
                    self._put_cache(query, num_results, outcome.hits, outcome.provider)
                    return outcome
                attempted_outcomes.append(outcome)

            agg_status = self._aggregate_failure_status([o.status for o in attempted_outcomes])
            rep = next((o for o in attempted_outcomes if o.status == agg_status), None)
            if rep is None and attempted_outcomes:
                rep = attempted_outcomes[-1]
            return SearchOutcome(
                hits=[],
                status=agg_status,
                provider=rep.provider if rep else "none",
                http_status=rep.http_status if rep else None,
                exception_type=rep.exception_type if rep else None,
                failure_reason=rep.failure_reason if rep else None,
                retry_count=sum(o.retry_count for o in attempted_outcomes),
            )

    async def search(self, query: str, num_results: int = 5) -> List[SearchHit]:
        """Backward-compatible helper returning hits list."""
        outcome = await self.search_with_status(query, num_results=num_results)
        return outcome.hits

    # ------------------------------------------------------------------
    # Provider implementations (HTTP 429 & timeout-aware)
    # ------------------------------------------------------------------

    async def _search_serper(self, query: str, num: int) -> SearchOutcome:
        last_status = SearchStatus.PROVIDER_ERROR
        last_http_status: Optional[int] = None
        last_exc_type: Optional[str] = None
        last_reason: Optional[str] = None
        for attempt in range(2):
            try:
                async def _do(client: httpx.AsyncClient):
                    return await client.post(
                        "https://google.serper.dev/search",
                        headers={
                            "X-API-KEY": settings.SERPER_API_KEY,  # type: ignore[arg-type]
                            "Content-Type": "application/json",
                        },
                        json={"q": query, "num": num},
                    )

                resp = await self._run_http_call(_do)
                if resp.status_code == 429:
                    last_status = SearchStatus.RATE_LIMITED
                    last_http_status = 429
                    last_reason = "http_429_rate_limited"
                    if attempt == 0:
                        await asyncio.sleep(0.8)
                        continue
                    return SearchOutcome(
                        hits=[],
                        status=SearchStatus.RATE_LIMITED,
                        provider="serper",
                        http_status=429,
                        failure_reason="http_429_rate_limited",
                        retry_count=attempt,
                    )

                resp.raise_for_status()
                hits = [
                    {
                        "title": str(item.get("title", "")),
                        "link": str(item.get("link", "")),
                        "snippet": str(item.get("snippet", "")),
                    }
                    for item in resp.json().get("organic", [])
                ]
                status = SearchStatus.SUCCESS if hits else SearchStatus.NO_RESULTS
                return SearchOutcome(
                    hits=hits,
                    status=status,
                    provider="serper",
                    http_status=resp.status_code,
                    retry_count=attempt,
                )

            except httpx.TimeoutException as exc:
                logger.warning("Serper search timed out.")
                return SearchOutcome(
                    hits=[],
                    status=SearchStatus.TIMEOUT,
                    provider="serper",
                    exception_type=type(exc).__name__,
                    failure_reason="timeout",
                    retry_count=attempt,
                )
            except httpx.HTTPStatusError as exc:
                sc = exc.response.status_code
                logger.warning("Serper API HTTP %s.", sc)
                return SearchOutcome(
                    hits=[],
                    status=SearchStatus.PROVIDER_ERROR,
                    provider="serper",
                    http_status=sc,
                    exception_type=type(exc).__name__,
                    failure_reason=f"http_{sc}",
                    retry_count=attempt,
                )
            except Exception as exc:
                logger.warning("Serper search error: %s", type(exc).__name__)
                last_status = SearchStatus.PROVIDER_ERROR
                last_exc_type = type(exc).__name__
                last_reason = "request_error"
                if attempt == 0:
                    await asyncio.sleep(0.4)
        return SearchOutcome(
            hits=[],
            status=last_status,
            provider="serper",
            http_status=last_http_status,
            exception_type=last_exc_type,
            failure_reason=last_reason,
            retry_count=1,
        )

    async def _search_brave(self, query: str, num: int) -> SearchOutcome:
        last_status = SearchStatus.PROVIDER_ERROR
        last_http_status: Optional[int] = None
        last_exc_type: Optional[str] = None
        last_reason: Optional[str] = None
        for attempt in range(2):
            try:
                async def _do(client: httpx.AsyncClient):
                    return await client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        headers={
                            "X-Subscription-Token": settings.BRAVE_API_KEY,  # type: ignore[arg-type]
                            "Accept": "application/json",
                        },
                        params={"q": query, "count": num},
                    )

                resp = await self._run_http_call(_do)
                if resp.status_code == 429:
                    last_status = SearchStatus.RATE_LIMITED
                    last_http_status = 429
                    last_reason = "http_429_rate_limited"
                    if attempt == 0:
                        await asyncio.sleep(0.8)
                        continue
                    return SearchOutcome(
                        hits=[],
                        status=SearchStatus.RATE_LIMITED,
                        provider="brave",
                        http_status=429,
                        failure_reason="http_429_rate_limited",
                        retry_count=attempt,
                    )

                resp.raise_for_status()
                hits = [
                    {
                        "title": str(item.get("title", "")),
                        "link": str(item.get("url", "")),
                        "snippet": str(item.get("description", "")),
                    }
                    for item in resp.json().get("web", {}).get("results", [])
                ]
                status = SearchStatus.SUCCESS if hits else SearchStatus.NO_RESULTS
                return SearchOutcome(
                    hits=hits,
                    status=status,
                    provider="brave",
                    http_status=resp.status_code,
                    retry_count=attempt,
                )

            except httpx.TimeoutException as exc:
                logger.warning("Brave search timed out.")
                return SearchOutcome(
                    hits=[],
                    status=SearchStatus.TIMEOUT,
                    provider="brave",
                    exception_type=type(exc).__name__,
                    failure_reason="timeout",
                    retry_count=attempt,
                )
            except httpx.HTTPStatusError as exc:
                sc = exc.response.status_code
                logger.warning("Brave API HTTP %s.", sc)
                return SearchOutcome(
                    hits=[],
                    status=SearchStatus.PROVIDER_ERROR,
                    provider="brave",
                    http_status=sc,
                    exception_type=type(exc).__name__,
                    failure_reason=f"http_{sc}",
                    retry_count=attempt,
                )
            except Exception as exc:
                logger.warning("Brave search error: %s", type(exc).__name__)
                last_status = SearchStatus.PROVIDER_ERROR
                last_exc_type = type(exc).__name__
                last_reason = "request_error"
                if attempt == 0:
                    await asyncio.sleep(0.4)
        return SearchOutcome(
            hits=[],
            status=last_status,
            provider="brave",
            http_status=last_http_status,
            exception_type=last_exc_type,
            failure_reason=last_reason,
            retry_count=1,
        )

    @staticmethod
    def _classify_ddg_exception(exc: Exception) -> Tuple[SearchStatus, Optional[int], str]:
        """
        Classify DDG/ddgs failures using typed exceptions, HTTP status codes, and
        explicit anti-bot / challenge signatures without conflating bot challenges
        with confirmed HTTP 429 rate limiting.
        """
        exc_name = type(exc).__name__.lower()
        exc_str = str(exc).lower()

        http_status: Optional[int] = getattr(exc, "status_code", None)
        if http_status is None:
            resp = getattr(exc, "response", None)
            if resp is not None:
                http_status = getattr(resp, "status_code", None)
        if http_status is None:
            m = re.search(r"\b(?:status[_\s=:]+|http\s+)([245]\d{2})\b", exc_str)
            if m:
                http_status = int(m.group(1))
            elif " 429" in exc_str or exc_str.startswith("429"):
                http_status = 429

        if "timeout" in exc_name or "timeout" in exc_str or isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return SearchStatus.TIMEOUT, http_status, "timeout"

        if (
            "ratelimit" in exc_name
            or http_status == 429
            or "429" in exc_str
            or "rate limit" in exc_str
            or "too many requests" in exc_str
        ):
            return SearchStatus.RATE_LIMITED, http_status or 429, "rate_limited"

        if any(tok in exc_str for tok in ("captcha", "challenge", "bot", "202", "403", "forbidden")):
            return SearchStatus.PROVIDER_ERROR, http_status, "anti_bot_or_challenge"

        if "no results found" in exc_str:
            return SearchStatus.NO_RESULTS, http_status, "no_results_exception"

        return SearchStatus.PROVIDER_ERROR, http_status, "ddgs_error"

    async def _search_ddg(self, query: str, num: int) -> SearchOutcome:
        """DuckDuckGo via `ddgs`, offloaded to thread with typed error & HTTP classification."""

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

        last_status = SearchStatus.PROVIDER_ERROR
        last_http_status: Optional[int] = None
        last_exc_type: Optional[str] = None
        last_reason: Optional[str] = None
        retries = 0
        for attempt in range(2):
            try:
                hits = await asyncio.to_thread(_sync_search)
                status = SearchStatus.SUCCESS if hits else SearchStatus.NO_RESULTS
                return SearchOutcome(
                    hits=hits,
                    status=status,
                    provider="ddg",
                    retry_count=attempt,
                )
            except Exception as exc:
                last_status, last_http_status, last_reason = self._classify_ddg_exception(exc)
                last_exc_type = type(exc).__name__
                logger.warning(
                    "DDG search error (attempt %d): type=%s http=%s reason=%s status=%s",
                    attempt + 1,
                    last_exc_type,
                    last_http_status,
                    last_reason,
                    last_status.value,
                )
            if attempt == 0:
                retries = 1
                await asyncio.sleep(0.6)

        return SearchOutcome(
            hits=[],
            status=last_status,
            provider="ddg",
            http_status=last_http_status,
            exception_type=last_exc_type,
            failure_reason=last_reason,
            retry_count=retries,
        )


search_service = SearchService()
