import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

class SearchResult(Dict[str, Any]):
    title: str
    link: str
    snippet: str

class SearchService:
    def __init__(self):
        self.timeout = settings.SEARCH_TIMEOUT_SECONDS

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Executes a web search across configured providers: Serper, Brave, or DuckDuckGo fallback."""
        provider = settings.SEARCH_PROVIDER.lower()
        
        if provider == "auto":
            if settings.SERPER_API_KEY:
                provider = "serper"
            elif settings.BRAVE_API_KEY:
                provider = "brave"
            else:
                provider = "ddg"

        results = []
        if provider == "serper" and settings.SERPER_API_KEY:
            results = await self._search_serper(query, num_results)
        elif provider == "brave" and settings.BRAVE_API_KEY:
            results = await self._search_brave(query, num_results)
        
        # Fallback to DDG if no results or if provider is ddg
        if not results:
            results = await self._search_ddg(query, num_results)
            
        return results

    async def _search_serper(self, query: str, num: int) -> List[Dict[str, str]]:
        try:
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": settings.SERPER_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {"q": query, "num": num}
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    organic = data.get("organic", [])
                    return [
                        {
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", "")
                        }
                        for item in organic
                    ]
                else:
                    logger.warning(f"Serper API error: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.warning(f"Serper search failed for '{query}': {e}")
        return []

    async def _search_brave(self, query: str, num: int) -> List[Dict[str, str]]:
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "X-Subscription-Token": settings.BRAVE_API_KEY,
                "Accept": "application/json"
            }
            params = {"q": query, "count": num}
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    web_results = data.get("web", {}).get("results", [])
                    return [
                        {
                            "title": item.get("title", ""),
                            "link": item.get("url", ""),
                            "snippet": item.get("description", "")
                        }
                        for item in web_results
                    ]
                else:
                    logger.warning(f"Brave API error: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.warning(f"Brave search failed for '{query}': {e}")
        return []

    async def _search_ddg(self, query: str, num: int) -> List[Dict[str, str]]:
        """Zero-dependency DuckDuckGo search fallback executed in a thread pool."""
        try:
            def _sync_ddg():
                try:
                    try:
                        from ddgs import DDGS
                    except ImportError:
                        from duckduckgo_search import DDGS
                    ddgs = DDGS(timeout=self.timeout)
                    items = list(ddgs.text(query, max_results=num))
                    return [
                        {
                            "title": item.get("title", ""),
                            "link": item.get("href", ""),
                            "snippet": item.get("body", "")
                        }
                        for item in items
                    ]
                except Exception as inner_e:
                    logger.warning(f"DDGS error: {inner_e}")
                    return []

            return await asyncio.to_thread(_sync_ddg)
        except Exception as e:
            logger.warning(f"DDG search error for query '{query}': {e}")
            return []

search_service = SearchService()
