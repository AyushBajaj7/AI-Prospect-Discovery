import time
from typing import Dict, Any, Optional, Tuple
from app.schemas import StructuredIntent
from app.services.llm_service import llm_service
from app.config import settings

class QueryService:
    def __init__(self):
        # In-memory cache mapping normalized query string -> (timestamp, StructuredIntent)
        self._cache: Dict[str, Tuple[float, StructuredIntent]] = {}

    async def get_structured_intent(self, query: str) -> StructuredIntent:
        """Extracts structured intent with in-memory caching to optimize API cost and latency."""
        norm_key = query.strip().lower()
        now = time.time()
        
        if norm_key in self._cache:
            cached_time, intent = self._cache[norm_key]
            if now - cached_time < settings.CACHE_TTL_SECONDS:
                return intent

        intent = await llm_service.parse_query(query)
        self._cache[norm_key] = (now, intent)
        return intent

    def generate_platform_queries(self, intent: StructuredIntent) -> Dict[str, str]:
        """Generates platform-specific targeted search queries for public profile discovery."""
        loc_str = ""
        if intent.location.city:
            loc_str = f'"{intent.location.city}"'
        elif intent.location.country:
            loc_str = f'"{intent.location.country}"'

        audience_term = f'"{intent.audience}"' if intent.audience else ""
        interest_term = f'"{intent.interest}"' if intent.interest and intent.interest != intent.audience else ""

        primary_terms = f"{audience_term} {loc_str}".strip()

        return {
            "Google": f"{primary_terms} (profile OR about OR founder OR official)",
            "LinkedIn": f'site:linkedin.com/in/ {primary_terms}',
            "X": f'site:x.com {primary_terms} -inurl:status',
            "YouTube": f'site:youtube.com/@ OR site:youtube.com/channel/ {primary_terms}',
            "Instagram": f'site:instagram.com {primary_terms} -inurl:/p/ -inurl:/reel/',
            "Facebook": f'site:facebook.com {primary_terms} -inurl:/story.php -inurl:/groups/',
            "Threads": f'site:threads.net/@ {primary_terms}',
            "WhatsApp": f'{primary_terms} ("wa.me" OR "api.whatsapp.com")',
            "Email": f'{primary_terms} ("@gmail.com" OR "contact" OR "email")',
        }

query_service = QueryService()
