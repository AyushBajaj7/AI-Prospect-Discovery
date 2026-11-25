"""Threads public profile adapter."""
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent


class ThreadsAdapter(BasePlatformAdapter):
    platform_name = "Threads"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = self._location_query_term(intent)
        audience = self._audience_query_term(intent)
        return f"site:threads.net {audience} {loc} -inurl:/post/".strip()
