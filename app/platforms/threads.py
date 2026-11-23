"""Threads public profile adapter."""
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent


class ThreadsAdapter(BasePlatformAdapter):
    platform_name = "Threads"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = ""
        if intent.location.city:
            loc = f'"{intent.location.city}"'
        elif intent.location.country:
            loc = f'"{intent.location.country}"'

        audience = f'"{intent.audience}"'
        return f"site:threads.net {audience} {loc} -inurl:/post/".strip()
