"""LinkedIn public profile adapter."""
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent


class LinkedInAdapter(BasePlatformAdapter):
    platform_name = "LinkedIn"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = self._location_query_term(intent)
        audience = self._audience_query_term(intent)
        return f"site:linkedin.com/in/ {audience} {loc}".strip()
