"""X (formerly Twitter) public profile adapter."""
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent


class XAdapter(BasePlatformAdapter):
    platform_name = "X"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = self._location_query_term(intent)
        audience = self._audience_query_term(intent)
        return f"site:x.com {audience} {loc} -inurl:/status/ -inurl:/hashtag".strip()
