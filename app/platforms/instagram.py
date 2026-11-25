"""Instagram public profile adapter."""
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent


class InstagramAdapter(BasePlatformAdapter):
    platform_name = "Instagram"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = self._location_query_term(intent)
        audience = self._audience_query_term(intent)
        return (
            f"site:instagram.com {audience} {loc} "
            f"-inurl:/p/ -inurl:/reel/ -inurl:/stories/ -inurl:/explore/"
        ).strip()
