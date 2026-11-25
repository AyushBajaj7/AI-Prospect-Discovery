"""YouTube public channel / creator adapter."""
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent


class YouTubeAdapter(BasePlatformAdapter):
    platform_name = "YouTube"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = self._location_query_term(intent)
        topic_clause = self._topic_query_term(intent)
        return (
            f"site:youtube.com {topic_clause} {loc} "
            f"-inurl:/watch -inurl:/playlist -inurl:/shorts"
        ).strip()
