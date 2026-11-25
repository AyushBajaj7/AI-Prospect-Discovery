"""YouTube public channel / creator adapter."""
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent


class YouTubeAdapter(BasePlatformAdapter):
    platform_name = "YouTube"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = self._location_query_term(intent)
        topic = intent.interest or intent.audience
        return (
            f'site:youtube.com "{topic}" {loc} '
            f"-inurl:/watch -inurl:/playlist -inurl:/shorts"
        ).strip()
