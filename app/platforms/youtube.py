"""YouTube public channel / creator adapter."""
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent


class YouTubeAdapter(BasePlatformAdapter):
    platform_name = "YouTube"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = ""
        if intent.location.city:
            loc = f'"{intent.location.city}"'
        elif intent.location.country:
            loc = f'"{intent.location.country}"'

        topic = intent.interest or intent.audience
        return (
            f'site:youtube.com "{topic}" {loc} '
            f"-inurl:/watch -inurl:/playlist -inurl:/shorts"
        ).strip()
