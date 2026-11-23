from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent

class YouTubeAdapter(BasePlatformAdapter):
    platform_name: str = "YouTube"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = f'"{intent.location.city}"' if intent.location.city else (f'"{intent.location.country}"' if intent.location.country else "")
        topic = intent.interest if intent.interest else intent.audience
        return f'site:youtube.com/@ OR site:youtube.com/channel/ "{topic}" {loc}'.strip()
