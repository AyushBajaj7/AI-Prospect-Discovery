from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent

class InstagramAdapter(BasePlatformAdapter):
    platform_name: str = "Instagram"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = f'"{intent.location.city}"' if intent.location.city else (f'"{intent.location.country}"' if intent.location.country else "")
        return f'site:instagram.com "{intent.audience}" {loc} -inurl:/p/ -inurl:/reel/ -inurl:/stories/'.strip()
