from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent

class FacebookAdapter(BasePlatformAdapter):
    platform_name: str = "Facebook"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = f'"{intent.location.city}"' if intent.location.city else (f'"{intent.location.country}"' if intent.location.country else "")
        return f'site:facebook.com "{intent.audience}" {loc} -inurl:/groups/ -inurl:/story.php -inurl:/events/'.strip()
