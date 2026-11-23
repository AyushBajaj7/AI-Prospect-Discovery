from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent

class XAdapter(BasePlatformAdapter):
    platform_name: str = "X"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = f'"{intent.location.city}"' if intent.location.city else (f'"{intent.location.country}"' if intent.location.country else "")
        return f'site:x.com "{intent.audience}" {loc} -inurl:status -inurl:hashtag'.strip()
