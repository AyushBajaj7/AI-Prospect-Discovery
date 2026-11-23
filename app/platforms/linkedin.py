from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent

class LinkedInAdapter(BasePlatformAdapter):
    platform_name: str = "LinkedIn"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = f'"{intent.location.city}"' if intent.location.city else (f'"{intent.location.country}"' if intent.location.country else "")
        return f'site:linkedin.com/in/ "{intent.audience}" {loc}'.strip()
