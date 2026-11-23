"""X (formerly Twitter) public profile adapter."""
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent


class XAdapter(BasePlatformAdapter):
    platform_name = "X"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = ""
        if intent.location.city:
            loc = f'"{intent.location.city}"'
        elif intent.location.country:
            loc = f'"{intent.location.country}"'

        audience = f'"{intent.audience}"'
        return f"site:x.com {audience} {loc} -inurl:/status/ -inurl:/hashtag".strip()
