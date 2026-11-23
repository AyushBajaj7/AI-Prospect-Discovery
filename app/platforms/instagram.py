"""Instagram public profile adapter."""
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent


class InstagramAdapter(BasePlatformAdapter):
    platform_name = "Instagram"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = ""
        if intent.location.city:
            loc = f'"{intent.location.city}"'
        elif intent.location.country:
            loc = f'"{intent.location.country}"'

        audience = f'"{intent.audience}"'
        return (
            f"site:instagram.com {audience} {loc} "
            f"-inurl:/p/ -inurl:/reel/ -inurl:/stories/ -inurl:/explore/"
        ).strip()
