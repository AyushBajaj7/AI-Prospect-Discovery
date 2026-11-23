"""Facebook public profile / page adapter."""
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent


class FacebookAdapter(BasePlatformAdapter):
    platform_name = "Facebook"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = ""
        if intent.location.city:
            loc = f'"{intent.location.city}"'
        elif intent.location.country:
            loc = f'"{intent.location.country}"'

        audience = f'"{intent.audience}"'
        return (
            f"site:facebook.com {audience} {loc} "
            f"-inurl:/groups/ -inurl:/story.php -inurl:/events/"
        ).strip()
