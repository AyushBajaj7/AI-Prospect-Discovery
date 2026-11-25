"""Facebook public profile / page adapter."""
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent


class FacebookAdapter(BasePlatformAdapter):
    platform_name = "Facebook"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = self._location_query_term(intent)
        audience = self._audience_query_term(intent)
        return (
            f"site:facebook.com {audience} {loc} "
            f"-inurl:/groups/ -inurl:/story.php -inurl:/events/"
        ).strip()
