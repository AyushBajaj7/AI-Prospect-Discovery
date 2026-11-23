from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent

class ThreadsAdapter(BasePlatformAdapter):
    platform_name: str = "Threads"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = f'"{intent.location.city}"' if intent.location.city else (f'"{intent.location.country}"' if intent.location.country else "")
        return f'site:threads.net/@ "{intent.audience}" {loc} -inurl:/post/'.strip()
