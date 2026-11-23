from typing import List
from urllib.parse import urlparse
from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent

class GoogleAdapter(BasePlatformAdapter):
    platform_name: str = "Google"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = f'"{intent.location.city}"' if intent.location.city else (f'"{intent.location.country}"' if intent.location.country else "")
        return f'"{intent.audience}" {loc} (founder OR owner OR "about me" OR portfolio OR official website) -site:linkedin.com -site:twitter.com -site:x.com -site:facebook.com -site:instagram.com -site:youtube.com'

    def validate_url(self, url: str) -> bool:
        if not url:
            return False
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Filter out major search/social/aggregation sites that have their own adapters or aren't personal/business sites
        excluded = [
            "google.", "bing.", "yahoo.", "wikipedia.org", "yelp.", "tripadvisor.",
            "linkedin.com", "x.com", "twitter.com", "facebook.com", "instagram.com",
            "youtube.com", "threads.net", "reddit.com", "quora.com", "amazon.", "ebay."
        ]
        if any(exc in domain for exc in excluded):
            return False
        return bool(parsed.scheme in ["http", "https"] and len(domain) > 3)
