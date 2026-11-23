"""
Google / generic-web adapter.

Discovers personal websites, portfolios, and "about" pages that are not
covered by specific social-platform adapters.
"""
from urllib.parse import urlparse

from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent

# Social platforms already handled by dedicated adapters
_EXCLUDED_DOMAINS = frozenset(
    [
        "linkedin.com", "x.com", "twitter.com", "instagram.com",
        "facebook.com", "youtube.com", "threads.net",
        "google.com", "bing.com", "wikipedia.org",
        "reddit.com", "quora.com", "medium.com",
        "yelp.com", "tripadvisor.com", "glassdoor.com",
        "amazon.com", "ebay.com", "etsy.com",
    ]
)


class GoogleAdapter(BasePlatformAdapter):
    platform_name = "Google"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = ""
        if intent.location.city:
            loc = f'"{intent.location.city}"'
        elif intent.location.country:
            loc = f'"{intent.location.country}"'

        audience = f'"{intent.audience}"'
        # Focus on personal / business sites with "about", "portfolio", or "contact"
        return (
            f"{audience} {loc} "
            f'(site:about.me OR "about me" OR portfolio OR founder OR "official website") '
            f"-site:linkedin.com -site:x.com -site:twitter.com "
            f"-site:facebook.com -site:instagram.com -site:youtube.com"
        ).strip()

    def validate_url(self, url: str) -> bool:
        if not url:
            return False
        try:
            netloc = urlparse(url.lower()).netloc
            # Strip www.
            base = netloc.removeprefix("www.")
            if any(base == excl or base.endswith("." + excl) for excl in _EXCLUDED_DOMAINS):
                return False
        except Exception:
            return False
        from app.utils.validators import is_valid_profile_url
        return is_valid_profile_url("Google", url)
