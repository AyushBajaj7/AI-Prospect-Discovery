"""
Google / generic-web adapter.

Discovers personal websites, portfolios, and "about" pages that are not
covered by specific social-platform adapters, excluding news articles,
event listings, and generic directories.
"""
import re
from urllib.parse import urlparse

from app.platforms.base import BasePlatformAdapter
from app.schemas import StructuredIntent
from app.utils.validators import is_valid_profile_url

# Social platforms, aggregators, news outlets & event directories excluded from Google hits
_EXCLUDED_DOMAINS = frozenset(
    {
        "linkedin.com", "x.com", "twitter.com", "instagram.com",
        "facebook.com", "youtube.com", "threads.net",
        "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
        "wikipedia.org", "reddit.com", "quora.com", "medium.com",
        "yelp.com", "tripadvisor.com", "glassdoor.com",
        "indeed.com", "pinterest.com", "tiktok.com",
        "amazon.com", "ebay.com", "etsy.com",
        "meetup.com", "eventbrite.com", "forbes.com", "bloomberg.com",
        "techcrunch.com", "businessinsider.com", "bbc.com", "bbc.co.uk",
        "reuters.com", "theguardian.com", "nytimes.com",
    }
)

# URL patterns indicating dated news articles / press stories rather than profiles
_NEWS_OR_ARTICLE_PATH_RE = re.compile(
    r"/(?:20\d{2}/\d{2}(?:/\d{2})?|news|articles?|blog|press|stories|events?)/",
    re.IGNORECASE,
)


class GoogleAdapter(BasePlatformAdapter):
    platform_name = "Google"

    def build_query(self, intent: StructuredIntent) -> str:
        loc = self._location_query_term(intent)
        audience = self._audience_query_term(intent)
        return (
            f"{audience} {loc} "
            f'(site:about.me OR "about me" OR portfolio OR founder OR "official website") '
            f"-site:linkedin.com -site:x.com -site:twitter.com "
            f"-site:facebook.com -site:instagram.com -site:youtube.com "
            f"-site:meetup.com -site:eventbrite.com"
        ).strip()

    def validate_url(self, url: str) -> bool:
        if not url:
            return False
        try:
            parsed = urlparse(url.strip())
            hostname = (parsed.hostname or "").lower()
            if not hostname:
                return False
            if hostname.startswith("news."):
                return False
            if any(hostname == excl or hostname.endswith("." + excl) for excl in _EXCLUDED_DOMAINS):
                return False
            if _NEWS_OR_ARTICLE_PATH_RE.search(parsed.path or ""):
                return False
        except Exception:
            return False
        return is_valid_profile_url("Google", url)
