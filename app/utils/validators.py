"""
Validators – URL and contact-detail validation.

All validation is local (no external HTTP calls) and fast.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Supports multi-part TLDs (e.g. .co.uk, .com.au) and standard TLDs.
# Domain: one or more labels separated by dots, each label [a-z0-9-]+,
# final label (TLD) is 2–24 alpha chars.
_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9_.+\-]+"          # local part
    r"@"
    r"(?:[a-zA-Z0-9\-]+\.)*"        # zero or more sub-domain labels
    r"[a-zA-Z0-9\-]+"              # domain label (before TLD)
    r"\.[a-zA-Z]{2,24}$"           # TLD (supports .co.uk etc.)
)
_PHONE_RE = re.compile(r"^\+?[\d\s\-(). ]{7,25}$")

# Profile-URL patterns per platform (must match root profile pages only)
PLATFORM_PATTERNS: dict[str, re.Pattern[str]] = {
    "LinkedIn": re.compile(
        r"^https?://([a-z]{2,3}\.)?linkedin\.com/in/[A-Za-z0-9%_\-]{3,}/?$",
        re.IGNORECASE,
    ),
    "X": re.compile(
        r"^https?://(www\.)?(x|twitter)\.com/[A-Za-z0-9_]{1,30}/?$",
        re.IGNORECASE,
    ),
    "YouTube": re.compile(
        r"^https?://(www\.)?youtube\.com/"
        r"(@[A-Za-z0-9_.\-]{3,}|channel/[A-Za-z0-9_\-]{10,}|c/[A-Za-z0-9_.\-]{3,}|user/[A-Za-z0-9_.\-]{3,})"
        r"/?$",
        re.IGNORECASE,
    ),
    "Instagram": re.compile(
        r"^https?://(www\.)?instagram\.com/[A-Za-z0-9_.]{1,30}/?$",
        re.IGNORECASE,
    ),
    "Facebook": re.compile(
        r"^https?://(www\.)?facebook\.com/[A-Za-z0-9.\-_]{3,}/?$",
        re.IGNORECASE,
    ),
    "Threads": re.compile(
        r"^https?://(www\.)?threads\.net/@[A-Za-z0-9_.]{1,30}/?$",
        re.IGNORECASE,
    ),
    "WhatsApp": re.compile(
        r"^https?://(wa\.me/\+?[0-9]{7,15}|api\.whatsapp\.com/send\?phone=[0-9]{7,15})",
        re.IGNORECASE,
    ),
}

# Paths that indicate a non-profile page (posts, feeds, search results, etc.)
_DISALLOWED_PATH_SEGMENTS: dict[str, list[str]] = {
    "linkedin.com": ["/posts/", "/pulse/", "/jobs/", "/company/", "/school/", "/feed/", "/learning/"],
    "x.com": ["/status/", "/hashtag/", "/search", "/i/", "/home", "/explore"],
    "twitter.com": ["/status/", "/hashtag/", "/search", "/i/", "/home", "/explore"],
    "instagram.com": ["/p/", "/reel/", "/stories/", "/explore/", "/direct/"],
    "facebook.com": ["/story.php", "/permalink.php", "/groups/", "/events/", "/watch/"],
    "youtube.com": ["/watch", "/playlist", "/feed/", "/shorts/"],
    "threads.net": ["/post/"],
}

# Domains that must never appear in prospect records
_FORBIDDEN_DOMAINS = frozenset(
    [
        "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
        "wikipedia.org", "reddit.com", "quora.com", "medium.com",
        "yelp.com", "tripadvisor.com", "glassdoor.com",
    ]
)

# Common fake / placeholder email domains to reject
_BOGUS_EMAIL_DOMAINS = frozenset(
    [
        "example.com", "example.org", "example.net",
        "email.com", "test.com", "test.org",
        "noreply.com", "mailinator.com", "guerrillamail.com",
        "sentry.io", "wixpress.com", "placeholder.com",
    ]
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_valid_email(email: str) -> bool:
    if not email or len(email) > 254:
        return False
    email = email.strip().lower()
    domain = email.split("@")[-1] if "@" in email else ""
    if domain in _BOGUS_EMAIL_DOMAINS:
        return False
    return bool(_EMAIL_RE.match(email))


def is_valid_phone(phone: str) -> bool:
    if not phone:
        return False
    digits = re.sub(r"\D", "", phone)
    # ITU-T E.164: 7–15 digits
    return 7 <= len(digits) <= 15 and bool(_PHONE_RE.match(phone.strip()))


def clean_url(url: str) -> str:
    """
    Strip tracking query parameters (utm_*, fbclid, igshid, ref, source)
    and trailing slashes from a URL.
    """
    if not url:
        return url
    try:
        parsed = urlparse(url)
        _STRIP_PARAMS = frozenset(["fbclid", "igshid", "ref", "source", "mc_cid", "mc_eid"])
        clean_qs = [
            (k, v)
            for k, v in parse_qsl(parsed.query)
            if not k.lower().startswith("utm_") and k.lower() not in _STRIP_PARAMS
        ]
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc.lower(),
                parsed.path.rstrip("/"),
                parsed.params,
                urlencode(clean_qs),
                "",  # strip fragment
            )
        )
    except Exception:
        return url


def is_valid_profile_url(platform: str, url: str) -> bool:
    """
    Return True only if `url` appears to be a genuine public profile page
    for `platform` (not a post, search, feed, or forbidden aggregator domain).
    """
    if not url:
        return False

    url_lower = url.lower()

    # Reject known aggregator / search domains
    try:
        netloc = urlparse(url_lower).netloc
        base_domain = ".".join(netloc.split(".")[-2:])  # e.g. "google.com"
        if base_domain in _FORBIDDEN_DOMAINS:
            return False
    except Exception:
        pass

    # Reject disallowed path segments (platform-specific)
    for domain, bad_paths in _DISALLOWED_PATH_SEGMENTS.items():
        if domain in url_lower:
            for bp in bad_paths:
                if bp in url_lower:
                    return False

    # Match against the platform-specific profile pattern
    pattern = PLATFORM_PATTERNS.get(platform)
    if pattern:
        return bool(pattern.match(url))

    # For Google / generic sources: accept any http/https URL not already rejected
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme in {"http", "https"} and parsed.netloc and len(parsed.netloc) > 3)
    except Exception:
        return False
