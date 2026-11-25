"""
Validators – URL and contact-detail validation.

All validation is local (no external HTTP calls) and fast.
Includes anti-SSRF host checks, bogus/placeholder email domain filtering,
and non-profile URL exclusion.
"""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Supports multi-part TLDs (e.g. .co.uk, .com.au) and standard TLDs.
# Prevents consecutive dots or leading/trailing dots in local part.
_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9_%+\-]+(?:\.[a-zA-Z0-9_%+\-]+)*"  # local part
    r"@"
    r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?\.)+"  # domain labels
    r"[a-zA-Z]{2,24}$"                                    # TLD
)

_PHONE_RE = re.compile(r"^\+?[\d\s\-(). ]{7,25}$")
# Reject common non-phone patterns: year ranges (2020-2025) or ISO dates (2024-01-15)
_DATE_OR_YEAR_RANGE_RE = re.compile(
    r"^(?:19|20)\d{2}[\s\-/.](?:(?:19|20)\d{2}|[01]\d[\s\-/.][0-3]\d)$"
)

# Profile-URL patterns per platform (must match root profile pages only)
PLATFORM_PATTERNS: dict[str, re.Pattern[str]] = {
    "LinkedIn": re.compile(
        r"^https?://([a-z]{2,3}\.)?linkedin\.com/in/[A-Za-z0-9%_\-.]{3,}/?$",
        re.IGNORECASE,
    ),
    "X": re.compile(
        r"^https?://(www\.|mobile\.)?(x|twitter)\.com/[A-Za-z0-9_]{1,30}/?$",
        re.IGNORECASE,
    ),
    "YouTube": re.compile(
        r"^https?://(www\.|m\.)?youtube\.com/"
        r"(@[A-Za-z0-9_.\-]{3,}|channel/[A-Za-z0-9_\-]{10,}|c/[A-Za-z0-9_.\-]{3,}|user/[A-Za-z0-9_.\-]{3,})"
        r"/?$",
        re.IGNORECASE,
    ),
    "Instagram": re.compile(
        r"^https?://(www\.|m\.)?instagram\.com/[A-Za-z0-9_.]{1,30}/?$",
        re.IGNORECASE,
    ),
    "Facebook": re.compile(
        r"^https?://(www\.|m\.|mbasic\.|business\.)?facebook\.com/"
        r"(?:[A-Za-z0-9.\-_]{3,}|people/[A-Za-z0-9.\-_/%]+/[0-9]+|profile\.php\?id=[0-9]+)/?$",
        re.IGNORECASE,
    ),
    "Threads": re.compile(
        r"^https?://(www\.)?threads\.net/@[A-Za-z0-9_.]{1,30}/?$",
        re.IGNORECASE,
    ),
    "WhatsApp": re.compile(
        r"^https?://(wa\.me/\+?[0-9]{7,15}|api\.whatsapp\.com/send\?phone=\+?[0-9]{7,15})",
        re.IGNORECASE,
    ),
}

# Exact non-profile paths / prefixes per domain (matched with or without trailing slash)
_DISALLOWED_PATH_SEGMENTS: dict[str, tuple[str, ...]] = {
    "linkedin.com": (
        "/posts", "/pulse", "/jobs", "/company", "/school",
        "/feed", "/learning", "/login", "/signup", "/help", "/legal",
    ),
    "x.com": (
        "/status", "/hashtag", "/search", "/i/", "/home", "/explore",
        "/login", "/signup", "/intent", "/settings", "/tos", "/privacy",
    ),
    "twitter.com": (
        "/status", "/hashtag", "/search", "/i/", "/home", "/explore",
        "/login", "/signup", "/intent", "/settings", "/tos", "/privacy",
    ),
    "instagram.com": (
        "/p/", "/reel", "/stories", "/explore", "/direct",
        "/accounts", "/developer", "/about", "/legal",
    ),
    "facebook.com": (
        "/story.php", "/permalink.php", "/groups", "/events", "/watch",
        "/login", "/help", "/policies", "/privacy", "/terms", "/marketplace",
        "/gaming", "/hashtag", "/directory", "/search", "/share", "/sharer",
        "/photo", "/videos", "/pages",
    ),
    "youtube.com": (
        "/watch", "/playlist", "/feed", "/shorts", "/results",
        "/account", "/premium", "/about", "/howyoutubeworks",
    ),
    "threads.net": ("/post", "/login", "/search"),
}

# Domains that must never appear in prospect records
_FORBIDDEN_DOMAINS = frozenset(
    {
        "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
        "wikipedia.org", "reddit.com", "quora.com", "medium.com",
        "yelp.com", "tripadvisor.com", "glassdoor.com",
        "indeed.com", "pinterest.com", "amazon.com",
    }
)

# Common fake / placeholder email domains to reject
_BOGUS_EMAIL_DOMAINS = frozenset(
    {
        "example.com", "example.org", "example.net",
        "email.com", "test.com", "test.org",
        "noreply.com", "mailinator.com", "guerrillamail.com",
        "sentry.io", "wixpress.com", "placeholder.com",
    }
)

# Query parameters stripped during URL normalisation
_STRIP_PARAMS = frozenset(
    {
        "fbclid", "igshid", "igsh", "ref", "ref_src", "ref_url",
        "source", "mc_cid", "mc_eid", "si", "trk", "trkinfo",
        "hl", "lang", "locale", "mibextid", "__tn__", "s", "t",
    }
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_unsafe_host(hostname: str) -> bool:
    """Reject localhost, loopback, link-local, or private IP hosts (anti-SSRF)."""
    if not hostname:
        return True
    h = hostname.lower().strip("[]")
    if h in {"localhost", "localhost.localdomain"} or h.endswith((".local", ".internal", ".localhost")):
        return True
    try:
        ip = ipaddress.ip_address(h)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
    except ValueError:
        return False


def _matches_domain(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith("." + domain)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_valid_email(email: str) -> bool:
    if not email or len(email) > 254:
        return False
    email = email.strip().lower()
    local, _, domain = email.partition("@")
    if not local or len(local) > 64 or not domain:
        return False
    if domain in _BOGUS_EMAIL_DOMAINS:
        return False
    return bool(_EMAIL_RE.match(email))


def is_valid_phone(phone: str) -> bool:
    if not phone:
        return False
    stripped = phone.strip()
    if _DATE_OR_YEAR_RANGE_RE.match(stripped):
        return False
    if not _PHONE_RE.match(stripped):
        return False

    digits = re.sub(r"\D", "", stripped)
    # ITU-T E.164: 7–15 digits
    if not (7 <= len(digits) <= 15):
        return False

    # Reject repeated single-digit placeholders like 00000000 or 111111111
    if len(set(digits)) == 1:
        return False

    # Unformatted bare digit strings without '+' are commonly post IDs, ZIPs, or stats
    # rather than genuine phone numbers unless formatted with separators or '+'.
    has_formatting = stripped.startswith("+") or any(ch in stripped for ch in ("-", " ", "(", "."))
    if not has_formatting:
        return False

    return True


def clean_url(url: str) -> str:
    """
    Strip tracking/locale query parameters, fragments, and trailing slashes from a URL.
    Preserves essential parameters (e.g., Facebook `id=` or WhatsApp `phone=`).
    """
    if not url:
        return url
    try:
        parsed = urlparse(url.strip())
        clean_qs = [
            (k, v)
            for k, v in parse_qsl(parsed.query)
            if not k.lower().startswith("utm_") and k.lower() not in _STRIP_PARAMS
        ]
        # Strip trailing slash on path unless path is "/"
        path = parsed.path.rstrip("/")
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                path,
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
    for `platform` (not a post, search, feed, private IP, or forbidden domain).
    """
    if not url:
        return False

    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False

    if parsed.scheme.lower() not in {"http", "https"}:
        return False

    hostname = (parsed.hostname or "").lower()
    if not hostname or "." not in hostname or _is_unsafe_host(hostname):
        return False

    # Reject known aggregator / search domains (including regional TLDs like google.co.uk)
    if any(_matches_domain(hostname, fd) for fd in _FORBIDDEN_DOMAINS):
        return False
    if hostname.startswith("google.") or ".google." in hostname:
        return False

    # Reject disallowed path segments (platform-specific)
    path_lower = (parsed.path or "").lower()
    for domain, bad_paths in _DISALLOWED_PATH_SEGMENTS.items():
        if _matches_domain(hostname, domain):
            for bp in bad_paths:
                # Match either "/segment" at end or "/segment/" / "/segment..." inside path
                if bp.endswith("/"):
                    if bp in path_lower:
                        return False
                elif path_lower == bp or path_lower.startswith(bp + "/") or (bp + "/") in path_lower:
                    return False

    # Match against the platform-specific profile pattern
    pattern = PLATFORM_PATTERNS.get(platform)
    if pattern:
        return bool(pattern.match(url.strip()))

    # For Google / generic sources: accept valid http/https external URLs not excluded above
    return len(hostname) > 3
