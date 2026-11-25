"""
Validators – URL validation, WhatsApp normalization, search-operator
sanitization, and contact-detail validation.

All validation is local (no external HTTP calls), ReDoS-safe, and fast.
Note: If live HTTP fetching of external profile URLs is ever added in the
future, add DNS resolution + redirect revalidation + connection-time IP
checks to protect against DNS rebinding and private-network SSRF.
"""
from __future__ import annotations

import ipaddress
import re
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Supports multi-part TLDs (e.g. .co.uk, .com.au) and standard TLDs.
_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9_%+\-]+(?:\.[a-zA-Z0-9_%+\-]+){0,8}"
    r"@"
    r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.){1,5}"
    r"[a-zA-Z]{2,24}$"
)

_PHONE_RE = re.compile(r"^\+?[\d\s\-(). ]{7,25}$")
# Reject common non-phone patterns: year ranges (2020-2025) or ISO dates (2024-01-15)
_DATE_OR_YEAR_RANGE_RE = re.compile(
    r"^(?:19|20)\d{2}[\s\-/.](?:(?:19|20)\d{2}|[01]\d[\s\-/.][0-3]\d)$"
)

# Profile-URL patterns per platform (must match root profile pages only).
# Per WhatsApp official specification, wa.me and send?phone= reject '+' and formatting chars.
PLATFORM_PATTERNS: dict[str, re.Pattern[str]] = {
    "LinkedIn": re.compile(
        r"^https?://([a-z]{2,3}\.)?linkedin\.com/in/[A-Za-z0-9%_\-.]{3,100}/?$",
        re.IGNORECASE,
    ),
    "X": re.compile(
        r"^https?://(www\.|mobile\.)?(x|twitter)\.com/[A-Za-z0-9_]{1,30}/?$",
        re.IGNORECASE,
    ),
    "YouTube": re.compile(
        r"^https?://(www\.|m\.)?youtube\.com/"
        r"(@[A-Za-z0-9_.\-]{3,80}|channel/[A-Za-z0-9_\-]{10,60}|c/[A-Za-z0-9_.\-]{3,80}|user/[A-Za-z0-9_.\-]{3,80})"
        r"/?$",
        re.IGNORECASE,
    ),
    "Instagram": re.compile(
        r"^https?://(www\.|m\.)?instagram\.com/[A-Za-z0-9_.]{1,30}/?$",
        re.IGNORECASE,
    ),
    "Facebook": re.compile(
        r"^https?://(www\.|m\.|mbasic\.|business\.)?facebook\.com/"
        r"(?:[A-Za-z0-9.\-_]{3,80}|people/[A-Za-z0-9.\-_/%]{2,120}/[0-9]{5,25}|profile\.php\?id=[0-9]{5,25})/?$",
        re.IGNORECASE,
    ),
    "Threads": re.compile(
        r"^https?://(www\.)?threads\.net/@[A-Za-z0-9_.]{1,30}/?$",
        re.IGNORECASE,
    ),
    "WhatsApp": re.compile(
        r"^https?://(?:wa\.me/[0-9]{7,15}|api\.whatsapp\.com/send\?phone=[0-9]{7,15})/?$",
        re.IGNORECASE,
    ),
}

# Exact non-profile paths / prefixes per domain
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

_FORBIDDEN_DOMAINS = frozenset(
    {
        "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
        "wikipedia.org", "reddit.com", "quora.com", "medium.com",
        "yelp.com", "tripadvisor.com", "glassdoor.com",
        "indeed.com", "pinterest.com", "amazon.com",
    }
)

_BOGUS_EMAIL_DOMAINS = frozenset(
    {
        "example.com", "example.org", "example.net",
        "email.com", "test.com", "test.org",
        "noreply.com", "mailinator.com", "guerrillamail.com",
        "sentry.io", "wixpress.com", "placeholder.com",
    }
)

_STRIP_PARAMS = frozenset(
    {
        "fbclid", "igshid", "igsh", "ref", "ref_src", "ref_url",
        "source", "mc_cid", "mc_eid", "si", "trk", "trkinfo",
        "hl", "lang", "locale", "mibextid", "__tn__", "s", "t",
    }
)

# Search-engine operators stripped from untrusted user/LLM intent before query building
_SEARCH_OPERATOR_RE = re.compile(
    r"(?:^|\s)[+\-]?(?:site|inurl|allinurl|intitle|allintitle|intext|allintext|"
    r"filetype|ext|cache|related|info|link|define):[^\s]*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_unsafe_host(hostname: str) -> bool:
    """Reject localhost, loopback, link-local, or private IPv4/IPv6 hosts."""
    if not hostname:
        return True
    h = hostname.lower().strip("[]")
    if h in {"localhost", "localhost.localdomain"} or h.endswith(
        (".local", ".internal", ".localhost")
    ):
        return True
    try:
        ip = ipaddress.ip_address(h)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )
    except ValueError:
        return False


def _matches_domain(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith("." + domain)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def sanitize_search_term(value: Optional[str], max_len: int = 80) -> str:
    """
    Remove search operators, quotes, parentheses, and control characters from
    intent fields prior to constructing search engine query strings.
    Prevents prompt-injected or user-supplied search operator manipulation.
    """
    if not value:
        return ""
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    cleaned = _SEARCH_OPERATOR_RE.sub(" ", cleaned)
    # Strip search syntax metacharacters: quotes, parens, brackets, wildcards
    cleaned = re.sub(r'["\'`()\[\]{}*|;\\]', " ", cleaned)
    # Remove boolean OR / AND operators standing alone and unary negation flags
    cleaned = re.sub(r"\b(?:OR|AND|NOT)\b", " ", cleaned)
    cleaned = re.sub(r"(?:^|\s)-+(?=\S)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_len]


def normalize_whatsapp_url(url: str) -> Optional[str]:
    """
    Normalize a wa.me or api.whatsapp.com/send link to official WhatsApp format
    (`https://wa.me/<digits>` with NO '+' sign or formatting characters).
    Returns None if the digits are missing, <7 or >15 digits, or bogus.
    """
    if not url:
        return None
    stripped = url.strip()
    match = re.match(
        r"^https?://(?:wa\.me/(\+?[0-9\-\s()]{7,25})|api\.whatsapp\.com/send\?[^#]*\bphone=(\+?[0-9\-\s()]{7,25}))",
        stripped,
        re.IGNORECASE,
    )
    if not match:
        return None
    raw_phone = match.group(1) or match.group(2)
    digits = re.sub(r"\D", "", raw_phone)
    if not (7 <= len(digits) <= 15) or len(set(digits)) == 1:
        return None
    return f"https://wa.me/{digits}"


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
    if not phone or len(phone) > 30:
        return False
    stripped = phone.strip()
    if _DATE_OR_YEAR_RANGE_RE.match(stripped):
        return False
    if not _PHONE_RE.match(stripped):
        return False

    digits = re.sub(r"\D", "", stripped)
    if not (7 <= len(digits) <= 15):
        return False

    if len(set(digits)) == 1:
        return False

    has_formatting = stripped.startswith("+") or any(ch in stripped for ch in ("-", " ", "(", "."))
    if not has_formatting:
        return False

    return True


def clean_url(url: str) -> str:
    """
    Strip tracking/locale query parameters, fragments, and trailing slashes from a URL,
    canonicalize regional LinkedIn subdomains (uk.linkedin.com -> www.linkedin.com),
    and normalize WhatsApp URLs to digit-only https://wa.me/<digits>.
    """
    if not url:
        return url
    stripped = url.strip()
    wa_norm = normalize_whatsapp_url(stripped)
    if wa_norm:
        return wa_norm

    try:
        parsed = urlparse(stripped)
        netloc = parsed.netloc.lower()
        hostname = (parsed.hostname or "").lower()
        # Canonicalize regional LinkedIn domains (e.g., uk.linkedin.com, ge.linkedin.com, linkedin.com)
        if re.match(r"^(?:[a-z]{2,3}\.)?linkedin\.com$", hostname):
            netloc = "www.linkedin.com"

        clean_qs = [
            (k, v)
            for k, v in parse_qsl(parsed.query)
            if not k.lower().startswith("utm_") and k.lower() not in _STRIP_PARAMS
        ]
        path = parsed.path.rstrip("/")
        return urlunparse(
            (
                parsed.scheme.lower(),
                netloc,
                path,
                parsed.params,
                urlencode(clean_qs),
                "",
            )
        )
    except Exception:
        return url


GENERIC_EMAIL_PREFIXES = frozenset(
    {
        "info", "hello", "contact", "support", "help", "admin", "team",
        "sales", "marketing", "office", "hi", "enquiries", "inquiries",
        "press", "media", "jobs", "careers", "founder", "founders",
        "cofounder", "cofounders", "ceo", "noreply", "hr", "recruiting",
        "talent", "billing", "pr", "partnerships", "connect", "mail",
        "general", "feedback", "bookings", "reservations", "orders",
    }
)

_NON_AUTHORITATIVE_PHONE_HOSTS = frozenset(
    {
        "linkedin.com", "x.com", "twitter.com", "instagram.com",
        "facebook.com", "fb.com", "youtube.com", "youtu.be",
        "threads.net", "threads.com", "wa.me", "whatsapp.com",
        "about.me", "linktr.ee", "bio.link", "beacons.ai", "carrd.co",
        "medium.com", "substack.com", "github.com",
    }
)


def is_generic_role_email(email: str) -> bool:
    """Return True if an email local-part is an organizational or role-based inbox."""
    if not email or "@" not in email:
        return False
    local = email.strip().lower().partition("@")[0]
    normalized_local = re.sub(r"[.\-_+]", "", local)
    return local in GENERIC_EMAIL_PREFIXES or normalized_local in GENERIC_EMAIL_PREFIXES


def derive_phone_context(profile_url: str, platform: str) -> Optional[str]:
    """
    Return a domain `phone-context` for local un-prefixed phone numbers ONLY
    when the source URL is a first-party personal or business website, never
    for third-party social networks, link-in-bio services, or messaging hosts.
    """
    if platform not in {"Google", "Email"} or not profile_url:
        return None
    try:
        hostname = (urlparse(profile_url.strip()).hostname or "").lower()
    except Exception:
        return None
    if not hostname or "." not in hostname or _is_unsafe_host(hostname):
        return None
    if any(_matches_domain(hostname, host) for host in _NON_AUTHORITATIVE_PHONE_HOSTS):
        return None
    return hostname


def format_tel_uri(phone: str, phone_context: Optional[str] = None) -> str:
    """
    Format a validated phone string as an RFC 3966-compliant `tel:` URI:
      * Global numbers starting with `+`: `tel:+<digits>` (no spaces/separators).
      * Local numbers without `+`: require a `phone-context` descriptor per
        RFC 3966 Section 5.1.5 (`tel:<digits>;phone-context=<domain>`). Never
        fabricates a country code; returns "" if no `phone_context` is known.
    """
    stripped = (phone or "").strip()
    has_plus = stripped.startswith("+")
    digits = re.sub(r"\D", "", stripped)
    if not digits:
        return ""
    if has_plus:
        return f"tel:+{digits}"
    ctx = (phone_context or "").strip().lower()
    if ctx:
        return f"tel:{digits};phone-context={ctx}"
    return ""


def is_valid_profile_url(platform: str, url: str) -> bool:
    """
    Return True only if `url` matches a genuine public profile format
    for `platform` (not a post, search, feed, private IPv4/IPv6, or forbidden domain).
    """
    if not url or len(url) > 512:
        return False

    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False

    if parsed.scheme.lower() not in {"http", "https"}:
        return False

    hostname = (parsed.hostname or "").lower()
    if not hostname or _is_unsafe_host(hostname) or "." not in hostname:
        return False

    if any(_matches_domain(hostname, fd) for fd in _FORBIDDEN_DOMAINS):
        return False
    if hostname.startswith("google.") or ".google." in hostname:
        return False

    path_lower = (parsed.path or "").lower()
    for domain, bad_paths in _DISALLOWED_PATH_SEGMENTS.items():
        if _matches_domain(hostname, domain):
            for bp in bad_paths:
                if bp.endswith("/"):
                    if bp in path_lower:
                        return False
                elif path_lower == bp or path_lower.startswith(bp + "/") or (bp + "/") in path_lower:
                    return False

    pattern = PLATFORM_PATTERNS.get(platform)
    if pattern:
        return bool(pattern.match(url.strip()))

    return len(hostname) > 3
