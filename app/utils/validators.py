import re
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
PHONE_REGEX = re.compile(r"^\+?[0-9\s\-().]{7,25}$")

PLATFORM_PATTERNS = {
    "LinkedIn": re.compile(r"^https?://([a-z]{2,3}\.)?linkedin\.com/in/[a-zA-Z0-9\-_%]+/?", re.IGNORECASE),
    "X": re.compile(r"^https?://(www\.)?(x|twitter)\.com/[a-zA-Z0-9_]{1,30}/?$", re.IGNORECASE),
    "YouTube": re.compile(r"^https?://(www\.)?youtube\.com/(@[a-zA-Z0-9_\-\.]+|channel/[a-zA-Z0-9_\-]+|c/[a-zA-Z0-9_\-]+|user/[a-zA-Z0-9_\-]+)/?$", re.IGNORECASE),
    "Instagram": re.compile(r"^https?://(www\.)?instagram\.com/[a-zA-Z0-9_\.\-]+/?$", re.IGNORECASE),
    "Facebook": re.compile(r"^https?://(www\.)?facebook\.com/[a-zA-Z0-9\.\-_]+/?$", re.IGNORECASE),
    "Threads": re.compile(r"^https?://(www\.)?threads\.net/@[a-zA-Z0-9_\.\-]+/?$", re.IGNORECASE),
    "WhatsApp": re.compile(r"^https?://(wa\.me/\+?[0-9]+|api\.whatsapp\.com/send\?phone=[0-9]+)", re.IGNORECASE),
}

DISALLOWED_PATHS = {
    "linkedin.com": ["/posts", "/pulse", "/jobs", "/company", "/school", "/feed", "/learning"],
    "x.com": ["/status", "/hashtag", "/search", "/i/", "/home", "/explore"],
    "twitter.com": ["/status", "/hashtag", "/search", "/i/", "/home", "/explore"],
    "instagram.com": ["/p/", "/reel/", "/stories/", "/explore/", "/direct/"],
    "facebook.com": ["/story.php", "/permalink.php", "/groups", "/events", "/share", "/watch"],
    "youtube.com": ["/watch", "/playlist", "/feed", "/shorts"],
    "threads.net": ["/post/"],
}

def is_valid_email(email: str) -> bool:
    if not email or len(email) > 254:
        return False
    # Avoid placeholder or bogus dummy emails
    if any(fake in email.lower() for fake in ["example.com", "email.com", "domain.com", "test.com", "sentry.io", "wixpress.com"]):
        return False
    return bool(EMAIL_REGEX.match(email.strip()))

def is_valid_phone(phone: str) -> bool:
    if not phone:
        return False
    digits = re.sub(r"\D", "", phone)
    # Most valid phone numbers have between 8 and 15 digits
    return 8 <= len(digits) <= 15 and bool(PHONE_REGEX.match(phone.strip()))

def clean_url(url: str) -> str:
    """Removes query params like utm_* or tracking identifiers."""
    try:
        parsed = urlparse(url)
        clean_query = [
            (k, v) for k, v in parse_qsl(parsed.query)
            if not k.lower().startswith("utm_") and k.lower() not in ["ref", "fbclid", "igshid", "source"]
        ]
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip("/"),
            parsed.params,
            urlencode(clean_query),
            ""
        ))
    except Exception:
        return url

def is_valid_profile_url(platform: str, url: str) -> bool:
    """Verifies that a URL represents a genuine user or business profile and not a post/feed/search page."""
    if not url:
        return False
    url_lower = url.lower()
    
    # Check disallowed paths
    for domain, bad_paths in DISALLOWED_PATHS.items():
        if domain in url_lower:
            for bp in bad_paths:
                if bp in url_lower:
                    return False

    pattern = PLATFORM_PATTERNS.get(platform)
    if pattern:
        return bool(pattern.match(url))
    
    # Generic URLs (e.g. Google source or websites)
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme in ["http", "https"] and parsed.netloc)
    except Exception:
        return False
