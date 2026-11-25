"""
Normalizers – clean raw SERP titles, normalize locations, and extract handles.
"""
from __future__ import annotations

import re
from typing import Literal, Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# SERP title clean-up patterns
# Applied in order; each strips a trailing platform-branding suffix.
# ---------------------------------------------------------------------------
_TITLE_CLEANUP: tuple[str, ...] = (
    r"\s*[\|•]\s*LinkedIn.*$",
    r"\s*-\s*LinkedIn.*$",
    r"\s*[\|•]\s*YouTube.*$",
    r"\s*-\s*YouTube.*$",
    r"\s*[\|•]\s*Instagram.*$",
    r"\s*on\s+Instagram.*$",
    r"\s*[\|•]\s*Facebook.*$",
    r"\s*-\s*Facebook.*$",
    r"\s*/\s*X\s*$",
    r"\s*[\|•]\s*X\s*$",
    r"\s*on\s+X:.*$",
    r"\s*on\s+Twitter:.*$",
    r"\s*[\|•]\s*Twitter.*$",
    r"\s*on\s+Threads.*$",
    r"\s*[\|•]\s*Threads.*$",
    r"\s*-\s*WhatsApp.*$",
    r"\s*[\|•]\s*WhatsApp.*$",
)

# Exact generic / boilerplate page titles that are never human/business names
_BOILERPLATE_EXACT = frozenset(
    {
        "home", "about", "posts", "top", "profile", "search", "news", "latest",
        "results", "login", "log in", "sign in", "sign up", "signup", "explore",
        "linkedin", "instagram", "facebook", "twitter", "x", "youtube",
        "threads", "whatsapp", "404", "not found", "page not found", "404 not found",
        "just a moment", "access denied", "forbidden", "error", "welcome",
    }
)

_BOILERPLATE_SUBSTRINGS = (
    "log in or sign up",
    "sign in to",
    "join linkedin",
    "verify you are human",
    "page not found",
    "see photos and videos",
    "people on facebook",
    "travel guide",
    "news and updates",
)

# Fragments used only when deciding whether the prefix before '-' is boilerplate
_BAD_SEGMENT_TOKENS = frozenset(
    {"home", "about", "posts", "top", "profile", "search", "news", "latest", "results"}
)


def clean_name(raw_title: str, platform: str = "") -> str:
    """
    Extract a clean person or business name from a raw SERP page title.

    Returns an empty string when no clean name can be determined.
    """
    if not raw_title:
        return ""

    name = raw_title.strip()

    # 1. Strip trailing platform branding
    for pattern in _TITLE_CLEANUP:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()

    # 2. Handle "Name (@handle)" – keep only the name part
    handle_match = re.match(r"^([^(@]{2,}?)\s*\(@[A-Za-z0-9_.]+\)", name)
    if handle_match:
        name = handle_match.group(1).strip()

    # 3. Handle "Name - Role | Company" – keep only the first segment
    for sep in (" - ", " – ", " — ", " | ", " : "):
        if sep in name:
            first_part = name.split(sep)[0].strip()
            words = first_part.split()
            if 1 <= len(words) <= 6 and not any(
                t in first_part.lower() for t in _BAD_SEGMENT_TOKENS
            ):
                name = first_part
                break

    # 4. Strip leading/trailing punctuation that may remain
    name = re.sub(r"^[\"\'\(\[\{]+|[\"\'\)\]\}]+$", "", name).strip()

    # 5. Reject boilerplate titles, truncated article titles (...), URL slugs, or invalid lengths
    lowered = name.lower()
    if len(name) < 2 or len(name) > 80:
        return ""
    if name.endswith("...") or name.endswith("…"):
        return ""
    if re.match(r"^20\d{2}\s+", name):  # year-prefixed editorial/article titles
        return ""
    if lowered in _BOILERPLATE_EXACT:
        return ""
    if any(sub in lowered for sub in _BOILERPLATE_SUBSTRINGS):
        return ""
    if re.match(r"^https?://", name, re.IGNORECASE):
        return ""

    return name


def extract_handle(url: str, platform: str) -> Optional[str]:
    """
    Extract the username / channel handle from a known-platform profile URL.
    Returns None when the handle cannot be determined.
    """
    if not url:
        return None
    try:
        path = urlparse(url).path.strip("/")
        if not path:
            return None
        parts = path.split("/")
        p_lower = platform.lower()

        if "linkedin" in p_lower:
            if len(parts) >= 2 and parts[0].lower() == "in":
                return parts[1].lower()
            return parts[-1].lower()

        if p_lower in {"x", "twitter"}:
            return parts[0].lower().lstrip("@")

        if p_lower == "instagram":
            return parts[0].lower().lstrip("@")

        if p_lower == "youtube":
            first = parts[0].lower()
            if first.startswith("@"):
                return first.lstrip("@")
            if len(parts) >= 2 and first in {"channel", "c", "user"}:
                return parts[1].lower()
            return first

        if p_lower == "threads":
            return parts[0].lower().lstrip("@")

        if p_lower == "facebook":
            first = parts[0].lower()
            if first == "people" and len(parts) >= 2:
                return parts[1].lower()
            if first not in {"pages", "groups", "profile.php"}:
                return first

    except Exception:
        pass
    return None


def normalize_location(location: Optional[str]) -> Optional[str]:
    """Standardise a location string; return None for empty / placeholder values."""
    if not location:
        return None
    cleaned = re.sub(r"\s+", " ", location).strip(" ,.-")
    if not cleaned or cleaned.lower() in {"unknown", "n/a", "none", "null"}:
        return None
    return cleaned[:100]


LOCATION_ALIASES: dict[str, tuple[str, ...]] = {
    "uk": ("uk", "united kingdom", "great britain", "britain", "england", "scotland", "wales"),
    "usa": ("usa", "united states", "u.s.", "u.s.a.", "america"),
    "uae": ("uae", "united arab emirates"),
    "washington dc": ("washington dc", "washington, dc", "washington d.c.", "washington, d.c."),
    "district of columbia": ("district of columbia", "d.c.", "washington dc"),
    "bengaluru": ("bengaluru", "bangalore"),
    "new york": ("new york", "nyc", "new york city"),
}


def location_matches_text(loc_term: str, text: str) -> bool:
    """Word-boundary aware location matcher that respects canonical aliases."""
    term_lower = loc_term.lower().strip()
    if not term_lower or not text:
        return False
    text_lower = text.lower()
    aliases = LOCATION_ALIASES.get(term_lower, (term_lower,))
    for alias in aliases:
        if len(alias) <= 4:
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text_lower):
                return True
        else:
            if re.search(rf"\b{re.escape(alias)}\b", text_lower):
                return True
    return False


# ---------------------------------------------------------------------------
# Entity classification (individual vs business vs parody/gimmick account)
# ---------------------------------------------------------------------------

_PARODY_OR_GIMMICK_RE = re.compile(
    r"(?:"
    r"\b(?:parody|satire|memes?|fan\s*page|fan\s*club|fan\s*account|stan\s*account|"
    r"unofficial\s+(?:fan|page|updates)|updates?\s+account|daily\s+updates)\b"
    r"|"
    r"\b(?:king|queen|god|lord|prince)\s+of\s+[a-z]+\b"
    r")",
    re.IGNORECASE,
)

_ORG_NAME_INDICATORS_RE = re.compile(
    r"\b(?:network|association|foundation|institute|chamber|centre|center|"
    r"enterprise|collective|agency|agencies|studios?|ventures|capital|solutions|"
    r"services|consultancy|consulting|group|academy|university|spotlight|"
    r"directory|outlets?|club|society|community|organization|organisation|"
    r"ltd|llc|inc|corp|gmbh|llp|pvt|plc)\b",
    re.IGNORECASE,
)

_ORG_URL_PATH_RE = re.compile(
    r"(?:eventbrite\.[^/]+/o/|meetup\.com/|/company/|/organization/|/school/|/outlets)",
    re.IGNORECASE,
)

_PERSON_ROLE_RE = re.compile(
    r"\b(?:founder|co-founder|cofounder|ceo|cto|cfo|owner|director|manager|"
    r"engineer|developer|designer|consultant|advisor|investor|coach|analyst|"
    r"journalist|writer|entrepreneur)\b",
    re.IGNORECASE,
)


def classify_candidate_entity(
    name: str,
    platform: str,
    profile_url: str,
    title: str = "",
    snippet: str = "",
) -> Literal["individual", "business", "parody_or_gimmick"]:
    """
    Classify a candidate prospect using combined name, URL, title, and snippet
    evidence rather than naive single-keyword blocking.
    """
    n_clean = (name or "").strip()
    url_lower = (profile_url or "").lower()
    combined_meta = f"{n_clean} {title} {snippet}".strip()

    # 1. Parody / meme / fan-page / gimmick accounts
    if _PARODY_OR_GIMMICK_RE.search(n_clean) or _PARODY_OR_GIMMICK_RE.search(title):
        return "parody_or_gimmick"
    if re.search(r"\b(?:parody|satire|fan\s*account|stan\s*account|not\s+affiliated)\b", snippet, re.IGNORECASE):
        return "parody_or_gimmick"
    # Multi-word completely lowercase title/gimmick handle on social without human-name or role cues
    n_words = n_clean.split()
    if (
        len(n_words) >= 2
        and n_clean.islower()
        and not _PERSON_ROLE_RE.search(f"{title} {snippet}")
        and platform != "LinkedIn"
    ):
        return "parody_or_gimmick"

    # 2. Organization / business / event community entities
    if _ORG_URL_PATH_RE.search(url_lower):
        return "business"

    is_personal_linkedin = platform == "LinkedIn" and "/in/" in url_lower
    has_org_name = bool(_ORG_NAME_INDICATORS_RE.search(n_clean))

    # An org-like name (e.g. "London Founder's Network", "Activate Studio",
    # "Centre for Fashion Enterprise") without a personal /in/ LinkedIn profile is a business/org
    if has_org_name and not is_personal_linkedin:
        return "business"

    # Generic Google domain without personal profile markers or person-name structure
    if platform in {"Google", "Email"} and not is_personal_linkedin:
        has_about_me = "about.me/" in url_lower
        looks_like_person_name = (
            2 <= len(n_words) <= 3
            and all(w[0].isupper() for w in n_words if w)
            and not any(ch in n_clean for ch in ("&", ":", "|"))
        )
        if not has_about_me and not looks_like_person_name:
            return "business"

    return "individual"


