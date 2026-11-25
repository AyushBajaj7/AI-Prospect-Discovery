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

    # 2. Handle "Name (@handle)" and trailing "Name @handle" – keep only the name part
    handle_match = re.match(r"^([^(@]{2,}?)\s*\(@[A-Za-z0-9_.]+\)", name)
    if handle_match:
        name = handle_match.group(1).strip()
    name = re.sub(r"\s+@[A-Za-z0-9_.-]+$", "", name).strip()

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
    r"digital\s+marketing|marketing|branding|advertising|publications?|magazine|"
    r"newsletter|podcast|hub|labs?|official|platform|software|technologies|"
    r"systems|analytics|insights|"
    r"restaurants?|hotels?|cafes?|cafés?|bistros?|bakeries|bakery|bars?|"
    r"lounges?|grills?|kitchens?|diners?|catering|resorts?|clinics?|salons?|"
    r"boutiques?|shops?|stores?|company|companies|firms?|partners|hospitality|"
    r"pizzerias?|steakhouses?|roastery|roasteries|"
    r"ltd|llc|inc|corp|gmbh|llp|pvt|plc)\b",
    re.IGNORECASE,
)

_ORG_URL_PATH_RE = re.compile(
    r"(?:eventbrite\.[^/]+/o/|meetup\.com/|/company/|/organization/|/school/|/outlets|/pages/|/pg/)",
    re.IGNORECASE,
)

_ORG_CONTEXT_RE = re.compile(
    r"\b(?:our\s+(?:agency|company|team|clients|services|mission|restaurant|menu|firm|studio)|"
    r"we\s+(?:are\s+a|help\s+(?:businesses|brands|companies|startups)|provide|offer|specialize|serve)|"
    r"full[\-\s]service\s+(?:digital|marketing|agency|creative|studio)|"
    r"official\s+(?:page|channel|website|account|community)|"
    r"(?:community|network|platform|hub|resource|guide|directory)\s+(?:for|of)\s+|"
    r"dedicated\s+to\s+(?:businesses|founders|startups|entrepreneurs)|"
    r"book\s+a\s+table|order\s+online|opening\s+hours)\b",
    re.IGNORECASE,
)

_PREPOSITION_ORG_NAME_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9'&\-]*(?:\s+[A-Za-z][A-Za-z0-9'&\-]*)?\s+of\s+[A-Za-z][A-Za-z\s\-]+$",
    re.IGNORECASE,
)

_PERSON_ROLE_RE = re.compile(
    r"\b(?:founder|co-founder|cofounder|ceo|cto|cfo|owner|director|manager|"
    r"engineer|developer|designer|consultant|advisor|investor|coach|analyst|"
    r"journalist|writer|author|entrepreneur|lover|enthusiast|fan|he/him|she/her|"
    r"they/them|i\s+am|i'm)\b",
    re.IGNORECASE,
)


def _handle_corroborates_name(name: str, platform: str, profile_url: str) -> bool:
    """Return True when the URL handle strongly echoes tokens of `name`."""
    handle = extract_handle(profile_url, platform) or ""
    handle_clean = re.sub(r"[^a-z0-9]", "", handle.lower())
    if len(handle_clean) < 4:
        return False
    name_tokens = [re.sub(r"[^a-z0-9]", "", tok.lower()) for tok in name.split()]
    name_tokens = [tok for tok in name_tokens if len(tok) >= 2]
    if len(name_tokens) < 2:
        return False
    # Check full surname OR >=2 name tokens inside handle (e.g. john+doe -> johndoe, a.b.+dharmadhikari)
    matched = sum(1 for tok in name_tokens if len(tok) >= 3 and tok in handle_clean)
    return matched >= 2 or (len(name_tokens[-1]) >= 6 and name_tokens[-1] in handle_clean)


def classify_candidate_entity(
    name: str,
    platform: str,
    profile_url: str,
    title: str = "",
    snippet: str = "",
) -> Literal["individual", "business", "unknown", "parody_or_gimmick"]:
    """
    Classify a candidate prospect using combined name, URL, title, and snippet
    evidence. Returns `"unknown"` when identity signals are genuinely ambiguous
    rather than assuming title-cased words belong to a person.
    """
    n_clean = (name or "").strip()
    url_lower = (profile_url or "").lower()
    combined_context = f"{title} {snippet}".strip()

    # 1. Explicit parody / satire / meme / fan-page / gimmick signals
    if _PARODY_OR_GIMMICK_RE.search(n_clean) or _PARODY_OR_GIMMICK_RE.search(title):
        return "parody_or_gimmick"
    if re.search(
        r"\b(?:parody\s+account|satire\s+account|fan\s*account|stan\s*account|not\s+affiliated)\b",
        snippet,
        re.IGNORECASE,
    ):
        return "parody_or_gimmick"

    # 2. Organization / business / event community entities
    if _ORG_URL_PATH_RE.search(url_lower):
        return "business"

    is_personal_linkedin = platform == "LinkedIn" and "/in/" in url_lower
    has_org_name = bool(_ORG_NAME_INDICATORS_RE.search(n_clean))

    if has_org_name and not is_personal_linkedin:
        return "business"

    if not is_personal_linkedin:
        if _PREPOSITION_ORG_NAME_RE.search(n_clean):
            return "business"
        n_words = n_clean.split()
        # Trailing uppercase business acronym like "Main Street ROI", "Apex PPC", "Acme B2B"
        if len(n_words) >= 2 and re.match(r"^[A-Z]{2,5}$", n_words[-1]) and n_words[-1] not in {"UK", "USA", "UAE", "CEO", "CTO", "MD"}:
            return "business"
        # First-person / institutional org copy in title or snippet without personal LinkedIn URL
        if _ORG_CONTEXT_RE.search(combined_context) and not re.search(
            r"\b(?:co-?founder\s+(?:of|at)|founder\s+(?:of|at)|owner\s+(?:of|at))\b",
            combined_context,
            re.IGNORECASE,
        ):
            return "business"

    if is_personal_linkedin:
        if _ORG_NAME_INDICATORS_RE.search(n_clean) and not _PERSON_ROLE_RE.search(combined_context):
            return "business"
        return "individual"

    n_words = n_clean.split()
    looks_like_person_name = (
        2 <= len(n_words) <= 3
        and all(w[0].isalpha() for w in n_words if w)
        and not any(ch in n_clean for ch in ("&", ":", "|"))
        and not any(location_matches_text(city_tok, n_clean) for city_tok in ("london", "dubai", "new york", "washington", "ahmedabad"))
    )

    has_person_role = bool(_PERSON_ROLE_RE.search(combined_context))
    has_handle_match = _handle_corroborates_name(n_clean, platform, profile_url)

    if platform in {"Google", "Email"}:
        if "about.me/" in url_lower:
            return "individual"
        if looks_like_person_name and (has_person_role or has_handle_match):
            return "individual"
        if len(n_words) >= 4 or any(ch in n_clean for ch in ("&", "|")):
            return "business"
        return "unknown"

    # Social accounts: require multi-word person name PLUS personal role or handle corroboration
    if looks_like_person_name and (has_person_role or has_handle_match):
        return "individual"

    return "unknown"


