"""
Normalizers – clean raw SERP titles and extract platform handles.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# SERP title clean-up patterns
# Applied in order; each strips a trailing platform-branding suffix.
# ---------------------------------------------------------------------------
_TITLE_CLEANUP: list[str] = [
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
    r"\s*[\|•]\s*YouTube.*$",
]

# Fragments that signal the name is not a human / business name
_BAD_NAME_TOKENS = frozenset(
    ["home", "about", "posts", "top", "profile", "search", "news", "latest", "results"]
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
            if 1 <= len(words) <= 6 and not any(t in first_part.lower() for t in _BAD_NAME_TOKENS):
                name = first_part
                break

    # 4. Strip leading/trailing punctuation that may remain
    name = re.sub(r"^[\"\'\(\[\{]+|[\"\'\)\]\}]+$", "", name).strip()

    # 5. Reject anything that looks like a URL slug or is too short / long
    if len(name) < 2 or len(name) > 80:
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
            if len(parts) >= 2 and parts[0] == "in":
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
            if first not in {"pages", "people", "groups", "profile.php"}:
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
    # Cap at a reasonable length
    return cleaned[:100]
