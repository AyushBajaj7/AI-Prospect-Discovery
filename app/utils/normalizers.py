import re
from typing import Optional
from urllib.parse import urlparse

# Common junk suffixes and noise patterns in SERP titles
TITLE_CLEANUP_PATTERNS = [
    r"\s*\|\s*LinkedIn.*$",
    r"\s*-\s*LinkedIn.*$",
    r"\s*-\s*YouTube.*$",
    r"\s*\|\s*YouTube.*$",
    r"\s*•\s*Instagram.*$",
    r"\s*\|\s*Instagram.*$",
    r"\s*on\s+Instagram.*$",
    r"\s*\|\s*Facebook.*$",
    r"\s*-\s*Facebook.*$",
    r"\s*/\s*X.*$",
    r"\s*\|\s*X.*$",
    r"\s*on\s+X:.*$",
    r"\s*on\s+Twitter:.*$",
    r"\s*\|\s*Twitter.*$",
    r"\s*on\s+Threads.*$",
    r"\s*\|\s*Threads.*$",
    r"\s*-\s*WhatsApp.*$",
    r"\s*\|\s*WhatsApp.*$",
]

def clean_name(raw_title: str, platform: str = "") -> str:
    """Extracts a clean human or entity name from a search result title."""
    if not raw_title:
        return ""
    
    name = raw_title.strip()
    
    # Strip common platform branding suffixes
    for pattern in TITLE_CLEANUP_PATTERNS:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()
    
    # Handle patterns like "Name (@handle)"
    handle_match = re.search(r"^([^(@]+)\s*\(@[a-zA-Z0-9_\.]+\)", name)
    if handle_match:
        name = handle_match.group(1).strip()

    # Handle "Name - Title or Role at Company"
    # Take the first segment if separated by '-' or '|' or '–' or ':'
    for sep in [" - ", " – ", " — ", " | ", " : "]:
        if sep in name:
            parts = name.split(sep)
            first_part = parts[0].strip()
            # If the first part looks like a realistic name (1 to 5 words)
            words = first_part.split()
            if 1 <= len(words) <= 5 and not any(k in first_part.lower() for k in ["profile", "home", "posts", "about", "top"]):
                name = first_part
                break

    # Strip surrounding quotes or parentheses
    name = re.sub(r"^[\"\'\(\[\{]+|[\"\'\)\]\}]+$", "", name).strip()
    return name

def extract_handle(url: str, platform: str) -> Optional[str]:
    """Extracts the username or profile handle from a supported platform URL."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return None
        parts = path.split("/")
        
        platform_lower = platform.lower()
        if "linkedin" in platform_lower:
            # path is in/<handle>
            if len(parts) >= 2 and parts[0] == "in":
                return parts[1].lower()
            return parts[-1].lower()
        elif platform_lower in ["x", "twitter"]:
            # path is <handle>
            return parts[0].lower().lstrip("@")
        elif platform_lower == "instagram":
            # path is <handle>
            return parts[0].lower().lstrip("@")
        elif platform_lower == "youtube":
            # path is @<handle> or channel/<id>
            first = parts[0].lower()
            if first.startswith("@"):
                return first.lstrip("@")
            elif len(parts) >= 2 and first in ["channel", "c", "user"]:
                return parts[1].lower()
            return first
        elif platform_lower == "threads":
            # path is @<handle>
            return parts[0].lower().lstrip("@")
        elif platform_lower == "facebook":
            # path is <handle> or profile.php
            first = parts[0].lower()
            if first not in ["pages", "people", "groups"]:
                return first
        return None
    except Exception:
        return None

def normalize_location(location: Optional[str]) -> Optional[str]:
    """Cleans and standardizes location strings."""
    if not location:
        return None
    cleaned = re.sub(r"\s+", " ", location).strip(" ,.-")
    if not cleaned or cleaned.lower() in ["unknown", "n/a", "none"]:
        return None
    return cleaned
