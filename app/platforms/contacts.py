"""
Contact extractor – finds publicly published email addresses, phone numbers,
and WhatsApp links inside free-text snippets and titles.

Only publicly visible information is extracted; nothing is guessed or generated.
"""
from __future__ import annotations

import re
from typing import Dict, List

from app.utils.validators import is_valid_email, is_valid_phone

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Supports multi-part TLDs (e.g. alex@company.co.uk) as well as standard TLDs
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9_%+\-]+(?:\.[a-zA-Z0-9_%+\-]+)*"
    r"@"
    r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2,24}",
    re.IGNORECASE,
)

# Conservative phone pattern requiring separator formatting or leading '+'.
# Avoids matching plain integers, follower counts, or year ranges (further validated).
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"\+[0-9]{1,3}[\s.\-]?(?:\(?[0-9]{2,4}\)?[\s.\-]?){1,3}[0-9]{3,5}"
    r"|"
    r"(?:\([0-9]{2,4}\)[\s.\-]?|[0-9]{2,4}[\s.\-])(?:[0-9]{2,4}[\s.\-]?){1,2}[0-9]{3,5}"
    r")"
    r"(?!\d)",
)

_WHATSAPP_RE = re.compile(
    r"https?://(?:wa\.me/\+?[0-9]{7,15}|api\.whatsapp\.com/send\?phone=\+?[0-9]{7,15})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_emails(text: str) -> List[str]:
    """Return a deduplicated list of valid public email addresses found in text."""
    if not text:
        return []
    seen: set[str] = set()
    result: List[str] = []
    for match in _EMAIL_RE.finditer(text):
        email = match.group(0).strip(".,;:'\"")
        email_lower = email.lower()
        if email_lower not in seen and is_valid_email(email_lower):
            seen.add(email_lower)
            result.append(email_lower)
    return result


def extract_phones(text: str) -> List[str]:
    """Return a deduplicated list of plausible public phone numbers found in text."""
    if not text:
        return []
    seen: set[str] = set()
    result: List[str] = []
    for match in _PHONE_RE.finditer(text):
        phone = match.group(0).strip(".,;:'\" ")
        if phone not in seen and is_valid_phone(phone):
            seen.add(phone)
            result.append(phone)
    return result


def extract_whatsapp_links(text: str) -> List[str]:
    """Return a deduplicated list of public WhatsApp links found in text."""
    if not text:
        return []
    seen: set[str] = set()
    result: List[str] = []
    for match in _WHATSAPP_RE.finditer(text):
        link = match.group(0).strip(".,;:'\" ")
        if link not in seen:
            seen.add(link)
            result.append(link)
    return result


def extract_all_contacts(text: str) -> Dict[str, List[str]]:
    """Convenience wrapper – returns all three contact categories at once."""
    return {
        "emails": extract_emails(text),
        "phones": extract_phones(text),
        "whatsapp_links": extract_whatsapp_links(text),
    }
