"""
Contact extractor – finds publicly published email addresses, phone numbers,
and normalized WhatsApp links inside free-text snippets and titles.

Input text length is capped before regex matching to eliminate ReDoS exposure.
"""
from __future__ import annotations

import re
from typing import Dict, List
from urllib.parse import urlparse

from app.utils.validators import (
    is_generic_role_email,
    is_valid_email,
    is_valid_phone,
    normalize_whatsapp_url,
)

_MAX_SCAN_CHARS = 10_000

# ---------------------------------------------------------------------------
# Compiled patterns (bounded repetitions for linear-time matching)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9_%+\-]{1,64}(?:\.[a-zA-Z0-9_%+\-]{1,64}){0,8}"
    r"@"
    r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.){1,5}"
    r"[a-zA-Z]{2,24}",
    re.IGNORECASE,
)

# Conservative phone pattern requiring separator formatting or leading '+'.
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
    bounded = text[:_MAX_SCAN_CHARS]
    seen: set[str] = set()
    result: List[str] = []
    for match in _EMAIL_RE.finditer(bounded):
        email = match.group(0).strip(".,;:'\"").lower()
        if email not in seen and is_valid_email(email):
            seen.add(email)
            result.append(email)
    return result


def extract_phones(text: str) -> List[str]:
    """Return a deduplicated list of plausible public phone numbers found in text."""
    if not text:
        return []
    bounded = text[:_MAX_SCAN_CHARS]
    seen: set[str] = set()
    result: List[str] = []
    for match in _PHONE_RE.finditer(bounded):
        phone = match.group(0).strip(".,;:'\" ")
        if phone not in seen and is_valid_phone(phone):
            seen.add(phone)
            result.append(phone)
    return result


def extract_whatsapp_links(text: str) -> List[str]:
    """
    Return deduplicated, officially normalized public WhatsApp links
    (`https://wa.me/<digits>`, no '+' sign).
    """
    if not text:
        return []
    bounded = text[:_MAX_SCAN_CHARS]
    seen: set[str] = set()
    result: List[str] = []
    for match in _WHATSAPP_RE.finditer(bounded):
        normalized = normalize_whatsapp_url(match.group(0).strip(".,;:'\" "))
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def extract_all_contacts(text: str) -> Dict[str, List[str]]:
    """Convenience wrapper – returns all three contact categories at once."""
    return {
        "emails": extract_emails(text),
        "phones": extract_phones(text),
        "whatsapp_links": extract_whatsapp_links(text),
    }


def filter_attributed_emails(
    emails: List[str],
    entity_type: str,
    profile_url: str,
) -> List[str]:
    """
    Distinguish personal emails from organization/role-based addresses (`info@`,
    `cofounder@`, `team@`, `sales@`, etc.):
      * Business entities retain organization contact emails.
      * Individual and unknown candidates drop generic role emails unless the
        profile URL itself is a custom first-party site on the email domain.
    """
    if not emails or entity_type == "business":
        return list(emails)

    try:
        profile_host = (urlparse(profile_url or "").hostname or "").lower().removeprefix("www.")
    except Exception:
        profile_host = ""

    kept: List[str] = []
    for email in emails:
        if not is_generic_role_email(email):
            kept.append(email)
            continue
        email_domain = email.partition("@")[2].lower().removeprefix("www.")
        if (
            profile_host
            and email_domain
            and (profile_host == email_domain or profile_host.endswith("." + email_domain))
        ):
            kept.append(email)
    return kept
