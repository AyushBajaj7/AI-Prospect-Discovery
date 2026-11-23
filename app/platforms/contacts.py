import re
from typing import List, Dict, Set
from app.utils.validators import is_valid_email, is_valid_phone

# Email pattern matching in text snippets
SNIPPET_EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    re.IGNORECASE
)

# Phone pattern matching in text snippets
SNIPPET_PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,5}",
    re.IGNORECASE
)

# WhatsApp links in text or urls
WHATSAPP_LINK_REGEX = re.compile(
    r"https?://(?:wa\.me/\+?\d+|api\.whatsapp\.com/send\?phone=\d+|chat\.whatsapp\.com/[a-zA-Z0-9]+)",
    re.IGNORECASE
)

def extract_emails(text: str) -> List[str]:
    """Finds all valid public email addresses in text."""
    if not text:
        return []
    matches = SNIPPET_EMAIL_REGEX.findall(text)
    valid_emails: Set[str] = set()
    for m in matches:
        cleaned = m.strip(".,;:()[]{}'\"")
        if is_valid_email(cleaned):
            valid_emails.add(cleaned.lower())
    return sorted(list(valid_emails))

def extract_phones(text: str) -> List[str]:
    """Finds all plausible public phone numbers in text."""
    if not text:
        return []
    matches = SNIPPET_PHONE_REGEX.findall(text)
    valid_phones: Set[str] = set()
    for m in matches:
        cleaned = m.strip(".,;:()[]{}'\"")
        if is_valid_phone(cleaned):
            valid_phones.add(cleaned)
    return sorted(list(valid_phones))

def extract_whatsapp_links(text: str) -> List[str]:
    """Finds all publicly published WhatsApp links in text."""
    if not text:
        return []
    matches = WHATSAPP_LINK_REGEX.findall(text)
    valid_links: Set[str] = set()
    for m in matches:
        cleaned = m.strip(".,;:()[]{}'\"")
        valid_links.add(cleaned)
    return sorted(list(valid_links))

def extract_all_contacts(text: str) -> Dict[str, List[str]]:
    return {
        "emails": extract_emails(text),
        "phones": extract_phones(text),
        "whatsapp_links": extract_whatsapp_links(text)
    }
