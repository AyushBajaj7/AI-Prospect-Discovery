"""
LLM Service – natural-language query understanding.

Provides multi-provider LLM integration (Gemini, OpenAI, Anthropic) with a
deterministic heuristic fallback that requires no external API key.

Security notes
--------------
* API keys are never logged or included in exception messages.
* LLM responses are JSON-parsed with strict Pydantic validation; a malformed
  or timed-out LLM response triggers the heuristic fallback cleanly.
* User query input is passed via separate user turn / system_instruction so
  it cannot override system prompt instructions.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, Optional

from app.config import settings
from app.schemas import LocationDetails, StructuredIntent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# City / country knowledge bases
# ---------------------------------------------------------------------------
KNOWN_LOCATIONS: Dict[str, Dict[str, Optional[str]]] = {
    "washington dc": {"city": "Washington DC", "state": "District of Columbia", "country": "USA"},
    "washington, d.c.": {"city": "Washington DC", "state": "District of Columbia", "country": "USA"},
    "washington d.c.": {"city": "Washington DC", "state": "District of Columbia", "country": "USA"},
    "new york city": {"city": "New York", "state": "New York", "country": "USA"},
    "new york": {"city": "New York", "state": "New York", "country": "USA"},
    "san francisco": {"city": "San Francisco", "state": "California", "country": "USA"},
    "los angeles": {"city": "Los Angeles", "state": "California", "country": "USA"},
    "chicago": {"city": "Chicago", "state": "Illinois", "country": "USA"},
    "austin": {"city": "Austin", "state": "Texas", "country": "USA"},
    "seattle": {"city": "Seattle", "state": "Washington", "country": "USA"},
    "boston": {"city": "Boston", "state": "Massachusetts", "country": "USA"},
    "miami": {"city": "Miami", "state": "Florida", "country": "USA"},
    "toronto": {"city": "Toronto", "state": "Ontario", "country": "Canada"},
    "vancouver": {"city": "Vancouver", "state": "British Columbia", "country": "Canada"},
    "london": {"city": "London", "state": None, "country": "UK"},
    "manchester": {"city": "Manchester", "state": None, "country": "UK"},
    "birmingham": {"city": "Birmingham", "state": None, "country": "UK"},
    "berlin": {"city": "Berlin", "state": None, "country": "Germany"},
    "munich": {"city": "Munich", "state": None, "country": "Germany"},
    "paris": {"city": "Paris", "state": None, "country": "France"},
    "ahmedabad": {"city": "Ahmedabad", "state": "Gujarat", "country": "India"},
    "bengaluru": {"city": "Bengaluru", "state": "Karnataka", "country": "India"},
    "bangalore": {"city": "Bengaluru", "state": "Karnataka", "country": "India"},
    "mumbai": {"city": "Mumbai", "state": "Maharashtra", "country": "India"},
    "delhi": {"city": "Delhi", "state": None, "country": "India"},
    "new delhi": {"city": "New Delhi", "state": "Delhi", "country": "India"},
    "hyderabad": {"city": "Hyderabad", "state": "Telangana", "country": "India"},
    "pune": {"city": "Pune", "state": "Maharashtra", "country": "India"},
    "dubai": {"city": "Dubai", "state": None, "country": "UAE"},
    "abu dhabi": {"city": "Abu Dhabi", "state": None, "country": "UAE"},
    "singapore": {"city": "Singapore", "state": None, "country": "Singapore"},
    "sydney": {"city": "Sydney", "state": "New South Wales", "country": "Australia"},
    "melbourne": {"city": "Melbourne", "state": "Victoria", "country": "Australia"},
    "tokyo": {"city": "Tokyo", "state": None, "country": "Japan"},
    "beijing": {"city": "Beijing", "state": None, "country": "China"},
    "shanghai": {"city": "Shanghai", "state": None, "country": "China"},
}

# Use whole-word matching only; ordered longest-first to avoid partial hits
KNOWN_COUNTRIES: Dict[str, str] = {
    "united states of america": "USA",
    "united states": "USA",
    "united kingdom": "UK",
    "united arab emirates": "UAE",
    "great britain": "UK",
    "saudi arabia": "Saudi Arabia",
    "new zealand": "New Zealand",
    "south africa": "South Africa",
    "switzerland": "Switzerland",
    "netherlands": "Netherlands",
    "australia": "Australia",
    "germany": "Germany",
    "france": "France",
    "canada": "Canada",
    "spain": "Spain",
    "italy": "Italy",
    "japan": "Japan",
    "china": "China",
    "india": "India",
    "brazil": "Brazil",
    "mexico": "Mexico",
    "singapore": "Singapore",
    "england": "UK",
    # Abbreviations – match as whole words only to avoid substring collisions
    "usa": "USA",
    "u.s.a.": "USA",
    "u.s.": "USA",
    "uae": "UAE",
    "uk": "UK",
}

US_STATE_MAP: Dict[str, str] = {
    "tx": "Texas", "texas": "Texas",
    "ca": "California", "california": "California",
    "ny": "New York",
    "fl": "Florida", "florida": "Florida",
    "wa": "Washington", "washington": "Washington",
    "il": "Illinois", "illinois": "Illinois",
    "ma": "Massachusetts", "massachusetts": "Massachusetts",
    "dc": "District of Columbia",
}

_STOPWORDS = frozenset({"and", "the", "for", "with", "all", "are", "who", "from", "based", "near"})

# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert NLP parser for a prospect-discovery engine.

Analyse the user's natural-language search query and return ONLY valid JSON
with no markdown code fences, no extra text, and no trailing commas.

Required JSON schema:
{
  "interest":    "<primary topic/interest, e.g. Football | Cricket | Startups | null>",
  "audience":    "<target audience or profession, e.g. Football lovers | Startup founders>",
  "location": {
    "city":    "<city name or null>",
    "state":   "<state/province or null>",
    "country": "<canonical country name: USA | UK | UAE | India | etc., or null>"
  },
  "target_type": "<individual | business>",
  "keywords":    ["<relevant keyword>", ...]
}

Rules:
1. target_type is "business" when the query primarily targets companies,
   agencies, or local businesses; use "individual" for fans/enthusiasts/founders.
2. Keep canonical short forms for USA, UK, UAE when returning country; expand DC
   to "District of Columbia" for state when city is Washington DC.
3. Return JSON null for fields that cannot be determined — never guess.
4. Never include commentary, explanations, or markdown. Only raw JSON.
"""


def _canonicalize_country(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    key = country.strip().lower()
    return KNOWN_COUNTRIES.get(key, country.strip())


def _canonicalize_intent(intent: StructuredIntent) -> StructuredIntent:
    if intent.location.country:
        intent.location.country = _canonicalize_country(intent.location.country)
    return intent


class LLMService:
    def __init__(self) -> None:
        self.stats: Dict[str, Any] = {
            "calls": 0,
            "total_latency_ms": 0.0,
            "last_latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "provider_used": "none",
        }

    # ------------------------------------------------------------------
    # Dynamic location parser helper (used when city is not in dictionary)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_dynamic_location(
        loc_clause: str,
        city: Optional[str],
        state: Optional[str],
        country: Optional[str],
    ) -> LocationDetails:
        if not loc_clause:
            return LocationDetails(city=city, state=state, country=country)

        # Clean out matched country & connectors like "in "
        remaining = re.sub(r"\bin\b", ",", loc_clause, flags=re.IGNORECASE)
        for c_token in sorted(KNOWN_COUNTRIES.keys(), key=len, reverse=True):
            remaining = re.sub(
                rf"\b{re.escape(c_token)}\b",
                "",
                remaining,
                flags=re.IGNORECASE,
            )

        segments = [
            re.sub(r"\s+", " ", seg).strip(" ,.-")
            for seg in remaining.split(",")
            if seg.strip(" ,.-")
        ]

        if not city and segments:
            first_seg = segments[0]
            # Check if first_seg ends with a state token, e.g. "Miami Florida"
            words = first_seg.split()
            if len(words) >= 2 and words[-1].lower() in US_STATE_MAP and not state:
                state = US_STATE_MAP[words[-1].lower()]
                country = country or "USA"
                city = " ".join(words[:-1]).title()
            else:
                city = first_seg.title() if first_seg.islower() else first_seg

            if len(segments) >= 2 and not state:
                second_seg = segments[1]
                st_mapped = US_STATE_MAP.get(second_seg.lower())
                if st_mapped:
                    state = st_mapped
                    country = country or "USA"
                else:
                    state = second_seg

        return LocationDetails(city=city, state=state, country=country)

    # ------------------------------------------------------------------
    # Deterministic heuristic fallback (no external dependency)
    # ------------------------------------------------------------------
    def _parse_heuristic(self, query: str) -> StructuredIntent:
        """
        Rule-based intent extractor. Handles both well-known locations and
        arbitrary ad-hoc cities/countries dynamically without hardcoding queries.
        """
        q_clean = re.sub(r"\s+", " ", query.strip())
        q_lower = q_clean.lower()

        # Split out audience vs location clause on prepositions
        prep_match = re.split(
            r"\s+(?:based\s+in|located\s+in|from|near|in)\s+",
            q_clean,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        audience_part = prep_match[0].strip() or q_clean
        loc_clause = prep_match[1].strip() if len(prep_match) > 1 else ""

        # --- 1. Location extraction ---
        city: Optional[str] = None
        state: Optional[str] = None
        country: Optional[str] = None

        # Check known city dictionary (longest-first)
        search_loc_area = (loc_clause or q_clean).lower()
        for loc_key in sorted(KNOWN_LOCATIONS.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(loc_key)}\b", search_loc_area):
                loc_data = KNOWN_LOCATIONS[loc_key]
                city = loc_data["city"]
                state = loc_data["state"]
                country = loc_data["country"]
                break

        # Check country dictionary (whole-word, longest-first)
        if not country:
            for c_token in sorted(KNOWN_COUNTRIES.keys(), key=len, reverse=True):
                if re.search(rf"(?<!\w){re.escape(c_token)}(?!\w)", search_loc_area):
                    country = KNOWN_COUNTRIES[c_token]
                    break

        # Dynamic fallback for cities/states not present in KNOWN_LOCATIONS
        if not city and loc_clause:
            location_details = self._extract_dynamic_location(
                loc_clause, city, state, country
            )
        else:
            location_details = LocationDetails(city=city, state=state, country=country)

        # --- 2. Audience extraction ---
        audience = audience_part

        # --- 3. Interest + target_type classification ---
        interest: Optional[str] = None
        target_type: str = "individual"

        a_lower = audience_part.lower()

        if re.search(r"\bfootball\b", a_lower):
            interest = "Football"
        elif re.search(r"\bcricket\b", a_lower):
            interest = "Cricket"
        elif re.search(r"\bdigital\s+marketing\b|\bmarketing\b", a_lower):
            interest = "Digital Marketing"
            target_type = "business" if "agenc" in a_lower else "individual"
        elif re.search(r"\bstartup\b|\bfounder", a_lower):
            interest = "Startups"
        elif re.search(r"\brestaurant\b", a_lower):
            interest = "Hospitality & Restaurant"
            target_type = "individual"
        elif re.search(r"\be-?commerce\b|\bretail\b", a_lower):
            interest = "E-Commerce & Retail"
            target_type = "business"
        else:
            # Strip common role nouns to isolate the core topic/interest dynamically
            topic_words = [
                w
                for w in re.findall(r"\b[A-Za-z][A-Za-z0-9\-]+\b", audience_part)
                if w.lower() not in _STOPWORDS
                and w.lower() not in {"lovers", "fans", "enthusiasts", "owners", "founders", "people"}
            ]
            interest = " ".join(topic_words) if topic_words else audience_part
            if any(term in a_lower for term in ("agencies", "agency", "companies", "studios", "firms")):
                target_type = "business"

        keywords = [
            w
            for w in re.findall(r"\b[A-Za-z][A-Za-z0-9]+\b", audience_part)
            if len(w) > 2 and w.lower() not in _STOPWORDS
        ]

        return StructuredIntent(
            interest=interest,
            audience=audience,
            location=location_details,
            target_type=target_type,  # type: ignore[arg-type]
            keywords=keywords,
        )

    # ------------------------------------------------------------------
    # LLM-backed parser with multi-provider support
    # ------------------------------------------------------------------
    async def parse_query(self, query: str) -> StructuredIntent:
        """
        Attempts LLM-based parsing with the configured provider, then falls
        back to the deterministic heuristic on any failure or timeout.
        """
        start = time.perf_counter()
        provider = settings.LLM_PROVIDER
        timeout_sec = float(settings.SEARCH_TIMEOUT_SECONDS)

        if provider == "auto":
            if settings.GEMINI_API_KEY:
                provider = "gemini"
            elif settings.OPENAI_API_KEY:
                provider = "openai"
            elif settings.ANTHROPIC_API_KEY:
                provider = "anthropic"
            else:
                provider = "fallback"

        result_intent: Optional[StructuredIntent] = None

        # --- Gemini ---
        if provider == "gemini" and settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=settings.GEMINI_API_KEY)
                model = genai.GenerativeModel(
                    settings.GEMINI_MODEL,
                    system_instruction=SYSTEM_PROMPT,
                    generation_config={"response_mime_type": "application/json"},
                )
                response = await asyncio.wait_for(
                    model.generate_content_async(query),
                    timeout=timeout_sec,
                )
                data = json.loads(response.text)
                result_intent = _canonicalize_intent(StructuredIntent(**data))
                self.stats["provider_used"] = "gemini"
            except Exception as exc:
                logger.warning("Gemini parse failed (%s); using heuristic.", type(exc).__name__)

        # --- OpenAI ---
        elif provider == "openai" and settings.OPENAI_API_KEY:
            try:
                from openai import AsyncOpenAI  # type: ignore

                client = AsyncOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    timeout=timeout_sec,
                )
                resp = await client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
                content = resp.choices[0].message.content or "{}"
                data = json.loads(content)
                result_intent = _canonicalize_intent(StructuredIntent(**data))
                if resp.usage:
                    self.stats["prompt_tokens"] += resp.usage.prompt_tokens
                    self.stats["completion_tokens"] += resp.usage.completion_tokens
                self.stats["provider_used"] = "openai"
            except Exception as exc:
                logger.warning("OpenAI parse failed (%s); using heuristic.", type(exc).__name__)

        # --- Anthropic ---
        elif provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            try:
                from anthropic import AsyncAnthropic  # type: ignore

                client = AsyncAnthropic(
                    api_key=settings.ANTHROPIC_API_KEY,
                    timeout=timeout_sec,
                )
                resp = await client.messages.create(
                    model=settings.ANTHROPIC_MODEL,
                    max_tokens=400,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": query}],
                )
                raw_text: str = resp.content[0].text
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    result_intent = _canonicalize_intent(StructuredIntent(**data))
                    if getattr(resp, "usage", None):
                        self.stats["prompt_tokens"] += resp.usage.input_tokens
                        self.stats["completion_tokens"] += resp.usage.output_tokens
                    self.stats["provider_used"] = "anthropic"
            except Exception as exc:
                logger.warning("Anthropic parse failed (%s); using heuristic.", type(exc).__name__)

        # --- Heuristic fallback ---
        if result_intent is None:
            result_intent = self._parse_heuristic(query)
            self.stats["provider_used"] = "heuristic_fallback"

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stats["calls"] += 1
        self.stats["last_latency_ms"] = elapsed_ms
        self.stats["total_latency_ms"] += elapsed_ms

        return result_intent


llm_service = LLMService()
