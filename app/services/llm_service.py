import json
import logging
import re
import time
from typing import Optional, Dict, Any, Tuple
from app.config import settings
from app.schemas import StructuredIntent, LocationDetails

logger = logging.getLogger(__name__)

KNOWN_LOCATIONS = {
    "washington dc": {"city": "Washington DC", "state": "District of Columbia", "country": "USA"},
    "new york": {"city": "New York", "state": "New York", "country": "USA"},
    "london": {"city": "London", "state": None, "country": "UK"},
    "ahmedabad": {"city": "Ahmedabad", "state": "Gujarat", "country": "India"},
    "dubai": {"city": "Dubai", "state": None, "country": "UAE"},
    "san francisco": {"city": "San Francisco", "state": "California", "country": "USA"},
    "los angeles": {"city": "Los Angeles", "state": "California", "country": "USA"},
    "chicago": {"city": "Chicago", "state": "Illinois", "country": "USA"},
    "toronto": {"city": "Toronto", "state": "Ontario", "country": "Canada"},
    "berlin": {"city": "Berlin", "state": None, "country": "Germany"},
    "paris": {"city": "Paris", "state": None, "country": "France"},
    "mumbai": {"city": "Mumbai", "state": "Maharashtra", "country": "India"},
    "bengaluru": {"city": "Bengaluru", "state": "Karnataka", "country": "India"},
    "bangalore": {"city": "Bengaluru", "state": "Karnataka", "country": "India"},
    "singapore": {"city": "Singapore", "state": None, "country": "Singapore"},
    "sydney": {"city": "Sydney", "state": "New South Wales", "country": "Australia"},
    "tokyo": {"city": "Tokyo", "state": None, "country": "Japan"},
}

KNOWN_COUNTRIES = {
    "usa": "USA", "united states": "USA", "us": "USA",
    "uk": "UK", "united kingdom": "UK", "britain": "UK",
    "uae": "UAE", "united arab emirates": "UAE",
    "india": "India", "canada": "Canada", "germany": "Germany",
    "france": "France", "australia": "Australia", "japan": "Japan"
}

SYSTEM_PROMPT = """You are an expert NLP parser for a prospect discovery engine.
Your task is to analyze a natural-language search query and return a valid JSON object extracting structured search parameters.

Return strictly valid JSON with the following structure:
{
  "interest": "<primary topic or interest, e.g. Football, Cricket, Tech>",
  "audience": "<target audience or profession, e.g. Football lovers, Startup founders, Restaurant owners>",
  "location": {
    "city": "<city or null>",
    "state": "<state or null>",
    "country": "<country or null>"
  },
  "target_type": "<individual or business>",
  "keywords": ["<keyword1>", "<keyword2>"]
}

Guidelines:
- If the query mentions 'owners', 'founders', 'agency owners', 'restaurant owners', set target_type to 'business' if looking for enterprises or 'individual' if looking for the persons. For owners/founders looking for people, set 'individual'.
- Extract standard geographical names (e.g. Washington DC -> state: District of Columbia, country: USA).
- No markdown wrapping, no extra text. Just raw JSON.
"""

class LLMService:
    def __init__(self):
        self.stats = {
            "calls": 0,
            "total_latency_ms": 0,
            "last_latency_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "provider_used": "none"
        }

    def _parse_heuristic(self, query: str) -> StructuredIntent:
        """Deterministic NLP parser for high accuracy without external API dependencies."""
        q_lower = query.lower().strip()
        
        # 1. Location extraction
        city, state, country = None, None, None
        
        # Check known country tokens
        for c_token, c_name in KNOWN_COUNTRIES.items():
            pattern = rf"\b{re.escape(c_token)}\b"
            if re.search(pattern, q_lower):
                country = c_name
                break

        # Check known city/location combos
        for loc_key, loc_data in KNOWN_LOCATIONS.items():
            pattern = rf"\b{re.escape(loc_key)}\b"
            if re.search(pattern, q_lower):
                city = loc_data["city"]
                state = loc_data["state"] or state
                country = loc_data["country"] or country
                break

        # Fallback regex if "in <Location>" format
        if not city:
            loc_match = re.search(r"\bin\s+([A-Za-z\s]+?)(?:\s+in\s+([A-Za-z\s]+))?$", query, re.IGNORECASE)
            if loc_match:
                candidate_1 = loc_match.group(1).strip()
                candidate_2 = loc_match.group(2).strip() if loc_match.group(2) else None
                if candidate_2:
                    city = candidate_1
                    country = candidate_2
                else:
                    city = candidate_1

        location_details = LocationDetails(city=city, state=state, country=country)

        # 2. Audience and Interest extraction
        # Cut off location prepositions like "in <city>"
        audience_part = re.split(r"\s+in\s+", query, flags=re.IGNORECASE)[0].strip()
        audience = audience_part if audience_part else query

        # Interest deduction
        interest = None
        target_type = "individual"

        if re.search(r"football", audience_part, re.IGNORECASE):
            interest = "Football"
        elif re.search(r"cricket", audience_part, re.IGNORECASE):
            interest = "Cricket"
        elif re.search(r"marketing|digital marketing", audience_part, re.IGNORECASE):
            interest = "Digital Marketing"
            target_type = "business"
        elif re.search(r"startup|founder", audience_part, re.IGNORECASE):
            interest = "Startups"
        elif re.search(r"restaurant", audience_part, re.IGNORECASE):
            interest = "Hospitality / Restaurant"
            target_type = "business"
        else:
            interest = audience_part

        keywords = [w for w in re.findall(r"\b[A-Za-z0-9]+\b", audience_part) if len(w) > 2]

        return StructuredIntent(
            interest=interest,
            audience=audience,
            location=location_details,
            target_type=target_type,
            keywords=keywords
        )

    async def parse_query(self, query: str) -> StructuredIntent:
        """Parses query using configured LLM provider, falling back safely to deterministic heuristics."""
        start_time = time.perf_counter()
        provider = settings.LLM_PROVIDER.lower()
        
        # Auto-detect provider if auto
        if provider == "auto":
            if settings.GEMINI_API_KEY:
                provider = "gemini"
            elif settings.OPENAI_API_KEY:
                provider = "openai"
            elif settings.ANTHROPIC_API_KEY:
                provider = "anthropic"
            else:
                provider = "fallback"

        result_intent = None

        if provider == "gemini" and settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                model = genai.GenerativeModel(
                    settings.GEMINI_MODEL,
                    generation_config={"response_mime_type": "application/json"}
                )
                prompt = f"{SYSTEM_PROMPT}\n\nQuery to parse: {query}"
                response = await model.generate_content_async(prompt)
                data = json.loads(response.text)
                result_intent = StructuredIntent(**data)
                self.stats["provider_used"] = "gemini"
            except Exception as e:
                logger.warning(f"Gemini API parse failed ({e}), falling back to heuristic")

        elif provider == "openai" and settings.OPENAI_API_KEY:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                response = await client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Query: {query}"}
                    ],
                    temperature=0.0
                )
                content = response.choices[0].message.content
                data = json.loads(content)
                result_intent = StructuredIntent(**data)
                if response.usage:
                    self.stats["prompt_tokens"] += response.usage.prompt_tokens
                    self.stats["completion_tokens"] += response.usage.completion_tokens
                self.stats["provider_used"] = "openai"
            except Exception as e:
                logger.warning(f"OpenAI API parse failed ({e}), falling back to heuristic")

        elif provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            try:
                from anthropic import AsyncAnthropic
                client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
                response = await client.messages.create(
                    model=settings.ANTHROPIC_MODEL,
                    max_tokens=500,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": f"Query: {query}"}]
                )
                text = response.content[0].text
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    result_intent = StructuredIntent(**data)
                    self.stats["provider_used"] = "anthropic"
            except Exception as e:
                logger.warning(f"Anthropic API parse failed ({e}), falling back to heuristic")

        if not result_intent:
            result_intent = self._parse_heuristic(query)
            self.stats["provider_used"] = "heuristic_fallback"

        duration_ms = (time.perf_counter() - start_time) * 1000
        self.stats["calls"] += 1
        self.stats["last_latency_ms"] = duration_ms
        self.stats["total_latency_ms"] += duration_ms

        return result_intent

llm_service = LLMService()
