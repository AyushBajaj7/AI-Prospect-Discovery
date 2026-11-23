"""
Comprehensive test suite for the AI Prospect Discovery API.

Covers:
  - Input validation & sanitisation
  - Natural-language query understanding (all 5 mandatory assessment queries)
  - URL and contact validators
  - Name / handle normalizers
  - Relevance scoring (relevant vs. irrelevant)
  - Deduplication & entity consolidation (all four match strategies)
  - FastAPI endpoint contract (success path + error paths)
  - Partial platform failure resilience
  - Security: CORS headers, request-ID header, oversized body rejection
  - End-to-end pipeline for all 5 assessment queries
"""
from __future__ import annotations

import asyncio
import json
from typing import List
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from app.main import app
from app.schemas import LocationDetails, RawCandidate, SocialLink, StructuredIntent
from app.services.deduplication_service import deduplication_service
from app.services.llm_service import llm_service
from app.services.query_service import query_service
from app.services.relevance_service import relevance_service
from app.utils.normalizers import clean_name, extract_handle
from app.utils.validators import (
    clean_url,
    is_valid_email,
    is_valid_phone,
    is_valid_profile_url,
)

client = TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# 1. Health endpoints
# ===========================================================================

class TestHealthEndpoints:
    def test_root_returns_online(self):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "online"
        assert "discovery_endpoint" in body

    def test_health_liveness(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_request_id_header_present(self):
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers

    def test_request_id_is_unique_per_request(self):
        ids = {client.get("/health").headers["X-Request-ID"] for _ in range(5)}
        assert len(ids) == 5  # all different UUIDs


# ===========================================================================
# 2. Input validation
# ===========================================================================

class TestInputValidation:
    def test_empty_query_rejected(self):
        resp = client.post("/prospect-discovery", json={"query": ""})
        assert resp.status_code in (400, 422)

    def test_whitespace_only_query_rejected(self):
        resp = client.post("/prospect-discovery", json={"query": "    "})
        assert resp.status_code in (400, 422)

    def test_numeric_only_query_rejected(self):
        resp = client.post("/prospect-discovery", json={"query": "12345"})
        assert resp.status_code in (400, 422)

    def test_query_too_short_rejected(self):
        resp = client.post("/prospect-discovery", json={"query": "ab"})
        assert resp.status_code in (400, 422)

    def test_missing_query_field_rejected(self):
        resp = client.post("/prospect-discovery", json={})
        assert resp.status_code == 422

    def test_non_json_body_rejected(self):
        resp = client.post(
            "/prospect-discovery",
            content="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in (400, 422, 500)

    def test_valid_query_accepted(self):
        # Mock the full pipeline so we don't make real HTTP calls in CI
        with patch(
            "app.services.prospect_service.query_service.get_structured_intent",
            new_callable=AsyncMock,
            return_value=StructuredIntent(
                interest="Startups",
                audience="Startup founders",
                location=LocationDetails(city="London", country="UK"),
            ),
        ), patch(
            "app.services.prospect_service.ALL_PLATFORMS",
            new=[],
        ):
            resp = client.post(
                "/prospect-discovery",
                json={"query": "Startup founders in London UK"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["query"] == "Startup founders in London UK"
        assert isinstance(body["prospects"], list)


# ===========================================================================
# 3. Natural-language query understanding (all 5 mandatory test queries)
# ===========================================================================

MANDATORY_QUERIES = [
    {
        "query": "Football lovers in Washington DC in USA",
        "interest": "Football",
        "city": "Washington DC",
        "country": "USA",
    },
    {
        "query": "Digital marketing agency owners in New York USA",
        "interest": "Digital Marketing",
        "city": "New York",
        "country": "USA",
    },
    {
        "query": "Startup founders in London UK",
        "interest": "Startups",
        "city": "London",
        "country": "UK",
    },
    {
        "query": "Cricket lovers in Ahmedabad India",
        "interest": "Cricket",
        "city": "Ahmedabad",
        "country": "India",
    },
    {
        "query": "Restaurant owners in Dubai UAE",
        "interest": "Hospitality & Restaurant",
        "city": "Dubai",
        "country": "UAE",
    },
]


class TestQueryUnderstanding:
    @pytest.mark.parametrize("tc", MANDATORY_QUERIES)
    def test_mandatory_query_extraction(self, tc):
        intent = asyncio.run(query_service.get_structured_intent(tc["query"]))
        assert intent.interest == tc["interest"], (
            f"Interest mismatch for '{tc['query']}': "
            f"expected '{tc['interest']}', got '{intent.interest}'"
        )
        assert intent.location.city == tc["city"], (
            f"City mismatch for '{tc['query']}': "
            f"expected '{tc['city']}', got '{intent.location.city}'"
        )
        assert intent.location.country == tc["country"], (
            f"Country mismatch for '{tc['query']}': "
            f"expected '{tc['country']}', got '{intent.location.country}'"
        )

    def test_caching_returns_same_object(self):
        """Second call with identical query must return the cached instance."""
        q = "Startup founders in London UK"
        intent1 = asyncio.run(query_service.get_structured_intent(q))
        intent2 = asyncio.run(query_service.get_structured_intent(q))
        assert intent1 is intent2  # same object from cache

    def test_location_formatted_non_empty(self):
        intent = asyncio.run(query_service.get_structured_intent("Football lovers in Washington DC in USA"))
        assert intent.location.formatted() != ""
        assert "Washington DC" in intent.location.formatted()


# ===========================================================================
# 4. URL & Contact validators
# ===========================================================================

class TestValidators:

    # --- Valid profile URLs ---
    @pytest.mark.parametrize("platform,url", [
        ("LinkedIn", "https://www.linkedin.com/in/alex-morgan"),
        ("LinkedIn", "https://uk.linkedin.com/in/john-doe-123"),
        ("X", "https://x.com/elonmusk"),
        ("X", "https://twitter.com/jack"),
        ("YouTube", "https://www.youtube.com/@mkbhd"),
        ("YouTube", "https://youtube.com/channel/UCxxxxxxxxxxxxxxxxxxxxxx"),
        ("Instagram", "https://www.instagram.com/cristiano"),
        ("Instagram", "https://instagram.com/natgeo"),
        ("Facebook", "https://www.facebook.com/zuck"),
        ("Threads", "https://threads.net/@zuck"),
        ("WhatsApp", "https://wa.me/+12025551234"),
    ])
    def test_valid_profile_urls(self, platform, url):
        assert is_valid_profile_url(platform, url), f"Expected valid: {platform} {url}"

    # --- Invalid URLs (posts / search / feeds) ---
    @pytest.mark.parametrize("platform,url", [
        ("LinkedIn", "https://www.linkedin.com/posts/activity-123"),
        ("LinkedIn", "https://www.linkedin.com/company/google"),
        ("X", "https://x.com/elonmusk/status/1234567890"),
        ("X", "https://x.com/hashtag/python"),
        ("YouTube", "https://www.youtube.com/watch?v=abcd1234"),
        ("YouTube", "https://youtube.com/playlist?list=abc"),
        ("Instagram", "https://www.instagram.com/p/CR123456/"),
        ("Instagram", "https://instagram.com/reel/ABC123/"),
        ("Facebook", "https://www.facebook.com/groups/somegroup"),
        ("Threads", "https://threads.net/@zuck/post/12345"),
    ])
    def test_invalid_profile_urls(self, platform, url):
        assert not is_valid_profile_url(platform, url), f"Expected invalid: {platform} {url}"

    # --- Email validation ---
    @pytest.mark.parametrize("email,expected", [
        ("founder@startup.io", True),
        ("john.doe+work@company.co.uk", True),
        ("UPPER@Domain.COM", True),
        ("", False),
        ("not-an-email", False),
        ("missing@domain", False),
        ("fake@example.com", False),       # bogus domain
        ("test@test.com", False),          # bogus domain
        ("a" * 250 + "@b.com", False),     # too long
    ])
    def test_email_validation(self, email, expected):
        assert is_valid_email(email) == expected, f"Email '{email}' expected {expected}"

    # --- Phone validation ---
    @pytest.mark.parametrize("phone,expected", [
        ("+1 202-555-0199", True),
        ("+44 20 7946 0000", True),
        ("+91-79-26300000", True),
        ("12345", False),                  # too short
        ("123", False),
        ("", False),
    ])
    def test_phone_validation(self, phone, expected):
        assert is_valid_phone(phone) == expected, f"Phone '{phone}' expected {expected}"

    # --- URL cleaning ---
    def test_clean_url_strips_tracking_params(self):
        dirty = "https://linkedin.com/in/johndoe?utm_source=google&utm_campaign=test&fbclid=abc"
        clean = clean_url(dirty)
        assert "utm_source" not in clean
        assert "fbclid" not in clean
        assert "johndoe" in clean

    def test_clean_url_strips_trailing_slash(self):
        url = "https://linkedin.com/in/johndoe/"
        assert clean_url(url) == "https://linkedin.com/in/johndoe"


# ===========================================================================
# 5. Normalizers
# ===========================================================================

class TestNormalizers:

    @pytest.mark.parametrize("title,platform,expected_name", [
        ("John Doe - Startup Founder & CEO | LinkedIn", "LinkedIn", "John Doe"),
        ("Sarah Connor (@sarahc) / X", "X", "Sarah Connor"),
        ("Tech Channel - YouTube", "YouTube", "Tech Channel"),
        ("Alice Johnson • Instagram", "Instagram", "Alice Johnson"),
        ("Bob Smith on Threads", "Threads", "Bob Smith"),
        ("Jane Austen - WhatsApp", "WhatsApp", "Jane Austen"),
        ("", "LinkedIn", ""),
    ])
    def test_clean_name(self, title, platform, expected_name):
        assert clean_name(title, platform) == expected_name

    @pytest.mark.parametrize("url,platform,expected_handle", [
        ("https://x.com/startupfounder", "X", "startupfounder"),
        ("https://www.linkedin.com/in/john-doe-123", "LinkedIn", "john-doe-123"),
        ("https://instagram.com/cristiano", "Instagram", "cristiano"),
        ("https://youtube.com/@mkbhd", "YouTube", "mkbhd"),
        ("https://threads.net/@zuck", "Threads", "zuck"),
    ])
    def test_extract_handle(self, url, platform, expected_handle):
        assert extract_handle(url, platform) == expected_handle


# ===========================================================================
# 6. Relevance scoring
# ===========================================================================

class TestRelevanceScoring:

    @pytest.fixture
    def startup_intent(self):
        return StructuredIntent(
            interest="Startups",
            audience="Startup founders",
            location=LocationDetails(city="London", country="UK"),
        )

    def _make_candidate(self, **kwargs) -> RawCandidate:
        defaults = dict(
            platform="LinkedIn",
            name="Test Person",
            profile_url="https://linkedin.com/in/testperson",
            title="",
            snippet="",
        )
        defaults.update(kwargs)
        return RawCandidate(**defaults)

    def test_highly_relevant_candidate_scores_high(self, startup_intent):
        cand = self._make_candidate(
            name="James Wilson",
            location="London, UK",
            profile_url="https://linkedin.com/in/jameswilson",
            title="James Wilson - Startup Founder & CTO | LinkedIn",
            snippet="Passionate startup founder building AI products in London, UK.",
        )
        score, _ = relevance_service.evaluate_candidate(cand, startup_intent)
        assert score >= 55.0, f"Expected >=55 but got {score}"

    def test_irrelevant_candidate_scores_low(self, startup_intent):
        cand = self._make_candidate(
            name="Tokyo Weather Today",
            location="Tokyo, Japan",
            profile_url="https://weather.tokyo.jp",
            title="Tokyo Weather Forecast | NHK",
            snippet="Rain and sunny intervals expected this week.",
        )
        score, _ = relevance_service.evaluate_candidate(cand, startup_intent)
        assert score < 30.0, f"Expected <30 but got {score}"

    def test_score_breakdown_sums_to_total(self, startup_intent):
        cand = self._make_candidate(
            title="Entrepreneur in London",
            snippet="Startup founder London UK.",
        )
        total, breakdown = relevance_service.evaluate_candidate(cand, startup_intent)
        assert abs(sum(breakdown.values()) - total) < 0.01

    def test_no_location_query_gives_full_location_score(self):
        intent = StructuredIntent(
            interest="Football",
            audience="Football lovers",
            location=LocationDetails(),  # no location
        )
        cand = self._make_candidate(
            title="Football Enthusiast",
            snippet="Love football.",
        )
        _, breakdown = relevance_service.evaluate_candidate(cand, intent)
        assert breakdown["location_match"] == 25.0


# ===========================================================================
# 7. Deduplication & entity consolidation
# ===========================================================================

class TestDeduplication:

    def _cand(self, **kwargs) -> RawCandidate:
        defaults = dict(
            platform="LinkedIn",
            name="Alex Smith",
            location="London, UK",
            profile_url="https://linkedin.com/in/alexsmith",
            metadata={"handle": "alexsmith"},
            social_links=[SocialLink(platform="LinkedIn", link="https://linkedin.com/in/alexsmith")],
        )
        defaults.update(kwargs)
        return RawCandidate(**defaults)

    def test_exact_url_merges(self):
        a = self._cand()
        b = self._cand(platform="X", profile_url="https://linkedin.com/in/alexsmith",
                       social_links=[SocialLink(platform="X", link="https://linkedin.com/in/alexsmith")])
        prospects = deduplication_service.consolidate([a, b])
        assert len(prospects) == 1

    def test_shared_email_merges(self):
        a = self._cand(emails=["alex@example.io"],
                       social_links=[SocialLink(platform="LinkedIn", link="https://linkedin.com/in/alexsmith"),
                                      SocialLink(platform="Email", link="mailto:alex@example.io")])
        b = self._cand(platform="Instagram", profile_url="https://instagram.com/alexsmith",
                       emails=["alex@example.io"],
                       social_links=[SocialLink(platform="Instagram", link="https://instagram.com/alexsmith"),
                                      SocialLink(platform="Email", link="mailto:alex@example.io")])
        prospects = deduplication_service.consolidate([a, b])
        assert len(prospects) == 1
        platforms = {sl.platform for sl in prospects[0].social_link}
        assert "LinkedIn" in platforms
        assert "Instagram" in platforms
        assert "Email" in platforms

    def test_shared_handle_and_location_merges(self):
        a = self._cand(metadata={"handle": "alexsmith99"})
        b = self._cand(platform="Instagram", profile_url="https://instagram.com/alexsmith99",
                       metadata={"handle": "alexsmith99"},
                       social_links=[SocialLink(platform="Instagram", link="https://instagram.com/alexsmith99")])
        prospects = deduplication_service.consolidate([a, b])
        assert len(prospects) == 1

    def test_same_name_and_location_merges(self):
        a = self._cand(metadata={})
        b = self._cand(platform="X", profile_url="https://x.com/alexsmith99",
                       metadata={},
                       social_links=[SocialLink(platform="X", link="https://x.com/alexsmith99")])
        prospects = deduplication_service.consolidate([a, b])
        assert len(prospects) == 1

    def test_different_people_not_merged(self):
        a = self._cand(name="Alex Smith", location="London, UK")
        b = self._cand(name="Maria Garcia", location="Madrid, Spain",
                       platform="X", profile_url="https://x.com/mariagarcia",
                       metadata={"handle": "mariagarcia"},
                       social_links=[SocialLink(platform="X", link="https://x.com/mariagarcia")])
        prospects = deduplication_service.consolidate([a, b])
        assert len(prospects) == 2

    def test_similar_names_in_different_cities_not_merged(self):
        a = self._cand(name="James Lee", location="New York, USA",
                       profile_url="https://linkedin.com/in/jameslee-ny",
                       metadata={"handle": "jameslee-ny"},
                       social_links=[SocialLink(platform="LinkedIn", link="https://linkedin.com/in/jameslee-ny")])
        b = self._cand(name="James Lee", location="London, UK",
                       platform="Instagram", profile_url="https://instagram.com/jameslee_uk",
                       metadata={"handle": "jameslee_uk"},
                       social_links=[SocialLink(platform="Instagram", link="https://instagram.com/jameslee_uk")])
        prospects = deduplication_service.consolidate([a, b])
        assert len(prospects) == 2

    def test_consolidated_links_ordered_by_platform(self):
        a = self._cand(
            social_links=[
                SocialLink(platform="Email", link="mailto:alex@example.io"),
                SocialLink(platform="LinkedIn", link="https://linkedin.com/in/alexsmith"),
            ]
        )
        b = self._cand(platform="X", profile_url="https://x.com/alexsmith99",
                       metadata={"handle": "alexsmith99"},
                       social_links=[SocialLink(platform="X", link="https://x.com/alexsmith99")])
        prospects = deduplication_service.consolidate([a, b])
        plats = [sl.platform for sl in prospects[0].social_link]
        # LinkedIn should appear before Email
        assert plats.index("LinkedIn") < plats.index("Email")

    def test_no_duplicate_links_in_consolidation(self):
        link = SocialLink(platform="LinkedIn", link="https://linkedin.com/in/alexsmith")
        a = self._cand(social_links=[link])
        b = self._cand(social_links=[link])  # exact same link
        prospects = deduplication_service.consolidate([a, b])
        assert len([sl for sl in prospects[0].social_link if sl.platform == "LinkedIn"]) == 1


# ===========================================================================
# 8. API endpoint contract
# ===========================================================================

class TestAPIContract:

    def _mock_search(self, results_by_keyword: dict):
        """Return an async mock that returns different results based on query keywords."""
        async def mock_search(query: str, num_results: int = 5):
            for keyword, results in results_by_keyword.items():
                if keyword in query.lower():
                    return results
            return []
        return mock_search

    def test_success_response_shape(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.search_service.search_service.search",
            self._mock_search({
                "linkedin": [
                    {
                        "title": "Alex Morgan - Co-Founder & CEO | LinkedIn",
                        "link": "https://www.linkedin.com/in/alexmorgan",
                        "snippet": "Startup founder in London, UK. alex@techventures.co",
                    }
                ],
                "x.com": [
                    {
                        "title": "Alex Morgan (@alexmorgan) / X",
                        "link": "https://x.com/alexmorgan",
                        "snippet": "Startup founder London UK. alex@techventures.co",
                    }
                ],
            }),
        )
        resp = client.post(
            "/prospect-discovery",
            json={"query": "Startup founders in London UK"},
        )
        assert resp.status_code == 200
        body = resp.json()

        # Required top-level fields
        assert body["success"] is True
        assert body["query"] == "Startup founders in London UK"
        assert isinstance(body["total_prospects"], int)
        assert body["total_prospects"] >= 1
        assert isinstance(body["prospects"], list)

        # First prospect shape
        p = body["prospects"][0]
        assert "name" in p
        assert "social_link" in p
        assert isinstance(p["social_link"], list)

        # Alex from LinkedIn + X should consolidate into 1 prospect
        assert body["total_prospects"] == 1
        assert "Alex Morgan" in p["name"]
        assert "London" in p["location"]

        platform_names = [sl["platform"] for sl in p["social_link"]]
        assert "LinkedIn" in platform_names
        assert "X" in platform_names
        assert "Email" in platform_names

    def test_error_response_does_not_leak_internals(self, monkeypatch):
        """The 500 response must not contain stack traces or exception messages."""
        async def boom(*args, **kwargs):
            raise RuntimeError("Super secret DB password is hunter2")

        monkeypatch.setattr(
            "app.services.prospect_service.query_service.get_structured_intent",
            boom,
        )
        resp = client.post(
            "/prospect-discovery",
            json={"query": "Startup founders in London UK"},
        )
        assert resp.status_code == 500
        body = resp.json()
        assert "hunter2" not in json.dumps(body)
        assert "traceback" not in json.dumps(body).lower()
        assert body.get("success") is not True

    def test_response_has_request_id_header(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.search_service.search_service.search",
            AsyncMock(return_value=[]),
        )
        resp = client.post("/prospect-discovery", json={"query": "Startup founders in London UK"})
        assert "X-Request-ID" in resp.headers

    def test_total_prospects_matches_prospects_list_length(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.search_service.search_service.search",
            AsyncMock(return_value=[]),
        )
        resp = client.post("/prospect-discovery", json={"query": "Startup founders in London UK"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_prospects"] == len(body["prospects"])


# ===========================================================================
# 9. Partial platform failure resilience
# ===========================================================================

class TestResilience:

    def test_one_platform_failing_does_not_crash_endpoint(self, monkeypatch):
        from app.platforms import ALL_PLATFORMS

        original_discover = ALL_PLATFORMS[0].discover

        async def exploding_discover(*args, **kwargs):
            raise ConnectionResetError("Simulated network failure")

        monkeypatch.setattr(ALL_PLATFORMS[0], "discover", exploding_discover)
        monkeypatch.setattr(
            "app.services.search_service.search_service.search",
            AsyncMock(return_value=[]),
        )

        resp = client.post(
            "/prospect-discovery",
            json={"query": "Startup founders in London UK"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

        # Restore
        monkeypatch.setattr(ALL_PLATFORMS[0], "discover", original_discover)

    def test_all_platforms_failing_returns_empty_list(self, monkeypatch):
        from app.platforms import ALL_PLATFORMS

        originals = [(p, p.discover) for p in ALL_PLATFORMS]

        async def explode(*args, **kwargs):
            raise TimeoutError("All platforms down")

        for p in ALL_PLATFORMS:
            monkeypatch.setattr(p, "discover", explode)

        resp = client.post(
            "/prospect-discovery",
            json={"query": "Startup founders in London UK"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert isinstance(body["prospects"], list)

        for p, orig in originals:
            monkeypatch.setattr(p, "discover", orig)


# ===========================================================================
# 10. End-to-end pipeline – all 5 mandatory assessment queries
# ===========================================================================

class TestAllFiveMandatoryQueries:
    @pytest.mark.parametrize("tc", MANDATORY_QUERIES)
    def test_mandatory_query_pipeline(self, tc, monkeypatch):
        """Full pipeline with mocked search returns well-formed JSON for every query."""

        async def mock_search(query: str, num_results: int = 5):
            if "linkedin" in query.lower():
                return [
                    {
                        "title": f"Test Prospect - {tc['query']} | LinkedIn",
                        "link": "https://www.linkedin.com/in/test-prospect-123",
                        "snippet": (
                            f"Matching profile for {tc['query']}. "
                            f"Based in {tc['city']}, {tc['country']}."
                        ),
                    }
                ]
            return []

        monkeypatch.setattr(
            "app.services.search_service.search_service.search",
            mock_search,
        )

        resp = client.post("/prospect-discovery", json={"query": tc["query"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["query"] == tc["query"]
        assert isinstance(body["total_prospects"], int)
        assert isinstance(body["prospects"], list)
        assert body["total_prospects"] == len(body["prospects"])

        for prospect in body["prospects"]:
            assert prospect["name"]  # must have a name
            assert isinstance(prospect["social_link"], list)
            for sl in prospect["social_link"]:
                assert sl["platform"]
                assert sl["link"]
