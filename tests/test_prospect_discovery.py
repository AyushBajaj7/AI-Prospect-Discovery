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

from app.main import RateLimitMiddleware, app
from app.schemas import LocationDetails, RawCandidate, SearchStatus, SocialLink, StructuredIntent
from app.services.deduplication_service import deduplication_service
from app.services.llm_service import KNOWN_LOCATIONS, llm_service
from app.services.query_service import query_service
from app.services.relevance_service import relevance_service
from app.services.search_service import SearchOutcome, SearchService
from app.utils.normalizers import clean_name, classify_candidate_entity, extract_handle
from app.utils.validators import (
    clean_url,
    format_tel_uri,
    is_valid_email,
    is_valid_phone,
    is_valid_profile_url,
    normalize_whatsapp_url,
    sanitize_search_term,
)

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_rate_limiter_and_query_cache_per_test():
    """Keep per-IP sliding window and intent cache deterministic across tests."""
    RateLimitMiddleware.reset()
    query_service.clear_cache()
    yield
    RateLimitMiddleware.reset()
    query_service.clear_cache()


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
        assert resp.status_code in (400, 422)

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
        ("WhatsApp", "https://wa.me/12025551234"),
    ])
    def test_valid_profile_urls(self, platform, url):
        assert is_valid_profile_url(platform, url), f"Expected valid: {platform} {url}"

    # --- Invalid URLs (posts / search / feeds / malformed WhatsApp) ---
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
        ("WhatsApp", "https://wa.me/+12025551234"),  # '+' invalid without normalization
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
        assert clean_url(url) == "https://www.linkedin.com/in/johndoe"


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


# ===========================================================================
# 11. Security hardening, anti-SSRF, false-positive prevention & edge cases
# ===========================================================================

class TestSecurityAndEdgeCases:

    def test_cors_blocks_untrusted_origin_and_disallows_credentials(self):
        resp = client.options(
            "/prospect-discovery",
            headers={
                "Origin": "https://evil-attacker.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != "*"
        assert resp.headers.get("access-control-allow-origin") != "https://evil-attacker.example"
        assert resp.headers.get("access-control-allow-credentials") != "true"

    def test_cors_allows_configured_origin(self):
        resp = client.get("/health", headers={"Origin": "http://localhost:4000"})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:4000"
        assert resp.headers.get("access-control-allow-credentials") != "true"

    def test_oversized_body_rejected_with_request_id(self):
        huge_query = "a" * (70 * 1024)
        resp = client.post(
            "/prospect-discovery",
            content=json.dumps({"query": huge_query}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        assert "X-Request-ID" in resp.headers
        assert resp.json()["success"] is False

    def test_malformed_content_length_returns_400_not_500(self):
        resp = client.post(
            "/prospect-discovery",
            content='{"query": "Startup founders in London UK"}',
            headers={
                "Content-Type": "application/json",
                "Content-Length": "not-a-number",
            },
        )
        assert resp.status_code == 400
        assert "X-Request-ID" in resp.headers
        assert resp.json()["success"] is False

    @pytest.mark.parametrize("bad_url", [
        "http://localhost:4000/admin",
        "http://127.0.0.1/secret",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1/internal",
        "https://www.google.co.uk/search?q=test",
        "javascript:alert(1)",
    ])
    def test_ssrf_and_unsafe_urls_rejected(self, bad_url):
        assert not is_valid_profile_url("Google", bad_url)

    @pytest.mark.parametrize("url,expected_clean", [
        ("https://www.instagram.com/cristiano/?hl=en&igsh=abc123", "https://www.instagram.com/cristiano"),
        ("https://www.linkedin.com/in/alex-morgan?trk=public_profile", "https://www.linkedin.com/in/alex-morgan"),
        ("https://www.youtube.com/@mkbhd?si=sharetoken123", "https://www.youtube.com/@mkbhd"),
    ])
    def test_clean_url_strips_platform_tracking_and_validates(self, url, expected_clean):
        cleaned = clean_url(url)
        assert cleaned == expected_clean

    def test_facebook_people_and_profile_php_urls_valid(self):
        assert is_valid_profile_url("Facebook", "https://www.facebook.com/people/Alex-Morgan/1000837462738")
        assert is_valid_profile_url("Facebook", "https://www.facebook.com/profile.php?id=1000837462738")
        assert not is_valid_profile_url("Facebook", "https://www.facebook.com/login")
        assert not is_valid_profile_url("Instagram", "https://www.instagram.com/explore")

    def test_multipart_tld_email_extracted_and_year_range_not_phone(self):
        from app.platforms.contacts import extract_all_contacts

        snippet = "Co-Founder 2020-2025 | Joined 2024-01-15 | 1500000 views | email alex@company.co.uk"
        contacts = extract_all_contacts(snippet)
        assert "alex@company.co.uk" in contacts["emails"]
        assert contacts["phones"] == []
        assert not is_valid_phone("2020-2025")
        assert not is_valid_phone("00000000")

    @pytest.mark.parametrize("boilerplate", [
        "Log in or sign up to view",
        "Home",
        "Instagram",
        "404 Not Found",
        "Just a moment",
        "2026 New Jersey Travel Guide: must-see attractions, popular ...",
        "Restaurant, Bars & Cafes News and Updates from Dubai, UAE and ...",
    ])
    def test_boilerplate_names_rejected(self, boilerplate):
        assert clean_name(boilerplate, "Instagram") == ""

    def test_dynamic_unknown_city_heuristic_understanding(self):
        intent = asyncio.run(
            query_service.get_structured_intent("Yoga instructors in Austin, Texas, USA")
        )
        assert intent.location.city == "Austin"
        assert intent.location.state == "Texas"
        assert intent.location.country == "USA"
        assert "Yoga" in (intent.interest or "")

    def test_ukraine_does_not_match_uk_location_but_united_kingdom_does(self):
        uk_intent = StructuredIntent(
            interest="Startups",
            audience="Startup founders",
            location=LocationDetails(city="London", country="UK"),
        )
        ukraine_cand = RawCandidate(
            platform="LinkedIn",
            name="Ivan Kovalenko",
            profile_url="https://linkedin.com/in/ivankovalenko",
            title="Founder | Tech in Kyiv, Ukraine",
            snippet="Building software products in Ukraine.",
        )
        _, ukraine_bd = relevance_service.evaluate_candidate(ukraine_cand, uk_intent)
        assert ukraine_bd["location_match"] == 0.0

        england_cand = RawCandidate(
            platform="LinkedIn",
            name="Emma Clarke",
            profile_url="https://linkedin.com/in/emmaclarke",
            title="Founder | Tech in Greater London, United Kingdom",
            snippet="Startup founder based in United Kingdom.",
        )
        _, england_bd = relevance_service.evaluate_candidate(england_cand, uk_intent)
        assert england_bd["location_match"] >= 20.0

    def test_india_vs_indiana_not_merged_as_same_location(self):
        a = RawCandidate(
            platform="LinkedIn",
            name="Rahul Sharma",
            location="India",
            profile_url="https://linkedin.com/in/rahul-in",
            social_links=[SocialLink(platform="LinkedIn", link="https://linkedin.com/in/rahul-in")],
        )
        b = RawCandidate(
            platform="X",
            name="Rahul Sharma",
            location="Indiana, USA",
            profile_url="https://x.com/rahul_indiana",
            social_links=[SocialLink(platform="X", link="https://x.com/rahul_indiana")],
        )
        prospects = deduplication_service.consolidate([a, b])
        assert len(prospects) == 2

    def test_transitive_deduplication_merges_three_profiles_via_bridge(self):
        a = RawCandidate(
            platform="LinkedIn",
            name="Sam Patel",
            location="London, UK",
            profile_url="https://linkedin.com/in/sampatel",
            emails=["sam@patel.io"],
            social_links=[SocialLink(platform="LinkedIn", link="https://linkedin.com/in/sampatel")],
        )
        b = RawCandidate(
            platform="X",
            name="Sam Patel",
            location="London, UK",
            profile_url="https://x.com/sampatel_uk",
            metadata={"handle": "sampatel_uk"},
            social_links=[SocialLink(platform="X", link="https://x.com/sampatel_uk")],
        )
        # C shares handle with B and email with A -> bridges A and B together
        c = RawCandidate(
            platform="Instagram",
            name="S. Patel",
            location="London, UK",
            profile_url="https://instagram.com/sampatel_uk",
            emails=["sam@patel.io"],
            metadata={"handle": "sampatel_uk"},
            social_links=[SocialLink(platform="Instagram", link="https://instagram.com/sampatel_uk")],
        )
        prospects = deduplication_service.consolidate([a, b, c])
        assert len(prospects) == 1
        assert len(prospects[0].social_link) == 3

    def test_adapter_does_not_fabricate_location_when_absent_from_text(self):
        from app.platforms.linkedin import LinkedInAdapter

        adapter = LinkedInAdapter()
        intent = StructuredIntent(
            interest="Football",
            audience="Football lovers",
            location=LocationDetails(city="Washington DC", state="District of Columbia", country="USA"),
        )
        loc = adapter._infer_location("John Doe - Sports Analyst | LinkedIn", intent)
        assert loc is None

    def test_different_people_on_same_social_platform_never_merged_by_shared_role_email(self):
        c1 = RawCandidate(
            platform="LinkedIn",
            name="Rahim Nathwani",
            location="London, UK",
            profile_url="https://www.linkedin.com/in/rahimnathwani",
            emails=["cofounder@rephrase.ai"],
            social_links=[
                SocialLink(platform="LinkedIn", link="https://www.linkedin.com/in/rahimnathwani"),
                SocialLink(platform="Email", link="mailto:cofounder@rephrase.ai"),
            ],
        )
        c2 = RawCandidate(
            platform="LinkedIn",
            name="Yashvardhan Rana",
            location="London, UK",
            profile_url="https://in.linkedin.com/in/yashvardhanrana",
            emails=["cofounder@rephrase.ai"],
            social_links=[
                SocialLink(platform="LinkedIn", link="https://in.linkedin.com/in/yashvardhanrana"),
                SocialLink(platform="Email", link="mailto:cofounder@rephrase.ai"),
            ],
        )
        prospects = deduplication_service.consolidate([c1, c2])
        assert len(prospects) == 2

    def test_google_adapter_rejects_news_articles_and_event_directories(self):
        from app.platforms.google import GoogleAdapter

        g = GoogleAdapter()
        assert not g.validate_url("https://news.northeastern.edu/2026/06/05/startup-pitch-competition")
        assert not g.validate_url("https://www.meetup.com/london-entrepreneur-startup-network")
        assert not g.validate_url("https://www.london.edu/insights/news/spotlight")
        assert not g.validate_url("https://www.trip.com/moments/destination-new-jersey-1427")
        assert not g.validate_url("https://www.caterermiddleeast.com/outlets")
        assert g.validate_url("https://about.me/alexmorgan")


# ===========================================================================
# 12. Security audit remediation, fallback chain, unknown cities & resilience
# ===========================================================================

class TestSecurityAndAccuracyAuditFixes:

    def test_whatsapp_normalization_strips_plus_and_formatting(self):
        from app.platforms.contacts import extract_whatsapp_links

        assert normalize_whatsapp_url("https://wa.me/+442079460991") == "https://wa.me/442079460991"
        assert (
            normalize_whatsapp_url("https://api.whatsapp.com/send?phone=+44 20 7946 0991&text=hi")
            == "https://wa.me/442079460991"
        )
        assert clean_url("https://wa.me/+12025551234") == "https://wa.me/12025551234"
        links = extract_whatsapp_links("Reach us at https://wa.me/+442079460991 today")
        assert links == ["https://wa.me/442079460991"]

    def test_search_operator_and_prompt_injection_sanitized(self):
        from app.platforms.linkedin import LinkedInAdapter

        malicious = 'Founders" OR site:internal.corp -site:linkedin.com inurl:admin'
        cleaned = sanitize_search_term(malicious)
        assert "site:" not in cleaned.lower()
        assert "inurl:" not in cleaned.lower()
        assert '"' not in cleaned
        assert "-site" not in cleaned.lower()

        adapter = LinkedInAdapter()
        injected_intent = StructuredIntent(
            interest='Tech" OR site:secret.internal',
            audience='CEO" OR site:internal.corp -site:linkedin.com',
            location=LocationDetails(city='London" OR inurl:login'),
        )
        q = adapter.build_query(injected_intent)
        assert "site:internal.corp" not in q
        assert "site:secret.internal" not in q
        assert "inurl:login" not in q
        assert q.startswith("site:linkedin.com/in/")

    @pytest.mark.parametrize("ipv6_url", [
        "http://[::1]:4000/admin",
        "http://[fe80::1]/secret",
        "http://[fc00::1]/internal",
        "http://[::ffff:127.0.0.1]/admin",
    ])
    def test_ipv6_private_and_loopback_urls_rejected(self, ipv6_url):
        assert not is_valid_profile_url("Google", ipv6_url)

    def test_redos_adversarial_long_inputs_complete_quickly(self):
        import time
        from app.platforms.contacts import extract_all_contacts

        adversarial = ("a." * 3000) + "@@@@" + ("1-2-" * 3000) + " https://wa.me/+" + ("9" * 5000)
        start = time.perf_counter()
        contacts = extract_all_contacts(adversarial)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.2, f"Regex scan took too long: {elapsed:.3f}s"
        assert contacts["emails"] == []

    def test_endpoint_rate_limiting_enforced(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS", 3)
        monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 60)
        monkeypatch.setattr(
            "app.services.search_service.search_service.search_with_status",
            AsyncMock(return_value=type("R", (), {"hits": [], "status": SearchStatus.NO_RESULTS})()),
        )

        RateLimitMiddleware.reset()
        for idx in range(3):
            resp = client.post(
                "/prospect-discovery",
                json={"query": f"Startup founders in London UK #{idx}"},
            )
            assert resp.status_code == 200
            assert resp.headers.get("X-RateLimit-Limit") == "3"

        blocked = client.post(
            "/prospect-discovery",
            json={"query": "Startup founders in London UK #4"},
        )
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers
        assert "X-Request-ID" in blocked.headers
        assert blocked.json()["success"] is False

    def test_genuine_unknown_city_and_country_parsing(self):
        assert "austin" not in KNOWN_LOCATIONS
        assert "nairobi" not in KNOWN_LOCATIONS
        assert "kenya" not in [k.lower() for k in KNOWN_LOCATIONS]

        i1 = asyncio.run(llm_service.parse_query("Startup founders in Nairobi Kenya"))
        assert i1.location.city == "Nairobi"
        assert i1.location.country == "Kenya"

        i2 = asyncio.run(llm_service.parse_query("Photographers in Reykjavik Iceland"))
        assert i2.location.city == "Reykjavik"
        assert i2.location.country == "Iceland"

        i3 = asyncio.run(llm_service.parse_query("Yoga instructors in Austin, Texas, USA"))
        assert i3.location.city == "Austin"
        assert i3.location.state == "Texas"
        assert i3.location.country == "USA"

    def test_target_type_consistency_rule(self):
        owner_intent = llm_service._parse_heuristic("Restaurant owners in Dubai UAE")
        agency_owner_intent = llm_service._parse_heuristic("Digital marketing agency owners in New York USA")
        org_intent = llm_service._parse_heuristic("Digital marketing agencies in New York USA")
        assert owner_intent.target_type == "individual"
        assert agency_owner_intent.target_type == "individual"
        assert org_intent.target_type == "business"

    def test_llm_multi_provider_fallback_chain(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "LLM_PROVIDER", "auto")
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "fake-openai")
        monkeypatch.setattr(settings, "GEMINI_API_KEY", "fake-gemini")
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "fake-anthropic")

        expected_gemini = StructuredIntent(
            interest="Football",
            audience="Football lovers",
            location=LocationDetails(city="London", country="UK"),
        )

        monkeypatch.setattr(llm_service, "_call_openai", AsyncMock(side_effect=RuntimeError("OpenAI 503")))
        monkeypatch.setattr(llm_service, "_call_gemini", AsyncMock(return_value=expected_gemini))
        monkeypatch.setattr(llm_service, "_call_anthropic", AsyncMock(side_effect=RuntimeError("Unused")))

        res = asyncio.run(llm_service.parse_query("Football lovers in London UK"))
        assert res.interest == "Football"
        assert llm_service.stats["provider_used"] == "gemini"
        assert llm_service.stats["provider_attempts"] == ["openai", "gemini"]

        # When Gemini also fails, must fall back to Anthropic before heuristic
        expected_claude = StructuredIntent(
            interest="Cricket",
            audience="Cricket lovers",
            location=LocationDetails(city="Ahmedabad", country="India"),
        )
        monkeypatch.setattr(llm_service, "_call_gemini", AsyncMock(side_effect=TimeoutError("timeout")))
        monkeypatch.setattr(llm_service, "_call_anthropic", AsyncMock(return_value=expected_claude))
        res2 = asyncio.run(llm_service.parse_query("Cricket lovers in Ahmedabad India"))
        assert res2.interest == "Cricket"
        assert llm_service.stats["provider_used"] == "anthropic"
        assert llm_service.stats["provider_attempts"] == ["openai", "gemini", "anthropic"]

    def test_john_smith_and_john_doe_not_merged_on_generic_or_shared_email(self):
        c1 = RawCandidate(
            platform="LinkedIn",
            name="John Smith",
            location="London, UK",
            profile_url="https://linkedin.com/in/johnsmith",
            emails=["contact@acme.co.uk"],
            social_links=[SocialLink(platform="LinkedIn", link="https://linkedin.com/in/johnsmith")],
        )
        c2 = RawCandidate(
            platform="X",
            name="John Doe",
            location="London, UK",
            profile_url="https://x.com/johndoe",
            emails=["contact@acme.co.uk"],
            social_links=[SocialLink(platform="X", link="https://x.com/johndoe")],
        )
        # Even with non-generic email, mismatched surnames cannot merge distinct full human names
        c3 = RawCandidate(
            platform="LinkedIn",
            name="John Smith",
            location="London, UK",
            profile_url="https://linkedin.com/in/johnsmith2",
            emails=["shared@acme.co.uk"],
            social_links=[SocialLink(platform="LinkedIn", link="https://linkedin.com/in/johnsmith2")],
        )
        c4 = RawCandidate(
            platform="X",
            name="John Doe",
            location="London, UK",
            profile_url="https://x.com/johndoe2",
            emails=["shared@acme.co.uk"],
            social_links=[SocialLink(platform="X", link="https://x.com/johndoe2")],
        )
        assert len(deduplication_service.consolidate([c1, c2])) == 2
        assert len(deduplication_service.consolidate([c3, c4])) == 2

    def test_strict_relevance_threshold_drops_subthreshold_candidates(self, monkeypatch):
        # Single weak signal (< 35 pts) must not leak into output via fallback
        async def weak_search(query: str, num_results: int = 5):
            if "x.com" in query.lower():
                return [
                    {
                        "title": "Random Person (@randomp) / X",
                        "link": "https://x.com/randomp",
                        "snippet": "Generic personal feed with no topic or city.",
                    }
                ]
            return []

        monkeypatch.setattr("app.services.search_service.search_service.search", weak_search)
        resp = client.post(
            "/prospect-discovery",
            json={"query": "Cricket lovers in Ahmedabad India"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_prospects"] == 0
        assert body["prospects"] == []

    def test_search_status_distinguishes_errors_and_never_caches_failures(self, monkeypatch):
        from app.config import settings

        svc = SearchService()
        monkeypatch.setattr(settings, "SEARCH_PROVIDER", "serper")
        monkeypatch.setattr(settings, "SERPER_API_KEY", "test-key")

        async def rate_limited_serper(q, n):
            from app.services.search_service import SearchOutcome
            return SearchOutcome(hits=[], status=SearchStatus.RATE_LIMITED, provider="serper")

        async def timeout_ddg(q, n):
            from app.services.search_service import SearchOutcome
            return SearchOutcome(hits=[], status=SearchStatus.TIMEOUT, provider="ddg")

        monkeypatch.setattr(svc, "_search_serper", rate_limited_serper)
        monkeypatch.setattr(svc, "_search_ddg", timeout_ddg)

        outcome = asyncio.run(svc.search_with_status("test query", 5))
        assert outcome.status == SearchStatus.RATE_LIMITED
        assert outcome.hits == []
        assert len(svc._cache) == 0  # failed / empty search is never cached

    def test_provider_health_present_and_tracks_rate_limits_and_timeouts(self, monkeypatch):
        from app.platforms import ALL_PLATFORMS

        monkeypatch.setattr(
            "app.services.search_service.search_service.search_with_status",
            AsyncMock(return_value=SearchOutcome(hits=[], status=SearchStatus.RATE_LIMITED, provider="ddg")),
        )
        # Default response strictly matches assessment keys (no extra provider_health key)
        resp_default = client.post(
            "/prospect-discovery",
            json={"query": "Startup founders in London UK"},
        )
        assert resp_default.status_code == 200
        body_default = resp_default.json()
        assert set(body_default.keys()) == {"success", "query", "total_prospects", "prospects"}
        assert resp_default.headers.get("X-Platforms-Failed") == str(len(ALL_PLATFORMS))

        # Diagnostics mode exposes provider_health in response body
        resp_diag = client.post(
            "/prospect-discovery?include_diagnostics=true",
            json={"query": "Startup founders in London UK"},
        )
        assert resp_diag.status_code == 200
        body_diag = resp_diag.json()
        assert body_diag["provider_health"] is not None
        assert body_diag["provider_health"]["platforms_rate_limited"] == len(ALL_PLATFORMS)
        assert body_diag["provider_health"]["platforms_succeeded"] == 0


# ===========================================================================
# 13. Security & accuracy remediation tests (XFF, LRU, SDKs, entities, tel)
# ===========================================================================

class TestP0ThroughP3Remediation:

    def test_spoofed_x_forwarded_for_ignored_when_proxy_untrusted(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "TRUSTED_PROXIES", "")
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS", 2)
        monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 60)
        monkeypatch.setattr(
            "app.services.search_service.search_service.search_with_status",
            AsyncMock(return_value=SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="ddg")),
        )
        RateLimitMiddleware.reset()

        for fake_ip in ("198.51.100.1", "198.51.100.2"):
            r = client.post(
                "/prospect-discovery",
                json={"query": "Startup founders in London UK"},
                headers={"X-Forwarded-For": fake_ip},
            )
            assert r.status_code == 200

        # Third request with yet another spoofed XFF must still be rate-limited!
        blocked = client.post(
            "/prospect-discovery",
            json={"query": "Startup founders in London UK"},
            headers={"X-Forwarded-For": "198.51.100.99"},
        )
        assert blocked.status_code == 429

    def test_x_forwarded_for_respected_only_from_trusted_proxy(self, monkeypatch):
        import httpx
        from app.config import settings

        monkeypatch.setattr(settings, "TRUSTED_PROXIES", "10.0.0.0/8")
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS", 1)
        monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 60)
        monkeypatch.setattr(
            "app.services.search_service.search_service.search_with_status",
            AsyncMock(return_value=SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="ddg")),
        )
        RateLimitMiddleware.reset()

        async def _run():
            transport = httpx.ASGITransport(app=app, client=("10.0.0.1", 50000))
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
                r1 = await ac.post(
                    "/prospect-discovery",
                    json={"query": "Startup founders in London UK"},
                    headers={"X-Forwarded-For": "203.0.113.10, 10.0.0.5"},
                )
                assert r1.status_code == 200

                # Same client IP via trusted proxy hits 429
                r1_dup = await ac.post(
                    "/prospect-discovery",
                    json={"query": "Startup founders in London UK"},
                    headers={"X-Forwarded-For": "203.0.113.10, 10.0.0.5"},
                )
                assert r1_dup.status_code == 429

                # Different client IP behind trusted proxy gets its own quota
                r2 = await ac.post(
                    "/prospect-discovery",
                    json={"query": "Startup founders in London UK"},
                    headers={"X-Forwarded-For": "203.0.113.11"},
                )
                assert r2.status_code == 200

        asyncio.run(_run())

    def test_rate_limit_store_prunes_expired_and_bounds_max_clients(self, monkeypatch):
        import httpx
        from app.config import settings

        monkeypatch.setattr(settings, "TRUSTED_PROXIES", "10.0.0.0/8")
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS", 5)
        monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 60)
        monkeypatch.setattr(settings, "RATE_LIMIT_MAX_CLIENTS", 5)
        monkeypatch.setattr(
            "app.services.search_service.search_service.search_with_status",
            AsyncMock(return_value=SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="ddg")),
        )
        RateLimitMiddleware.reset()

        async def _run():
            transport = httpx.ASGITransport(app=app, client=("10.0.0.1", 50000))
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
                for i in range(12):
                    resp = await ac.post(
                        "/prospect-discovery",
                        json={"query": "Startup founders in London UK"},
                        headers={"X-Forwarded-For": f"203.0.113.{i + 1}"},
                    )
                    assert resp.status_code == 200

        asyncio.run(_run())
        # Must strictly cap at RATE_LIMIT_MAX_CLIENTS (never unbounded growth)
        assert len(RateLimitMiddleware._hits) == 5
        # Oldest 7 IPs were evicted; newest 5 remain
        assert "203.0.113.12" in RateLimitMiddleware._hits
        assert "203.0.113.1" not in RateLimitMiddleware._hits

        # Expired buckets must be pruned automatically on next check
        RateLimitMiddleware._hits["203.0.113.10"].clear()
        RateLimitMiddleware._prune_buckets(now=10**9, window=60, max_clients=5)
        assert len(RateLimitMiddleware._hits) == 0

    def test_sdk_minimum_requirements_and_structured_output_apis(self):
        from pathlib import Path
        import inspect
        import anthropic
        from google import genai
        from google.genai import types
        from openai import AsyncOpenAI

        req_text = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text()
        assert "fastapi>=0.115.6" in req_text
        assert "anthropic>=0.77.0" in req_text
        assert "google-genai>=1.60.0" in req_text
        assert "httpx>=0.28.1" in req_text
        assert "openai>=1.58.0" in req_text

        assert hasattr(anthropic, "transform_schema")
        transformed = anthropic.transform_schema(StructuredIntent)
        assert isinstance(transformed, dict)
        assert "output_config" in inspect.signature(anthropic.AsyncAnthropic(api_key="t").messages.create).parameters

        cfg = types.GenerateContentConfig(
            system_instruction="test",
            response_mime_type="application/json",
            response_schema=StructuredIntent,
        )
        assert cfg.response_schema is StructuredIntent

        assert hasattr(AsyncOpenAI(api_key="t").beta.chat.completions, "parse")

    def test_linkedin_regional_subdomains_canonicalized_and_deduplicated(self):
        u1 = clean_url("https://uk.linkedin.com/in/alex-morgan/")
        u2 = clean_url("https://ge.linkedin.com/in/alex-morgan?trk=public")
        u3 = clean_url("https://www.linkedin.com/in/alex-morgan")
        assert u1 == "https://www.linkedin.com/in/alex-morgan"
        assert u2 == "https://www.linkedin.com/in/alex-morgan"
        assert u3 == "https://www.linkedin.com/in/alex-morgan"

        c1 = RawCandidate(
            platform="LinkedIn",
            name="Alex Morgan",
            location="London, UK",
            profile_url=u1,
            social_links=[SocialLink(platform="LinkedIn", link=u1)],
        )
        c2 = RawCandidate(
            platform="LinkedIn",
            name="Alex Morgan",
            location="London, UK",
            profile_url=u2,
            social_links=[SocialLink(platform="LinkedIn", link=u2)],
        )
        prospects = deduplication_service.consolidate([c1, c2])
        assert len(prospects) == 1
        assert len(prospects[0].social_link) == 1
        assert prospects[0].social_link[0].link == "https://www.linkedin.com/in/alex-morgan"

    def test_tel_uri_rfc3966_formatted_without_spaces(self):
        from app.services.profile_extractor import profile_extractor

        assert format_tel_uri("+44 20 7946 0991") == "tel:+442079460991"
        assert format_tel_uri("+1 (202) 555-0199") == "tel:+12025550199"

        cand = RawCandidate(
            platform="Google",
            name="Alex Morgan",
            profile_url="https://about.me/alexmorgan",
            title="Alex Morgan",
            snippet="Call me at +44 20 7946 0991 in London",
        )
        enriched = profile_extractor.enrich_candidate(
            cand,
            StructuredIntent(interest="Startups", audience="Startup founders"),
        )
        tel_links = [sl.link for sl in enriched.social_links if sl.platform == "Phone"]
        assert tel_links == ["tel:+442079460991"]

    def test_verification_status_honestly_labeled(self, monkeypatch):
        from app.platforms.google import GoogleAdapter
        from app.platforms.linkedin import LinkedInAdapter

        monkeypatch.setattr(
            "app.services.search_service.search_service.search",
            AsyncMock(return_value=[{
                "title": "Alex Morgan - Startup Founder | LinkedIn",
                "link": "https://uk.linkedin.com/in/alexmorgan",
                "snippet": "Founder in London, UK",
            }]),
        )
        li_hits = asyncio.run(
            LinkedInAdapter().discover(StructuredIntent(interest="Startups", audience="Startup founders"))
        )
        assert len(li_hits) == 1
        assert li_hits[0].verification_status == "url_format_validated"

        monkeypatch.setattr(
            "app.services.search_service.search_service.search",
            AsyncMock(return_value=[{
                "title": "Alex Morgan",
                "link": "https://about.me/alexmorgan",
                "snippet": "Founder in London, UK",
            }]),
        )
        g_hits = asyncio.run(
            GoogleAdapter().discover(StructuredIntent(interest="Startups", audience="Startup founders"))
        )
        assert len(g_hits) == 1
        assert g_hits[0].verification_status == "search_indexed_web"

    def test_target_type_and_parody_filtering_removes_orgs_and_gimmicks(self):
        intent_ind = StructuredIntent(
            interest="Startups",
            audience="Startup founders",
            location=LocationDetails(city="London", country="UK"),
            target_type="individual",
        )
        person = RawCandidate(
            platform="LinkedIn",
            name="Rahim Nathwani",
            location="London, UK",
            profile_url="https://www.linkedin.com/in/rahimnathwani",
            title="Rahim Nathwani - Co-Founder | LinkedIn",
            snippet="Startup founder in London, UK.",
            entity_type=classify_candidate_entity(
                "Rahim Nathwani", "LinkedIn", "https://www.linkedin.com/in/rahimnathwani"
            ),
        )
        org = RawCandidate(
            platform="Google",
            name="London Founder's Network",
            location="London, UK",
            profile_url="https://www.londonfoundersnetwork.com",
            title="London Founder's Network",
            snippet="Community for startup founders in London, UK.",
            entity_type=classify_candidate_entity(
                "London Founder's Network", "Google", "https://www.londonfoundersnetwork.com"
            ),
        )
        parody = RawCandidate(
            platform="X",
            name="Parody Kartik Sharma",
            location="London, UK",
            profile_url="https://x.com/parodykartik",
            title="Parody Kartik Sharma (@parodykartik) / X",
            snippet="Startup founder parody memes in London UK.",
            entity_type=classify_candidate_entity(
                "Parody Kartik Sharma", "X", "https://x.com/parodykartik"
            ),
        )
        gimmick = RawCandidate(
            platform="X",
            name="king of cricketer",
            location="London, UK",
            profile_url="https://x.com/kingcricketer",
            title="king of cricketer / X",
            snippet="Startup founders London UK.",
            entity_type=classify_candidate_entity(
                "king of cricketer", "X", "https://x.com/kingcricketer"
            ),
        )
        kept = relevance_service.filter_relevant_candidates([person, org, parody, gimmick], intent_ind)
        assert [c.name for c in kept] == ["Rahim Nathwani"]

    def test_query_service_caches_exact_provider_without_global_bleed(self, monkeypatch):
        query_service.clear_cache()

        async def mock_parse(q: str):
            llm_service.stats["provider_used"] = "openai" if "london" in q.lower() else "gemini"
            return (
                StructuredIntent(interest="Startups", audience="Founders"),
                llm_service.stats["provider_used"],
            )

        monkeypatch.setattr(llm_service, "parse_query_with_provider", mock_parse)

        _, p1 = asyncio.run(query_service.get_structured_intent_with_provider("Founders in London"))
        assert p1 == "openai"

        # Another unrelated query mutates llm_service.stats['provider_used'] to 'gemini'
        _, p2 = asyncio.run(query_service.get_structured_intent_with_provider("Founders in Berlin"))
        assert p2 == "gemini"

        # Re-querying cached first query must still report 'openai', NOT 'gemini'
        _, p1_cached = asyncio.run(query_service.get_structured_intent_with_provider("Founders in London"))
        assert p1_cached == "openai"

    def test_search_status_preserves_rate_limit_over_trailing_no_results(self, monkeypatch):
        from app.config import settings

        svc = SearchService()
        monkeypatch.setattr(settings, "SEARCH_PROVIDER", "auto")
        monkeypatch.setattr(settings, "SERPER_API_KEY", "s-key")
        monkeypatch.setattr(settings, "BRAVE_API_KEY", "b-key")

        monkeypatch.setattr(
            svc, "_search_serper",
            AsyncMock(return_value=SearchOutcome(hits=[], status=SearchStatus.RATE_LIMITED, provider="serper")),
        )
        monkeypatch.setattr(
            svc, "_search_brave",
            AsyncMock(return_value=SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="brave")),
        )
        monkeypatch.setattr(
            svc, "_search_ddg",
            AsyncMock(return_value=SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="ddg")),
        )

        outcome = asyncio.run(svc.search_with_status("test fallback aggregation", 5))
        assert outcome.status == SearchStatus.RATE_LIMITED


class TestAuditRemediationAndObservability:
    """Tests covering LLM reasoning/thinking controls, funnel telemetry,
    location grounding, entity classification, RFC 3966 tel URIs, and CORS."""

    def test_llm_providers_configure_reasoning_and_thinking_controls(self, monkeypatch):
        from types import SimpleNamespace
        from app.config import settings
        from app.services import llm_service as llm_mod

        monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "ant-test")
        monkeypatch.setattr(settings, "GEMINI_API_KEY", "gem-test")

        valid_intent_json = json.dumps({
            "interest": "Startups",
            "audience": "Startup founders",
            "location": {"city": "London", "state": None, "country": "UK"},
            "target_type": "individual",
            "keywords": ["startup", "founder"],
        })

        # 1. OpenAI: reasoning_effort="none", temperature=0.0, max_completion_tokens=400
        captured_openai = {}

        class FakeOpenAIClient:
            def __init__(self, **kwargs):
                completions = SimpleNamespace(parse=self._parse)
                self.beta = SimpleNamespace(chat=SimpleNamespace(completions=completions))

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def _parse(self, **kwargs):
                captured_openai.update(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                parsed=StructuredIntent.model_validate_json(valid_intent_json),
                                content=valid_intent_json,
                            )
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45),
                )

        import openai
        monkeypatch.setattr(openai, "AsyncOpenAI", FakeOpenAIClient)
        if hasattr(llm_mod, "AsyncOpenAI"):
            monkeypatch.setattr(llm_mod, "AsyncOpenAI", FakeOpenAIClient)

        intent_oai = asyncio.run(llm_service._call_openai("Startup founders in London UK", 5.0))
        assert intent_oai.audience == "Startup founders"
        assert captured_openai["reasoning_effort"] == "none"
        assert captured_openai["temperature"] == 0.0
        assert captured_openai["max_completion_tokens"] == 400

        # 2. Anthropic: thinking={"type": "disabled"} + output_config["effort"] == "low"
        captured_ant = {}

        class FakeAnthropicClient:
            def __init__(self, **kwargs):
                self.messages = SimpleNamespace(create=self._create)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def _create(self, **kwargs):
                captured_ant.update(kwargs)
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text=valid_intent_json)],
                    usage=SimpleNamespace(input_tokens=110, output_tokens=48),
                )

        import anthropic
        monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAnthropicClient)
        if hasattr(llm_mod, "AsyncAnthropic"):
            monkeypatch.setattr(llm_mod, "AsyncAnthropic", FakeAnthropicClient)

        intent_ant = asyncio.run(llm_service._call_anthropic("Startup founders in London UK", 5.0))
        assert intent_ant.location.city == "London"
        assert captured_ant["thinking"] == {"type": "disabled"}
        assert captured_ant["output_config"]["effort"] == "low"

        # 3. Gemini: ThinkingConfig(thinking_level="MINIMAL"), max_output_tokens=400, thoughts_token_count tracked
        captured_gem = {}

        class FakeGeminiAio:
            def __init__(self):
                self.models = SimpleNamespace(generate_content=self._generate)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def _generate(self, **kwargs):
                captured_gem.update(kwargs)
                return SimpleNamespace(
                    text=valid_intent_json,
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=100,
                        candidates_token_count=42,
                        thoughts_token_count=7,
                    ),
                )

        class FakeGeminiClient:
            def __init__(self, **kwargs):
                self.aio = FakeGeminiAio()

        from google import genai
        monkeypatch.setattr(genai, "Client", FakeGeminiClient)

        intent_gem = asyncio.run(llm_service._call_gemini("Startup founders in London UK", 5.0))
        assert intent_gem.target_type == "individual"
        cfg = captured_gem["config"]
        assert cfg.max_output_tokens == 400
        assert str(cfg.thinking_config.thinking_level).upper().endswith("MINIMAL")
        assert llm_service.stats["thinking_tokens"] >= 7

    def test_platform_pipeline_metrics_exposed_in_diagnostics(self, monkeypatch):
        from app.services.prospect_service import prospect_service
        from app.platforms.base import BasePlatformAdapter
        from app.services.search_service import search_service

        RateLimitMiddleware.reset()
        query_service.clear_cache()

        async def mock_search_with_status(query_str: str, num_results: int = 10):
            if "site:linkedin.com" in query_str:
                return SearchOutcome(
                    hits=[
                        # invalid URL for LinkedIn (/company/ instead of /in/)
                        {"title": "Acme Corp", "link": "https://www.linkedin.com/company/acme", "snippet": "Startup"},
                        # parody handle/name
                        {"title": "Parody Founder (@parody) | LinkedIn", "link": "https://www.linkedin.com/in/parodyfounder", "snippet": "Startup founders in London UK"},
                        # valid high-relevance founder
                        {"title": "Jane Smith - Startup Founder | LinkedIn", "link": "https://www.linkedin.com/in/janesmith", "snippet": "Startup founder in London, UK"},
                        # low-relevance off-topic hit
                        {"title": "Bob Jones - Chef | LinkedIn", "link": "https://www.linkedin.com/in/bobjones", "snippet": "Pastry chef in Tokyo"},
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="serper",
                )
            return SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="serper")

        monkeypatch.setattr(search_service, "search_with_status", mock_search_with_status)
        monkeypatch.setattr(
            llm_service,
            "parse_query_with_provider",
            AsyncMock(return_value=(
                StructuredIntent(
                    interest="Startups",
                    audience="Startup founders",
                    location=LocationDetails(city="London", country="UK"),
                    target_type="individual",
                    keywords=["startup", "founder"],
                ),
                "heuristic",
            )),
        )

        # Ensure LinkedIn adapter passes BasePlatformAdapter's default URL validation cleanly
        resp = client.post(
            "/prospect-discovery?include_diagnostics=true",
            json={"query": "Startup founders in London UK"},
        )
        assert resp.status_code == 200
        body = resp.json()
        pm = body["provider_health"]["platform_metrics"]
        assert "LinkedIn" in pm
        li = pm["LinkedIn"]
        assert li["search_status"] == SearchStatus.SUCCESS.value
        assert li["search_hits"] == 4
        assert li["invalid_urls"] >= 1
        assert li["parody_or_gimmick"] >= 1
        assert li["below_relevance_threshold"] >= 1
        assert li["retained_candidates"] == 1
        assert body["total_prospects"] == 1

    def test_entity_classification_preserves_lowercase_names_and_hospitality_orgs(self):
        assert classify_candidate_entity("john doe", "X", "https://x.com/johndoe") != "parody_or_gimmick"
        for org_name in ("Dubai Bistro", "Grand Hotel Dubai", "Cafe Roma"):
            assert classify_candidate_entity(org_name, "Google", "https://example.com") == "business"

        intent_biz = StructuredIntent(
            interest="Restaurants",
            audience="Restaurant owners",
            location=LocationDetails(city="Dubai", country="UAE"),
            target_type="business",
            keywords=["restaurant", "bistro"],
        )
        candidates = [
            RawCandidate(
                platform="Google",
                name="Dubai Bistro",
                location="Dubai, UAE",
                profile_url="https://dubaibistro.ae",
                title="Dubai Bistro - Restaurant in Dubai",
                snippet="Fine dining restaurant in Dubai, UAE.",
                entity_type="business",
            )
        ]
        kept = relevance_service.filter_relevant_candidates(candidates, intent_biz)
        assert [c.name for c in kept] == ["Dubai Bistro"]

    def test_infer_location_never_fabricates_unconfirmed_country(self):
        from app.platforms.linkedin import LinkedInAdapter

        adapter = LinkedInAdapter()
        intent = StructuredIntent(
            interest="Startups",
            audience="Startup Coach",
            location=LocationDetails(city="Ahmedabad", country="India"),
            target_type="individual",
        )
        # Snippet mentions Ahmedabad only, NOT India -> must stay "Ahmedabad"
        inferred = adapter._infer_location("Startup Coach in Ahmedabad building SaaS", intent)
        assert inferred == "Ahmedabad"

    def test_format_tel_uri_rfc3966_local_and_global(self):
        assert format_tel_uri("202-555-0199", phone_context="acme.com") == "tel:2025550199;phone-context=acme.com"
        assert format_tel_uri("202-555-0199") == ""
        assert format_tel_uri("+44 20 7946 0991") == "tel:+442079460991"

    def test_cors_exposes_telemetry_headers(self):
        RateLimitMiddleware.reset()
        resp = client.post(
            "/prospect-discovery",
            headers={"Origin": "http://localhost:4000"},
            json={"query": ""},
        )
        expose = resp.headers.get("access-control-expose-headers", "")
        assert "X-LLM-Provider" in expose
        assert "X-Platforms-Succeeded" in expose

    def test_google_excludes_social_mirrors_and_cleans_trailing_handles(self, monkeypatch):
        from app.platforms.google import GoogleAdapter
        from app.services.search_service import search_service

        g = GoogleAdapter()
        for mirror_url in (
            "https://twstalker.com/Ms_Felicia_",
            "https://nitter.net/someone",
            "https://nitter.privacydev.net/someone",
            "https://www.sotwe.com/handle",
            "https://picuki.com/profile/someone",
            "https://imginn.com/someone/",
        ):
            assert not g.validate_url(mirror_url), f"Expected mirror rejected: {mirror_url}"

        assert clean_name("MsFelicia @Ms_Felicia_", "Google") == "MsFelicia"
        assert clean_name("Jane Doe @janedoe", "X") == "Jane Doe"

        RateLimitMiddleware.reset()
        query_service.clear_cache()

        async def mock_search(query_str: str, num_results: int = 5):
            if "about.me" in query_str:
                return SearchOutcome(
                    hits=[
                        {
                            "title": "MsFelicia @Ms_Felicia_",
                            "link": "https://twstalker.com/Ms_Felicia_",
                            "snippet": "Football lover in Washington DC, USA",
                        },
                        {
                            "title": "Dan Sreebny - Washington DC",
                            "link": "https://about.me/dansreebny",
                            "snippet": "Football lover based in Washington DC, USA.",
                        },
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="ddg",
                )
            return SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="ddg")

        monkeypatch.setattr(search_service, "search_with_status", mock_search)
        resp = client.post(
            "/prospect-discovery",
            json={"query": "Football lovers in Washington DC in USA"},
        )
        assert resp.status_code == 200
        body = resp.json()
        urls = [sl["link"] for p in body["prospects"] for sl in p["social_link"]]
        assert not any("twstalker.com" in u for u in urls)
        assert any("about.me/dansreebny" in u for u in urls)

    def test_sample_business_names_excluded_on_individual_queries_e2e(self, monkeypatch):
        from app.services.search_service import search_service

        RateLimitMiddleware.reset()
        query_service.clear_cache()

        async def mock_london(query_str: str, num_results: int = 5):
            if "facebook.com" in query_str:
                return SearchOutcome(
                    hits=[
                        {
                            "title": "Grow London Local | Facebook",
                            "link": "https://www.facebook.com/GrowLondonLocal",
                            "snippet": "Community for startup founders in London, UK.",
                        }
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="ddg",
                )
            if "youtube.com" in query_str:
                return SearchOutcome(
                    hits=[
                        {
                            "title": "Startups of London - YouTube",
                            "link": "https://www.youtube.com/c/startupsoflondon",
                            "snippet": "Startup founders in London, UK.",
                        }
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="ddg",
                )
            if "linkedin.com" in query_str:
                return SearchOutcome(
                    hits=[
                        {
                            "title": "Charlie Walker - Co-Founder | LinkedIn",
                            "link": "https://www.linkedin.com/in/charlie-walker-846184112",
                            "snippet": "Startup founder in London, UK.",
                        }
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="ddg",
                )
            return SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="ddg")

        monkeypatch.setattr(search_service, "search_with_status", mock_london)
        r_lon = client.post("/prospect-discovery", json={"query": "Startup founders in London UK"})
        assert r_lon.status_code == 200
        names_lon = [p["name"] for p in r_lon.json()["prospects"]]
        assert names_lon == ["Charlie Walker"]

        async def mock_ny(query_str: str, num_results: int = 5):
            if "facebook.com" in query_str:
                return SearchOutcome(
                    hits=[
                        {
                            "title": "Main Street ROI | Facebook",
                            "link": "https://www.facebook.com/MainStreetROI",
                            "snippet": "Digital marketing agency serving small businesses in New York.",
                        }
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="ddg",
                )
            if "about.me" in query_str:
                return SearchOutcome(
                    hits=[
                        {
                            "title": "Winston Digital Marketing",
                            "link": "https://www.winstondigitalmarketing.com",
                            "snippet": "Full-service digital marketing agency in New York, USA.",
                        }
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="ddg",
                )
            if "linkedin.com" in query_str:
                return SearchOutcome(
                    hits=[
                        {
                            "title": "Steven Rank - Agency Owner | LinkedIn",
                            "link": "https://www.linkedin.com/in/steven-rank-038b242",
                            "snippet": "Digital marketing agency owner in New York, USA.",
                        }
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="ddg",
                )
            return SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="ddg")

        RateLimitMiddleware.reset()
        monkeypatch.setattr(search_service, "search_with_status", mock_ny)
        r_ny = client.post(
            "/prospect-discovery",
            json={"query": "Digital marketing agency owners in New York USA"},
        )
        assert r_ny.status_code == 200
        names_ny = [p["name"] for p in r_ny.json()["prospects"]]
        assert names_ny == ["Steven Rank"]

    def test_identity_gate_rejects_short_handles_and_unverified_unknowns(self, monkeypatch):
        from app.services.search_service import search_service

        RateLimitMiddleware.reset()
        query_service.clear_cache()

        async def mock_dubai(query_str: str, num_results: int = 5):
            if "site:x.com" in query_str:
                return SearchOutcome(
                    hits=[
                        {
                            "title": "AJ (@TheSerbianG) / X",
                            "link": "https://x.com/TheSerbianG",
                            "snippet": "Restaurant owners in Dubai UAE food scene.",
                        },
                        {
                            "title": "Raph (@raph_thoug) / X",
                            "link": "https://x.com/raph_thoug",
                            "snippet": "Restaurant owner insights in Dubai UAE.",
                        },
                        {
                            "title": "Tariq Al Mansoor (@tariqalmansoor) / X",
                            "link": "https://x.com/tariqalmansoor",
                            "snippet": "Restaurant owner in Dubai, UAE.",
                        },
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="ddg",
                )
            return SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="ddg")

        monkeypatch.setattr(search_service, "search_with_status", mock_dubai)
        resp = client.post(
            "/prospect-discovery?include_diagnostics=true",
            json={"query": "Restaurant owners in Dubai UAE"},
        )
        assert resp.status_code == 200
        body = resp.json()
        names = [p["name"] for p in body["prospects"]]
        assert "AJ" not in names
        assert "Raph" not in names
        assert names == ["Tariq Al Mansoor"]
        assert body["provider_health"]["platform_metrics"]["X"]["insufficient_identity_evidence"] == 2

    def test_generic_company_email_not_attached_to_unrelated_individuals(self, monkeypatch):
        from app.services.search_service import search_service

        RateLimitMiddleware.reset()
        query_service.clear_cache()

        async def mock_search(query_str: str, num_results: int = 5):
            if "site:linkedin.com" in query_str:
                return SearchOutcome(
                    hits=[
                        {
                            "title": "Charlie Walker - Startup Founder | LinkedIn",
                            "link": "https://www.linkedin.com/in/charlie-walker-846184112",
                            "snippet": "Startup founder in London, UK. cofounder@rephrase.ai charlie@walkerventures.uk",
                        },
                        {
                            "title": "Rahim Nathwani - Startup Founder | LinkedIn",
                            "link": "https://www.linkedin.com/in/rahimnathwani",
                            "snippet": "Startup founder in London, UK. Contact cofounder@rephrase.ai or info@acme.io",
                        },
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="ddg",
                )
            return SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="ddg")

        monkeypatch.setattr(search_service, "search_with_status", mock_search)
        resp = client.post("/prospect-discovery", json={"query": "Startup founders in London UK"})
        assert resp.status_code == 200
        prospects = resp.json()["prospects"]
        assert len(prospects) == 2
        all_links = [sl["link"] for p in prospects for sl in p["social_link"]]
        assert "mailto:cofounder@rephrase.ai" not in all_links
        assert "mailto:info@acme.io" not in all_links
        assert "mailto:charlie@walkerventures.uk" in all_links

    def test_local_phone_omits_social_domain_phone_context(self):
        from app.services.profile_extractor import profile_extractor

        cand_li = RawCandidate(
            platform="LinkedIn",
            name="Alex Morgan",
            profile_url="https://www.linkedin.com/in/alexmorgan",
            title="Alex Morgan",
            snippet="Call 202-555-0199 or +44 20 7946 0991",
        )
        enriched_li = profile_extractor.enrich_candidate(
            cand_li,
            StructuredIntent(interest="Startups", audience="Startup founders"),
        )
        phone_links_li = [sl.link for sl in enriched_li.social_links if sl.platform == "Phone"]
        assert phone_links_li == ["tel:+442079460991"]
        assert not any("linkedin.com" in link for link in phone_links_li)

        cand_site = RawCandidate(
            platform="Google",
            name="Acme Bistro",
            profile_url="https://acmebistro.ae/contact",
            title="Acme Bistro",
            snippet="Call 202-555-0199",
        )
        enriched_site = profile_extractor.enrich_candidate(
            cand_site,
            StructuredIntent(interest="Restaurants", audience="Restaurants", target_type="business"),
        )
        assert [sl.link for sl in enriched_site.social_links if sl.platform == "Phone"] == [
            "tel:2025550199;phone-context=acmebistro.ae"
        ]

    def test_ddg_typed_rate_limit_and_anti_bot_diagnostics(self, monkeypatch):
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

        svc = SearchService()

        # 1. Typed RatelimitException (even with generic message text) -> RATE_LIMITED
        class FakeDDGSRate:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def text(self, q, max_results=5):
                raise RatelimitException("upstream throttle")

        import ddgs
        monkeypatch.setattr(ddgs, "DDGS", FakeDDGSRate)
        out_rl = asyncio.run(svc._search_ddg("test query", 5))
        assert out_rl.status == SearchStatus.RATE_LIMITED
        assert out_rl.exception_type == "RatelimitException"
        assert out_rl.http_status == 429
        assert out_rl.failure_reason == "rate_limited"
        assert out_rl.retry_count == 1

        # 2. HTTP 202 anti-bot challenge -> PROVIDER_ERROR + anti_bot_or_challenge (never RATE_LIMITED)
        class FakeDDGSBot:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def text(self, q, max_results=5):
                raise DDGSException("HTTP 202 captcha challenge triggered")

        monkeypatch.setattr(ddgs, "DDGS", FakeDDGSBot)
        out_bot = asyncio.run(svc._search_ddg("test query", 5))
        assert out_bot.status == SearchStatus.PROVIDER_ERROR
        assert out_bot.http_status == 202
        assert out_bot.failure_reason == "anti_bot_or_challenge"

        # 3. Typed TimeoutException -> TIMEOUT
        class FakeDDGSTimeout:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def text(self, q, max_results=5):
                raise TimeoutException("timed out")

        monkeypatch.setattr(ddgs, "DDGS", FakeDDGSTimeout)
        out_to = asyncio.run(svc._search_ddg("test query", 5))
        assert out_to.status == SearchStatus.TIMEOUT
        assert out_to.failure_reason == "timeout"

    def test_llm_targeted_retry_on_unsupported_reasoning_param_only(self, monkeypatch):
        from types import SimpleNamespace
        from app.config import settings

        valid_intent_json = json.dumps({
            "interest": "Startups",
            "audience": "Startup founders",
            "location": {"city": "London", "state": None, "country": "UK"},
            "target_type": "individual",
            "keywords": ["startup"],
        })

        class Param400Error(Exception):
            def __init__(self, msg: str):
                super().__init__(msg)
                self.status_code = 400

        # 1. OpenAI: retries once when 400 rejects reasoning_effort, succeeds without it
        calls = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.beta = SimpleNamespace(
                    chat=SimpleNamespace(completions=SimpleNamespace(parse=self._parse))
                )
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            async def _parse(self, **kwargs):
                calls.append(kwargs)
                if "reasoning_effort" in kwargs:
                    raise Param400Error("Unsupported parameter: 'reasoning_effort' is not supported for this model")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        parsed=StructuredIntent.model_validate_json(valid_intent_json),
                        content=valid_intent_json,
                    ))],
                    usage=SimpleNamespace(prompt_tokens=90, completion_tokens=40),
                )

        import openai
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(settings, "OPENAI_MODEL", "gpt-6-luna")
        monkeypatch.setattr(openai, "AsyncOpenAI", FakeOpenAI)
        intent = asyncio.run(llm_service._call_openai("Startup founders in London UK", 5.0))
        assert intent.audience == "Startup founders"
        assert len(calls) == 2
        assert "reasoning_effort" in calls[0]
        assert "reasoning_effort" not in calls[1]

        # 2. Non-reasoning 400 (e.g. model_not_found) MUST NOT trigger single-provider retry
        calls.clear()

        class FakeOpenAIBadModel(FakeOpenAI):
            async def _parse(self, **kwargs):
                calls.append(kwargs)
                raise Param400Error("The model `bad-model` does not exist (model_not_found)")

        monkeypatch.setattr(openai, "AsyncOpenAI", FakeOpenAIBadModel)
        with pytest.raises(Param400Error):
            asyncio.run(llm_service._call_openai("Startup founders in London UK", 5.0))
        assert len(calls) == 1

    def test_dc_gridiron_and_topic_channel_rejected_on_individual_queries(self, monkeypatch):
        from app.services.search_service import search_service
        from app.utils.normalizers import _handle_corroborates_name

        assert not _handle_corroborates_name("Dc Gridiron", "YouTube", "https://www.youtube.com/@dcgridiron")
        assert _handle_corroborates_name("Alex Morgan", "X", "https://x.com/alexmorgan")
        assert (
            classify_candidate_entity(
                "Dc Gridiron",
                "YouTube",
                "https://www.youtube.com/@dcgridiron",
                "Dc Gridiron - YouTube",
                "Washington DC football highlights",
            )
            == "business"
        )

        RateLimitMiddleware.reset()
        query_service.clear_cache()

        async def mock_dc(query_str: str, num_results: int = 5):
            if "youtube.com" in query_str:
                return SearchOutcome(
                    hits=[
                        {
                            "title": "Dc Gridiron - YouTube",
                            "link": "https://www.youtube.com/@dcgridiron",
                            "snippet": "Football lovers in Washington DC, USA.",
                        }
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="ddg",
                )
            if "linkedin.com" in query_str:
                return SearchOutcome(
                    hits=[
                        {
                            "title": "Dan Sreebny | LinkedIn",
                            "link": "https://www.linkedin.com/in/dan-sreebny-06b57718",
                            "snippet": "Football lover in Washington DC, District of Columbia.",
                        }
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="ddg",
                )
            return SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="ddg")

        monkeypatch.setattr(search_service, "search_with_status", mock_dc)
        resp = client.post(
            "/prospect-discovery?include_diagnostics=true",
            json={"query": "Football lovers in Washington DC in USA"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert [p["name"] for p in body["prospects"]] == ["Dan Sreebny"]
        assert body["provider_health"]["platform_metrics"]["YouTube"]["target_type_mismatches"] == 1

    def test_unverified_and_conflicting_locations_rejected(self, monkeypatch):
        from app.services.search_service import search_service

        # 1. Explicit conflicting city/country rejected
        intent_dubai = StructuredIntent(
            interest="Restaurants",
            audience="Restaurant owners",
            location=LocationDetails(city="Dubai", country="UAE"),
            target_type="individual",
        )
        manchester_cand = RawCandidate(
            platform="LinkedIn",
            name="Nam Tran",
            location="Manchester, UK",
            profile_url="https://www.linkedin.com/in/nam-tran-7b642547",
            title="Nam Tran - Restaurant Owner",
            snippet="Restaurant owner in Manchester UK formerly visited Dubai.",
            entity_type="individual",
        )
        dubai_cand = RawCandidate(
            platform="LinkedIn",
            name="Vince Farahjian",
            location="Dubai, UAE",
            profile_url="https://www.linkedin.com/in/vince-farahjian-ab517383",
            title="Vince Farahjian - Restaurant Owner",
            snippet="Restaurant owner in Dubai, UAE.",
            entity_type="individual",
        )
        assert [
            c.name
            for c in relevance_service.filter_relevant_candidates(
                [manchester_cand, dubai_cand], intent_dubai
            )
        ] == ["Vince Farahjian"]

        # 2. Candidate without ANY location confirmation rejected on city query
        RateLimitMiddleware.reset()
        query_service.clear_cache()

        async def mock_london(query_str: str, num_results: int = 5):
            if "x.com" in query_str:
                return SearchOutcome(
                    hits=[
                        {
                            "title": "Miguel Carranza (@elwatto) / X",
                            "link": "https://x.com/elwatto",
                            "snippet": "Startup founder sharing SaaS tips.",
                        },
                        {
                            "title": "Henning Sanden (@henningsanden) / X",
                            "link": "https://x.com/henningsanden",
                            "snippet": "Startup founder based in London.",
                        },
                    ],
                    status=SearchStatus.SUCCESS,
                    provider="ddg",
                )
            return SearchOutcome(hits=[], status=SearchStatus.NO_RESULTS, provider="ddg")

        monkeypatch.setattr(search_service, "search_with_status", mock_london)
        resp = client.post(
            "/prospect-discovery?include_diagnostics=true",
            json={"query": "Startup founders in London UK"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert [p["name"] for p in body["prospects"]] == ["Henning Sanden"]
        assert body["provider_health"]["platform_metrics"]["X"]["location_unverified"] == 1

    def test_default_config_host_binds_loopback(self):
        from app.config import Settings

        assert Settings(_env_file=None).HOST == "127.0.0.1"






