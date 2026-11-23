import asyncio
import pytest
from starlette.testclient import TestClient
from app.main import app
from app.services.llm_service import llm_service
from app.services.query_service import query_service
from app.services.deduplication_service import deduplication_service
from app.services.relevance_service import relevance_service
from app.schemas import RawCandidate, StructuredIntent, LocationDetails, SocialLink
from app.utils.validators import is_valid_profile_url, is_valid_email, is_valid_phone
from app.utils.normalizers import clean_name, extract_handle

client = TestClient(app)

# 1. Test root health endpoint
def test_root_health():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoint" in data

# 2. Test natural language query understanding on the 5 mandatory assessment test queries
def test_nl_query_understanding_mandatory_cases():
    test_cases = [
        {
            "query": "Football lovers in Washington DC in USA",
            "expected_interest": "Football",
            "expected_city": "Washington DC",
            "expected_country": "USA",
        },
        {
            "query": "Digital marketing agency owners in New York USA",
            "expected_interest": "Digital Marketing",
            "expected_city": "New York",
            "expected_country": "USA",
        },
        {
            "query": "Startup founders in London UK",
            "expected_interest": "Startups",
            "expected_city": "London",
            "expected_country": "UK",
        },
        {
            "query": "Cricket lovers in Ahmedabad India",
            "expected_interest": "Cricket",
            "expected_city": "Ahmedabad",
            "expected_country": "India",
        },
        {
            "query": "Restaurant owners in Dubai UAE",
            "expected_interest": "Hospitality / Restaurant",
            "expected_city": "Dubai",
            "expected_country": "UAE",
        },
    ]

    async def _run_all():
        for tc in test_cases:
            intent = await query_service.get_structured_intent(tc["query"])
            assert intent.interest == tc["expected_interest"]
            assert intent.location.city == tc["expected_city"]
            assert intent.location.country == tc["expected_country"]

    asyncio.run(_run_all())

# 3. Test URL & Profile Validators
def test_validators():
    # Valid profile URLs
    assert is_valid_profile_url("LinkedIn", "https://www.linkedin.com/in/alex-morgan")
    assert is_valid_profile_url("X", "https://x.com/elonmusk")
    assert is_valid_profile_url("Instagram", "https://instagram.com/cristiano")
    assert is_valid_profile_url("YouTube", "https://youtube.com/@mkbhd")
    assert is_valid_profile_url("Threads", "https://threads.net/@zuck")
    assert is_valid_profile_url("WhatsApp", "https://wa.me/+1234567890")

    # Invalid URLs (posts, status, disallowed paths)
    assert not is_valid_profile_url("LinkedIn", "https://www.linkedin.com/posts/activity-123")
    assert not is_valid_profile_url("X", "https://x.com/elonmusk/status/123456789")
    assert not is_valid_profile_url("Instagram", "https://instagram.com/p/CR123456")
    assert not is_valid_profile_url("YouTube", "https://youtube.com/watch?v=abcd123")

    # Contacts validation
    assert is_valid_email("founder@startup.io")
    assert not is_valid_email("fake-email-no-domain")
    assert is_valid_phone("+1 202-555-0199")
    assert not is_valid_phone("123")

# 4. Test Normalizers
def test_normalizers():
    assert clean_name("John Doe - Founder & CEO | LinkedIn", "LinkedIn") == "John Doe"
    assert clean_name("Sarah Connor (@sarahc) / X", "X") == "Sarah Connor"
    assert clean_name("Tech Channel - YouTube", "YouTube") == "Tech Channel"
    assert extract_handle("https://x.com/startupfounder", "X") == "startupfounder"
    assert extract_handle("https://www.linkedin.com/in/john-doe-123", "LinkedIn") == "john-doe-123"

# 5. Test Profile Consolidation and Deduplication
def test_deduplication_and_consolidation():
    # Two profiles representing the same person across LinkedIn and Instagram with matching handle and location
    candidate_linkedin = RawCandidate(
        platform="LinkedIn",
        name="Alex Smith",
        location="London, UK",
        profile_url="https://linkedin.com/in/alexsmith",
        social_links=[SocialLink(platform="LinkedIn", link="https://linkedin.com/in/alexsmith")],
        metadata={"handle": "alexsmith"}
    )
    candidate_instagram = RawCandidate(
        platform="Instagram",
        name="Alex Smith",
        location="London, UK",
        profile_url="https://instagram.com/alexsmith",
        social_links=[SocialLink(platform="Instagram", link="https://instagram.com/alexsmith")],
        metadata={"handle": "alexsmith"}
    )
    # A distinct person with different name and location
    candidate_other = RawCandidate(
        platform="X",
        name="Elena Rostova",
        location="Dubai, UAE",
        profile_url="https://x.com/erostova",
        social_links=[SocialLink(platform="X", link="https://x.com/erostova")],
        metadata={"handle": "erostova"}
    )

    prospects = deduplication_service.consolidate([candidate_linkedin, candidate_instagram, candidate_other])
    assert len(prospects) == 2

    alex = next(p for p in prospects if p.name == "Alex Smith")
    assert alex.location == "London, UK"
    assert len(alex.social_link) == 2
    platforms = {sl.platform for sl in alex.social_link}
    assert "LinkedIn" in platforms
    assert "Instagram" in platforms

# 6. Test Relevance Scoring
def test_relevance_scoring():
    intent = StructuredIntent(
        interest="Startups",
        audience="Startup founders",
        location=LocationDetails(city="London", country="UK")
    )
    # Highly relevant candidate
    relevant_candidate = RawCandidate(
        platform="LinkedIn",
        name="James Wilson",
        location="London, UK",
        profile_url="https://linkedin.com/in/jameswilson",
        title="James Wilson - Startup Founder & Tech Lead in London | LinkedIn",
        snippet="Passionate startup founder building AI products in London, UK."
    )
    score_rel, _ = relevance_service.evaluate_candidate(relevant_candidate, intent)
    assert score_rel >= 60.0

    # Irrelevant candidate
    irrelevant_candidate = RawCandidate(
        platform="Google",
        name="Tokyo Weather Forecast",
        location="Tokyo, Japan",
        profile_url="https://weather.tokyo",
        title="Tokyo Daily Weather",
        snippet="Rain and sunny intervals expected across Tokyo today."
    )
    score_irrel, _ = relevance_service.evaluate_candidate(irrelevant_candidate, intent)
    assert score_irrel < 30.0

# 7. Test API Error Handling (Validation Errors & Empty Query)
def test_api_validation_errors():
    # Empty query string
    response = client.post("/prospect-discovery", json={"query": ""})
    assert response.status_code == 422 or response.status_code == 400

    # Missing query field
    response = client.post("/prospect-discovery", json={})
    assert response.status_code == 422

# 8. End-to-End API endpoint test with mock search results to verify JSON response contract
def test_end_to_end_api_contract(monkeypatch):
    # Mock search service to return deterministic mock search items
    async def mock_search(query: str, num_results: int = 5):
        if "linkedin" in query.lower():
            return [
                {
                    "title": "Alex Morgan - Co-Founder & CEO | LinkedIn",
                    "link": "https://www.linkedin.com/in/alexmorgan",
                    "snippet": "Co-founder at TechVentures based in London, UK. Contact: alex@techventures.co"
                }
            ]
        elif "x.com" in query.lower():
            return [
                {
                    "title": "Alex Morgan (@alexmorgan) / X",
                    "link": "https://x.com/alexmorgan",
                    "snippet": "Tech startup founder in London UK. Email: alex@techventures.co"
                }
            ]
        return []

    from app.services import search_service
    monkeypatch.setattr(search_service.search_service, "search", mock_search)

    response = client.post(
        "/prospect-discovery",
        json={"query": "Startup founders in London UK"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["query"] == "Startup founders in London UK"
    assert data["total_prospects"] >= 1
    
    prospect = data["prospects"][0]
    assert "Alex Morgan" in prospect["name"]
    assert "London" in prospect["location"]
    # Check that social links contain LinkedIn, X, and Email
    platforms = [sl["platform"] for sl in prospect["social_link"]]
    assert "LinkedIn" in platforms
    assert "X" in platforms
    assert "Email" in platforms
    assert any(sl["link"] == "mailto:alex@techventures.co" for sl in prospect["social_link"])

# 9. Test Partial Platform Failure Resilience
def test_partial_platform_failure_resilience(monkeypatch):
    from app.platforms import ALL_PLATFORMS

    # Pick the first adapter and simulate an unexpected crash/exception
    orig_discover = ALL_PLATFORMS[0].discover
    async def faulty_discover(*args, **kwargs):
        raise ConnectionResetError("Remote server connection reset forcibly")

    monkeypatch.setattr(ALL_PLATFORMS[0], "discover", faulty_discover)

    # Mock remaining platforms to return valid candidates
    async def mock_search(query: str, num_results: int = 5):
        return [
            {
                "title": "David Beckham - Football Legend | Instagram",
                "link": "https://www.instagram.com/davidbeckham",
                "snippet": "Football enthusiast and owner in USA."
            }
        ]

    from app.services import search_service
    monkeypatch.setattr(search_service.search_service, "search", mock_search)

    response = client.post(
        "/prospect-discovery",
        json={"query": "Football lovers in Washington DC in USA"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    # Verify that despite adapter failure, discovery succeeded with remaining platforms
    assert data["total_prospects"] >= 1

# 10. Test End-to-End Execution on All 5 Mandatory Assessment Queries
def test_all_five_assessment_queries_end_to_end(monkeypatch):
    queries = [
        "Football lovers in Washington DC in USA",
        "Digital marketing agency owners in New York USA",
        "Startup founders in London UK",
        "Cricket lovers in Ahmedabad India",
        "Restaurant owners in Dubai UAE"
    ]

    async def mock_search(query: str, num_results: int = 5):
        return [
            {
                "title": "Test Prospect - Verified Profile | LinkedIn",
                "link": "https://www.linkedin.com/in/test-prospect",
                "snippet": f"Associated with {query}. Contact: test@prospect.com"
            }
        ]

    from app.services import search_service
    monkeypatch.setattr(search_service.search_service, "search", mock_search)

    for q in queries:
        resp = client.post("/prospect-discovery", json={"query": q})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["query"] == q
        assert isinstance(body["prospects"], list)
        assert body["total_prospects"] >= 1
        first = body["prospects"][0]
        assert "name" in first
        assert "location" in first
        assert isinstance(first["social_link"], list)

