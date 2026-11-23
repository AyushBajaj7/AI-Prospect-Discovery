"""
End-to-end smoke test script.
Verifies that the FastAPI app and discovery pipeline can be initialized,
serves the root & health endpoints, and executes a real query discovery.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starlette.testclient import TestClient
from app.main import app

def run_smoke_test():
    client = TestClient(app)
    
    print("1. Testing Root endpoint...")
    r = client.get("/")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    print("   Root response:", r.json())

    print("\n2. Testing /health endpoint...")
    r = client.get("/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    print("   Health response:", r.json())
    assert "X-Request-ID" in r.headers
    print("   X-Request-ID:", r.headers["X-Request-ID"])

    print("\n3. Testing POST /prospect-discovery with 'Startup founders in London UK'...")
    # This executes query understanding, DDG/search, extraction, deduplication, scoring:
    r = client.post("/prospect-discovery", json={"query": "Startup founders in London UK"})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    print("   Status code:", r.status_code)
    print("   Success:", data.get("success"))
    print("   Total prospects found:", data.get("total_prospects"))
    if data.get("prospects"):
        print("   Sample prospect:")
        p = data["prospects"][0]
        print(f"     Name: {p.get('name')}")
        print(f"     Location: {p.get('location')}")
        print(f"     Social Links ({len(p.get('social_link', []))}):")
        for link in p.get("social_link", []):
            print(f"       - {link.get('platform')}: {link.get('link')}")
    else:
        print("   (0 prospects returned live - search provider returned empty or rate limited, but request succeeded cleanly)")

    print("\n4. Testing validation rejection on invalid query...")
    r = client.post("/prospect-discovery", json={"query": "12345"})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("   Correctly rejected numeric query with 422:", r.json())

    print("\nALL SMOKE CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_smoke_test()
