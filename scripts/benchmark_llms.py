"""
Cost estimation benchmark script for Prospect Discovery API:
Reports LLM token costs, Search API costs, and combined discovery estimates
across current 2026 published provider pricing.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.llm_service import SYSTEM_PROMPT

TEST_QUERIES = [
    "Football lovers in Washington DC in USA",
    "Digital marketing agency owners in New York USA",
    "Startup founders in London UK",
    "Cricket lovers in Ahmedabad India",
    "Restaurant owners in Dubai UAE",
]

# Published token pricing per 1M tokens (USD)
LLM_PRICING = {
    "OpenAI gpt-6-luna (Standard)": {"input": 0.10, "output": 0.50},
    "OpenAI gpt-6-luna (Batch/Flex)": {"input": 0.05, "output": 0.25},
    "Google Gemini 3.1 Flash-Lite": {"input": 0.25, "output": 1.50},
    "Google Gemini 3.5 Flash-Lite": {"input": 0.30, "output": 2.50},
    "Anthropic Claude Haiku 5.5 (<=100K)": {"input": 0.10, "output": 0.50},
}

# Published search pricing per 1,000 queries; each uncached discovery runs 9 platform queries
SEARCH_PRICING_PER_1K_QUERIES = {
    "Serper Starter": 1.00,
    "Brave Search": 5.00,
    "DuckDuckGo Fallback": 0.00,
}
QUERIES_PER_DISCOVERY = 9


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 characters per token)."""
    return max(1, len(text) // 4)


def run_benchmark() -> None:
    sample_query = TEST_QUERIES[0]
    full_prompt = f"{SYSTEM_PROMPT}\n\n{sample_query}"
    estimated_input_tokens = estimate_tokens(full_prompt)
    sample_output = {
        "interest": "Football",
        "audience": "Football lovers",
        "location": {
            "city": "Washington DC",
            "state": "District of Columbia",
            "country": "USA",
        },
        "target_type": "individual",
        "keywords": ["Football", "lovers"],
    }
    estimated_output_tokens = estimate_tokens(json.dumps(sample_output))

    print("=" * 88)
    print("LLM + SEARCH API COST ESTIMATES FOR PROSPECT DISCOVERY (UNCACHED)")
    print("=" * 88)
    print(f"Estimated Input Tokens per Request:  ~{estimated_input_tokens} tokens")
    print(f"Estimated Output Tokens per Request: ~{estimated_output_tokens} tokens")
    print("-" * 88)
    print(
        f"{'LLM Provider / Model':<35} | {'Input/1M':<9} | {'Output/1M':<10} | "
        f"{'Cost/1K Req':<12} | {'Cost/Req':<12}"
    )
    print("-" * 88)

    for model, rates in LLM_PRICING.items():
        cost_per_req = (
            (estimated_input_tokens / 1_000_000 * rates["input"])
            + (estimated_output_tokens / 1_000_000 * rates["output"])
        )
        cost_per_1k = cost_per_req * 1000
        print(
            f"{model:<35} | ${rates['input']:<8.2f} | ${rates['output']:<9.2f} | "
            f"${cost_per_1k:<11.4f} | ${cost_per_req:<11.7f}"
        )

    print("-" * 88)
    print("Search API Costs (Assuming 9 uncached platform queries per discovery):")
    for sp, rate in SEARCH_PRICING_PER_1K_QUERIES.items():
        per_req = (QUERIES_PER_DISCOVERY / 1000.0) * rate
        per_1k = per_req * 1000
        print(f"  {sp:<25} -> ${per_req:.4f} / discovery  (${per_1k:.2f} / 1,000 discoveries)")
    print("=" * 88)


if __name__ == "__main__":
    run_benchmark()
