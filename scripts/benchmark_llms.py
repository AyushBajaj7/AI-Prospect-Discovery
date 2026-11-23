import asyncio
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.llm_service import SYSTEM_PROMPT
from app.schemas import StructuredIntent

TEST_QUERIES = [
    "Football lovers in Washington DC in USA",
    "Digital marketing agency owners in New York USA",
    "Startup founders in London UK",
    "Cricket lovers in Ahmedabad India",
    "Restaurant owners in Dubai UAE"
]

# Published token pricing per 1M tokens (USD)
PRICING = {
    "OpenAI gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "Google Gemini 2.5 Flash": {"input": 0.075, "output": 0.30},
    "Anthropic Claude 3.5 Haiku": {"input": 0.80, "output": 4.00},
}

def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (~4 characters per token)."""
    return max(1, len(text) // 4)

def run_benchmark():
    print("=" * 80)
    print("LLM MODEL BENCHMARK & COST ANALYSIS FOR PROSPECT DISCOVERY")
    print("=" * 80)
    
    # Calculate input prompt token size
    sample_query = TEST_QUERIES[0]
    full_prompt = f"{SYSTEM_PROMPT}\n\nQuery to parse: {sample_query}"
    estimated_input_tokens = estimate_tokens(full_prompt)
    sample_output = {
        "interest": "Football",
        "audience": "Football lovers",
        "location": {"city": "Washington DC", "state": "District of Columbia", "country": "USA"},
        "target_type": "individual",
        "keywords": ["football", "lovers", "washington", "dc"]
    }
    estimated_output_tokens = estimate_tokens(json.dumps(sample_output))

    print(f"Estimated Input Tokens per Request:  ~{estimated_input_tokens} tokens")
    print(f"Estimated Output Tokens per Request: ~{estimated_output_tokens} tokens")
    print("-" * 80)
    print(f"{'Provider / Model':<30} | {'Input ($/1M)':<13} | {'Output ($/1M)':<14} | {'Cost/1K Req':<12} | {'Cost/Req ($)':<12}")
    print("-" * 80)

    for model, rates in PRICING.items():
        cost_per_req = (estimated_input_tokens / 1_000_000 * rates["input"]) + (estimated_output_tokens / 1_000_000 * rates["output"])
        cost_per_1k = cost_per_req * 1000
        print(f"{model:<30} | ${rates['input']:<12.3f} | ${rates['output']:<13.3f} | ${cost_per_1k:<11.4f} | ${cost_per_req:<12.6f}")

    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
