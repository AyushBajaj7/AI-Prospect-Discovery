# AI-Powered Multi-Platform Prospect Discovery API

[![CI Matrix](https://github.com/AyushBajaj7/AI-Prospect-Discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/AyushBajaj7/AI-Prospect-Discovery/actions/workflows/ci.yml)
[![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](.github/workflows/ci.yml)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Validation](https://img.shields.io/badge/Validation-Pydantic%20v2-orange.svg)](https://docs.pydantic.dev/)
[![Tests](https://img.shields.io/badge/Tests-160%20Passing-brightgreen.svg)](tests/test_prospect_discovery.py)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An intelligent, modular prospect discovery engine built with **FastAPI**, **Pydantic v2**, and modern structured-output **LLM SDKs** (`google-genai`, `openai`, `anthropic`). It accepts natural-language queries, parses structured intent with multi-provider fallback, discovers publicly accessible profiles across **9 discovery platforms**, normalizes and validates public contact channels (`Email`, `Phone`, `WhatsApp`), filters strictly by weighted relevance, deduplicates across platforms, and returns strict assessment JSON (`success`, `query`, `total_prospects`, `prospects`) with telemetry in CORS-exposed `X-*` HTTP headers by default (and full `provider_health` + per-platform `platform_metrics` funnel diagnostics via `?include_diagnostics=true`).

---

## 1. Architecture & Processing Flow

```
                          User Natural Language Query
                                      │
                                      ▼
                     Input Validation & Rate Limiting
              (64 KB ASGI cap, per-IP sliding window, CORS)
                                      │
                                      ▼
                 Query Understanding Layer (LLM Fallback Chain)
        (OpenAI GPT-6 Luna -> Gemini 3.5 Flash-Lite -> Claude Haiku 5.5
                 -> Offline Deterministic + Dynamic NLP Fallback)
                                      │
                                      ▼
                  Sanitized Search Query Generator (9 Platforms)
          (Strips quotes & search operators before constructing queries)
                                      │
                                      ▼
              Bounded Concurrent Discovery (HTTPX Lifespan Pool)
  ┌────────┬──────────┬───┬─────────┬─────────┬────────┬───────┬────────┬───────┐
  │ Google │ LinkedIn │ X │ YouTube │Instagram│Facebook│Threads│WhatsApp│ Email │
  └───┬────┴────┬─────┴─┬─┴────┬────┴────┬────┴───┬────┴───┬───┴────┬───┴───┬───┘
      └─────────┴───────┴──────┴─────────┴────────┴────────┴────────┴───────┘
                                      │
                                      ▼
                     Profile Extraction & Contact Validation
          (Strict URL rules, verified names, RFC Email, E.164 Phone,
           officially normalized WhatsApp links without '+' prefix)
                                      │
                                      ▼
                      Strict Relevance Filtering (0 – 100)
                (Audience: 40 | Interest: 20 | Location: 25 | Evidence: 15)
                                      │
                                      ▼
                  Conflict-Aware Deduplication & Consolidation
            (Same-platform conflict guard, surname/initial rules,
             no generic-email cross-person merges)
                                      │
                                      ▼
   Strict JSON Response + CORS-Exposed Header Telemetry (and ?include_diagnostics=true)
                 (FastAPI POST http://localhost:4000/prospect-discovery)
```

---

## 2. Supported Discovery Platforms & Contact Channels

All **9 discovery platforms** specified in the assessment run concurrently with explicit `SearchStatus` tracking (`SUCCESS`, `NO_RESULTS`, `RATE_LIMITED`, `TIMEOUT`, `PROVIDER_ERROR`):

| No. | Platform | Discovery Target | Methodology & Validation |
|:---:|---|---|---|
| **1** | **Google** | Personal websites & portfolios | Targets personal sites and `about.me` pages while excluding aggregators, news articles, and event directories. |
| **2** | **LinkedIn** | Public professional profiles | Targets `site:linkedin.com/in/`. Blocks posts, company pages, jobs, and login URLs. |
| **3** | **X (Twitter)** | Public user profiles | Targets `site:x.com` excluding `/status`, `/hashtag`, `/explore`, and `/login`. |
| **4** | **YouTube** | Public creator channels | Targets `site:youtube.com` (`@handle`, `/channel/`, `/c/`, `/user/`) excluding `/watch`, `/shorts`, `/playlist`. |
| **5** | **Instagram** | Public profiles | Targets `site:instagram.com` excluding `/p/`, `/reel`, `/stories`, `/explore`. |
| **6** | **WhatsApp** | Public contact links | Discovers public `wa.me/` and `api.whatsapp.com/send` links and normalizes them to `https://wa.me/<digits>` (no `+`). |
| **7** | **Facebook** | Public profiles & pages | Targets public creator pages, `/people/<name>/<id>`, and `profile.php?id=...` excluding groups/login/events. |
| **8** | **Email** | Public email addresses | Dedicated `EmailAdapter` + snippet extractor finding RFC-valid `mailto:` contacts with bogus-domain rejection. |
| **9** | **Threads** | Public profiles | Targets `site:threads.net/@<handle>` excluding `/post/`. |

> **Contact Links Note:** Per the assessment output rules, publicly discovered **`Phone`** numbers (`tel:+...`) are also extracted and included inside each prospect's `social_link` array alongside `Email` (`mailto:...`) and `WhatsApp`. `Phone` is an extracted contact channel, not a tenth discovery platform.
>
> **Platform Limitations:** Major social networks (LinkedIn, Instagram, Facebook, X, Threads) do not offer unauthenticated people-search APIs. Rather than bypassing logins or scraping private pages, this engine queries **search-indexed public profiles** via Serper, Brave Search API, or DuckDuckGo fallback (limited to 4 concurrent queries via semaphore).

---

## 3. API Specification

### Endpoint

```http
POST http://localhost:4000/prospect-discovery
Content-Type: application/json
```

### Request Body

```json
{
  "query": "Startup founders in London UK"
}
```

### Success Response (`200 OK`)

*(Default response strictly matches the assessment contract: `success`, `query`, `total_prospects`, and `prospects`. Provider health telemetry is returned via `X-LLM-Provider` / `X-Platforms-*` HTTP headers, or inside the JSON body when `?include_diagnostics=true` is requested.)*

```json
{
  "success": true,
  "query": "Startup founders in London UK",
  "total_prospects": 1,
  "prospects": [
    {
      "name": "Alex Morgan",
      "location": "London, UK",
      "social_link": [
        { "platform": "LinkedIn", "link": "https://www.linkedin.com/in/alexmorgan" },
        { "platform": "X", "link": "https://x.com/alexmorgan" },
        { "platform": "Email", "link": "mailto:alex@techventures.co" },
        { "platform": "Phone", "link": "tel:+442079460991" },
        { "platform": "WhatsApp", "link": "https://wa.me/442079460991" }
      ]
    }
  ]
}
```

### Error Responses

- **400 Bad Request**: Malformed `Content-Length` header.
- **413 Request Entity Too Large**: Declared or chunked/streamed body exceeding 64 KB.
- **422 Unprocessable Entity**: Empty, whitespace-only, non-alphabetic, or schema-invalid query.
- **429 Too Many Requests**: Per-IP sliding-window rate limit exceeded (returns `Retry-After` and `X-Request-ID`).
- **500 Internal Server Error**: Sanitized error returning a `request_id` without exposing stack traces.

---

## 4. Security & Production Hardening

| Security Control | Implementation Details |
|---|---|
| **Prompt & Search Operator Injection** | System prompts are passed via dedicated SDK system roles (`system_instruction` / `system`), and all parsed `StructuredIntent` fields pass through `sanitize_search_term()` to strip `site:`, `inurl:`, `-site:`, `OR`, quotes, and control characters before query generation. |
| **Endpoint Rate Limiting** | Pure ASGI `RateLimitMiddleware` uses `scope["client"][0]` by default, trusts `X-Forwarded-For` only from configured `TRUSTED_PROXIES`, purges expired buckets, and bounds memory via `RATE_LIMIT_MAX_CLIENTS` LRU eviction (returning `429` + `Retry-After`). *(Back with Redis or gateway throttling in multi-worker clusters.)* |
| **Payload Size Protection** | Pure ASGI `BodySizeLimitMiddleware` enforces a strict 64 KB cap on both `Content-Length` headers and chunked/streaming bodies. |
| **CORS & Error Sanitization** | CORS uses an explicit `CORS_ORIGINS` allowlist (`allow_credentials=False`). Exception handlers return sanitized messages correlated via `X-Request-ID`. |
| **Host Validation & SSRF Posture** | The engine never fetches untrusted candidate profile URLs over HTTP (only fixed upstream Serper/Brave/DDG endpoints are contacted). Additionally, `is_valid_profile_url()` blocks private/loopback/link-local IPv4 and IPv6 literals (`127.0.0.1`, `[::1]`, `[fe80::1]`, `[fc00::1]`). *(If external profile-page fetching is added later, DNS resolution, rebinding checks, and redirect revalidation must be added.)* |
| **ReDoS & Bounded Scanning** | Contact regexes avoid nested unbounded quantifiers and cap input text scanning to 10,000 characters (`_MAX_SCAN_CHARS`). |
| **Contact & Verification Honesty** | Social URLs are labeled `url_format_validated` (or `search_indexed_web` for general web hits), not live-verified ownership. Emails reject bogus domains (`example.com`, `test.com`); phones normalize to RFC 3966 `tel:` URIs without spaces and reject date/year ranges (`2020-2025`); WhatsApp links strip `+` (`https://wa.me/<digits>`); locations remain `null` unless confirmed in text. |

---

## 5. Entity Resolution & Duplicate Detection

1. **Exact Normalized Profile URL**: Canonicalizes regional LinkedIn subdomains (`uk.linkedin.com` / `ge.linkedin.com` -> `www.linkedin.com`) and strips tracking parameters before matching.
2. **Conflict Guard (Exclusive Platforms)**: Two different accounts on the same social platform (`LinkedIn` + `LinkedIn`, `X` + `X`, etc.) are never merged—even transitively—preventing different individuals from collapsing together.
3. **Personal Email Match**: Shared non-generic emails merge only when names match via exact full name or surname + initial (`Sam Patel` + `S. Patel`). Generic company addresses (`info@`, `contact@`, `cofounder@`) never link distinct people such as `John Smith` vs `John Doe`.
4. **Verified Phone / Platform Handle + Location**: Compatible tokens only (`India` does not match `Indiana, USA`).
5. **Exact Full Name + Both Verified Locations**: Multi-word full name match with non-conflicting verified locations on both candidates.

---

## 6. Relevance Scoring & Entity Filtering

$$\text{Score} = \text{Audience Match (40)} + \text{Interest Match (20)} + \text{Location Match (25)} + \text{Evidence Quality (15)}$$

Before scoring, candidate entities are classified as `individual`, `business`, `unknown`, or `parody_or_gimmick` (using both organizational and hospitality indicators plus explicit parody/gimmick signals without penalizing lowercase personal display names). Confirmed `parody_or_gimmick` accounts and explicit `target_type` mismatches (such as organizations on individual-target queries) are dropped while `unknown` candidates remain eligible for relevance scoring, and only candidates scoring `>= RELEVANCE_THRESHOLD` (default `35.0`) are returned—never falling back to sub-threshold results. Per-platform funnel rejections (`invalid_urls`, `rejected_names`, `missing_required_contact`, `parody_or_gimmick`, `target_type_mismatches`, `below_relevance_threshold`, `retained_candidates`) are logged at `INFO` level and reported in `provider_health.platform_metrics` when `?include_diagnostics=true`.

---

## 7. LLM & Search API Research – Cost, Speed, and Accuracy

### A. LLM Model Comparison & Selection

Models using retired/deprecated SDKs or IDs (`google-generativeai`, `gpt-4o-mini`, `claude-3-5-haiku-20241022`, and `gemini-2.5-flash` scheduled to route to Gemini 3.5 Flash-Lite on Dec 1, 2026) were replaced with current native structured-output SDKs and minimum supported versions (`openai>=1.50.0`, `google-genai>=1.0.0`, `anthropic>=0.77.0`):

| Evaluation Criteria | OpenAI `gpt-6-luna` | Anthropic `claude-haiku-5-5` | Google `gemini-3.1-flash-lite` | Google `gemini-3.5-flash-lite` |
|---|:---:|:---:|:---:|:---:|
| **SDK Used** | `openai` (`beta.chat.completions.parse`) | `anthropic` (`output_config.format` `json_schema`) | `google-genai` (`client.aio` + `response_schema`) | `google-genai` (`client.aio` + `response_schema`) |
| **Input Price / 1M Tokens *(published)*** | **$0.10** ($0.05 Batch/Flex) | **$0.10** (≤100K prompt) | $0.25 | $0.30 |
| **Output Price / 1M Tokens *(published)*** | **$0.50** ($0.25 Batch/Flex) | **$0.50** (≤100K prompt) | $1.50 | $2.50 *(incl. thinking)* |
| **Native JSON Schema** | Yes | Yes (`output_config.format`) | Yes (`response_schema`) | Yes (`response_schema`) |
| **Est. LLM Cost / Req *(est. 309 in / 52 out)*** | **$0.0000569** | **$0.0000569** | $0.0001553 | $0.0002227 |
| **Est. LLM Cost / 1,000 Req *(estimated)*** | **$0.0569** ($0.0285 Flex) | **$0.0569** | $0.1553 | $0.2227 |

*(Estimated character-heuristic token counts via `scripts/benchmark_llms.py` using `len(text) // 4`: ~309 input tokens and ~52 visible JSON output tokens—reflecting the visible JSON schema only, NOT live API-billed token telemetry or reasoning/thinking tokens billed under output pricing. All three SDK adapters explicitly minimize thinking/reasoning overhead in code via `reasoning_effort="none"` on OpenAI, `thinking={"type": "disabled"}` + `output_config={"effort": "low"}` on Anthropic, and `ThinkingConfig(thinking_level="MINIMAL")` with `thoughts_token_count` accounting on Gemini. At 296 input / 52 output tokens without thinking tokens, `gemini-3.5-flash-lite` is `$0.0002188` per request or `$0.2188` per 1,000 requests; actual billed costs vary across model tiers and live workloads.)*

### B. Baseline Search API Cost Estimates (9 Queries per Uncached Discovery)

Baseline estimates assuming 9 uncached queries per `POST /prospect-discovery` request before credits, caching, retries, or fallback-provider calls:

| Search Provider | Published Rate / 1K Queries | Est. Search Cost / Discovery | Est. Search Cost / 1,000 Discoveries |
|---|:---:|:---:|:---:|
| **Serper Starter** | $1.00 | **$0.0090** | **$9.00** |
| **Brave Search** | $5.00 | $0.0450 | $45.00 |
| **DuckDuckGo Fallback** | $0.00 *(no paid API fee)* | $0.0000 | $0.00 |

### C. Combined Baseline Estimate per Discovery (Uncached, Before Credits/Retries/Caching)

| Stack Configuration | Est. LLM Cost / Req | Est. Search Cost / Req | Baseline Combined / Req | Baseline Combined / 1K |
|---|---:|---:|---:|---:|
| **Serper Starter + `gpt-6-luna` / `claude-haiku-5-5`** | $0.0000569 | $0.0090 | **$0.009057** | **~$9.06** |
| **Serper Starter + `gemini-3.1-flash-lite`** | $0.0001553 | $0.0090 | **$0.009155** | **~$9.16** |
| **Serper Starter + `gemini-3.5-flash-lite`** | $0.0002227 | $0.0090 | **$0.009223** | **~$9.22** |
| **Brave Search + `gpt-6-luna`** | $0.0000569 | $0.0450 | $0.045057 | ~$45.06 |

**Key Cost Finding:** Search API queries cost **~40x to ~790x more** than LLM query parsing. Consequently, our bounded **TTL + LRU cache** (`CACHE_TTL_SECONDS=3600`, caching both `StructuredIntent` and non-empty search results—never caching failures) and **multi-provider fallback** (`OpenAI -> Gemini -> Anthropic -> offline NLP`) provide the largest real-world cost reduction.

**Final Recommendation:** **`gpt-6-luna`** paired with **Serper Starter** offers the lowest combined API cost (`~$0.00906` per uncached discovery), with **`gemini-3.5-flash-lite`** and **`claude-haiku-5-5`** configured as live automatic failovers before offline heuristic parsing.

---

## 8. Practical Test Cases & Verified Live Results

### A. Query Understanding & Target Type Consistency

Rule: `target_type` is `"individual"` whenever the query targets human roles (founders, owners, lovers, instructors), and `"business"` only when targeting organization entities directly (`"Digital marketing agencies"`). Location understanding uses LLMs when configured and a general heuristic+dynamic location parser offline (`"Nairobi Kenya"`, `"Reykjavik Iceland"`, `"Austin, Texas, USA"`).

| No. | Search Query | Audience | Interest | Location | Target Type |
|:---:|---|---|---|---|:---:|
| **1** | `Football lovers in Washington DC in USA` | Football lovers | Football | Washington DC, District of Columbia, USA | individual |
| **2** | `Digital marketing agency owners in New York USA` | Digital marketing agency owners | Digital Marketing | New York, New York, USA | individual |
| **3** | `Startup founders in London UK` | Startup founders | Startups | London, UK | individual |
| **4** | `Cricket lovers in Ahmedabad India` | Cricket lovers | Cricket | Ahmedabad, Gujarat, India | individual |
| **5** | `Restaurant owners in Dubai UAE` | Restaurant owners | Hospitality & Restaurant | Dubai, UAE | individual |

### B. Real Captured Outputs Across All 5 Queries (`TestClient` Live Run)

Executed live via `POST /prospect-discovery?include_diagnostics=true` using DuckDuckGo fallback (zero paid API keys). Every record, `total_prospects`, `request_id`, and `platform_metrics` summary below is captured from the **exact same HTTP response** (and verified on default `/` for exact keys `["prospects", "query", "success", "total_prospects"]`) in [`samples/mandatory_queries_results.json`](samples/mandatory_queries_results.json):

1. **`Football lovers in Washington DC in USA`** -> `total_prospects: 5` (`retained=5`; `7` platforms `SUCCESS`, `1` `TIMEOUT`, `1` `PROVIDER_ERROR`; across `LinkedIn` and `YouTube`: `Dan Sreebny`, `Basil Mokhallalati`, `Katie Green`, `Alejandro Heredia`, `Dc Gridiron`).
2. **`Digital marketing agency owners in New York USA`** -> `total_prospects: 5` (`retained=5`; `6` platforms `SUCCESS`, `1` `TIMEOUT`, `2` `PROVIDER_ERROR`; across `LinkedIn`: `Marc Blanchard`, `Bill Decker`, `Erfan Azimi`, `Seb Weller`, `Ashley Casiano`).
3. **`Startup founders in London UK`** -> `total_prospects: 2` (`retained=2`; `6` platforms `SUCCESS`, `2` `TIMEOUT`, `1` `PROVIDER_ERROR`; across `X`: `Miguel Carranza`, `Henning Sanden`).
4. **`Cricket lovers in Ahmedabad India`** -> `total_prospects: 1` (`retained=1`; `5` platforms `SUCCESS`, `2` `TIMEOUT`, `2` `PROVIDER_ERROR`; on `X`: `amit kumar`).
5. **`Restaurant owners in Dubai UAE`** -> `total_prospects: 4` (`retained=4`; `8` platforms `SUCCESS`, `1` `PROVIDER_ERROR`; across `LinkedIn` and `Facebook`: `Samer Mustaklim`, `Vince Farahjian`, `Mr Chef Subra`, `Mitul Pandya`).

Sample excerpt from the captured strict JSON (`Football lovers in Washington DC in USA`, showing component-verified locations without unconfirmed country fabrication and zero excluded mirror/agency noise):

```json
{
  "success": true,
  "query": "Football lovers in Washington DC in USA",
  "total_prospects": 5,
  "prospects": [
    {
      "name": "Dan Sreebny",
      "location": "Washington DC, District of Columbia",
      "social_link": [
        { "platform": "LinkedIn", "link": "https://www.linkedin.com/in/dan-sreebny-06b57718" }
      ]
    },
    {
      "name": "Basil Mokhallalati",
      "location": "Washington DC, District of Columbia",
      "social_link": [
        { "platform": "LinkedIn", "link": "https://www.linkedin.com/in/basilmo" }
      ]
    },
    {
      "name": "Alejandro Heredia",
      "social_link": [
        { "platform": "LinkedIn", "link": "https://www.linkedin.com/in/alejandro-herediasa" }
      ]
    },
    {
      "name": "Dc Gridiron",
      "social_link": [
        { "platform": "YouTube", "link": "https://www.youtube.com/@dcgridiron" }
      ]
    }
  ]
}
```

Full unmodified single-response JSON payloads, diagnostic failure counters, and exact candidate-to-prospect reconciliation for all 5 queries are saved in [`samples/mandatory_queries_results.json`](samples/mandatory_queries_results.json).

---

## 9. Installation & Setup

```bash
git clone https://github.com/AyushBajaj7/AI-Prospect-Discovery.git
cd AI-Prospect-Discovery

python -m venv venv
# Linux/macOS: source venv/bin/activate
# Windows:     venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

Key environment variables (`.env.example`):

```env
PORT=4000
HOST=0.0.0.0
CORS_ORIGINS=http://localhost:4000,http://127.0.0.1:4000
RATE_LIMIT_REQUESTS=30
RATE_LIMIT_WINDOW_SECONDS=60
RATE_LIMIT_MAX_CLIENTS=10000
TRUSTED_PROXIES=

SEARCH_PROVIDER=auto
SERPER_API_KEY=your_serper_api_key_here
BRAVE_API_KEY=your_brave_api_key_here

LLM_PROVIDER=auto
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_MODEL=gpt-6-luna
GEMINI_MODEL=gemini-3.5-flash-lite
ANTHROPIC_MODEL=claude-haiku-5-5
```

---

## 10. Running the Application & Verification Scripts

```bash
# Start FastAPI server (Swagger UI at http://localhost:4000/docs)
python -m uvicorn app.main:app --host 0.0.0.0 --port 4000

# Send a discovery request
curl -X POST "http://localhost:4000/prospect-discovery" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"Startup founders in London UK\"}"

# Run cost analysis & end-to-end smoke test
python scripts/benchmark_llms.py
python scripts/smoke_test.py
```

---

## 11. Automated Testing & CI Matrix

The test suite contains **160 automated unit and integration tests** in `tests/test_prospect_discovery.py` and runs automatically across Python `3.10`, `3.11`, `3.12`, `3.13`, and `3.14` plus a `minimum-deps` compatibility job pinning declared minimum package bounds (`google-genai==1.60.0`, `httpx==0.28.1`, `anthropic==0.77.0`, `openai==1.50.0`) in [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

```bash
pytest tests/ -v
```

Coverage spans health & request-ID tracking, input validation, query understanding (5 mandatory queries + genuinely uncatalogued cities `Nairobi Kenya`, `Reykjavik Iceland`, `Austin, Texas, USA`), prompt/search-operator injection prevention, IPv4 & IPv6 anti-SSRF rules, ReDoS adversarial timing bounds, sliding-window `429` rate limiting, multi-provider LLM fallback chains, strict `RELEVANCE_THRESHOLD` filtering, `SearchStatus` / non-cached failure telemetry, official WhatsApp normalization, and entity conflict guards.

---

## 12. Deployment

```bash
# Non-root hardened Docker image (protected by .dockerignore)
docker build -t ai-prospect-discovery .
docker run -d -p 4000:4000 --env-file .env ai-prospect-discovery
```

For Render / Railway / Cloud Run: point start command to `uvicorn app.main:app --host 0.0.0.0 --port 4000` and configure secrets via the provider dashboard.

---

## 13. Verification Checklist

- [x] **FastAPI Endpoint**: `POST /prospect-discovery` running on port `4000`, Swagger UI at `/docs`.
- [x] **All 9 Discovery Platforms**: Google, LinkedIn, X, YouTube, Instagram, Facebook, Threads, WhatsApp, Email + extracted `Phone` links.
- [x] **Modern LLM SDKs & Fallback**: `openai`, `google-genai`, `anthropic` (`output_config.format`), chaining sequentially before heuristics.
- [x] **Security Hardening**: Operator-injection sanitization, `429` rate limiter, 64 KB body cap, IPv4/IPv6 checks, non-root Docker, clean Git history.
- [x] **Result Accuracy**: Strict relevance filtering, conflict-safe deduplication, evidence-only locations, official WhatsApp URL normalization.
- [x] **Test Coverage & CI**: **160/160** tests passing locally + GitHub Actions matrix (`3.10`, `3.11`, `3.12`, `3.13`, `3.14`) and `minimum-deps` job.
