"""
FastAPI application entry-point.

Security & reliability hardening:
- Explicit CORS origins allowlist (never wildcard with credentials).
- Pure ASGI BodySizeLimitMiddleware enforcing a strict 64 KB cap (including chunked).
- Pure ASGI RequestIDMiddleware attaching X-Request-ID to every response.
- Pure ASGI RateLimitMiddleware (sliding window per client IP on POST /prospect-discovery).
- Shared HTTPX connection pool managed via FastAPI lifespan.
- Sanitized error handlers (never leak stack traces or exception messages).
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, Deque, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import settings
from app.schemas import ProspectDiscoveryResponse, ProspectQueryRequest
from app.services.prospect_service import prospect_service
from app.services.search_service import search_service

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application Lifespan (shared HTTPX client pool)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    await search_service.startup()
    try:
        yield
    finally:
        await search_service.shutdown()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Multi-platform AI prospect discovery engine. "
        "Accepts a natural-language query, discovers publicly accessible "
        "profiles across 9 platforms, deduplicates them, and returns "
        "structured JSON."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pure ASGI Middleware: per-IP sliding-window rate limiter
# Note: Process-local storage protects single-worker instances; use Redis or
# gateway throttling in multi-worker / clustered deployments.
# ---------------------------------------------------------------------------
class RateLimitMiddleware:
    """Sliding-window per-client-IP rate limiter for POST /prospect-discovery."""

    # Class-level store reference cleared easily in unit tests if needed
    _hits: Dict[str, Deque[float]] = defaultdict(deque)

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    @classmethod
    def reset(cls) -> None:
        """Reset rate-limit counters (useful in test isolation)."""
        cls._hits.clear()

    @staticmethod
    def _client_ip(scope: Scope) -> str:
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        forwarded = headers.get(b"x-forwarded-for")
        if forwarded:
            return forwarded.decode("latin-1").split(",")[0].strip() or "unknown"
        client = scope.get("client")
        return client[0] if client else "unknown"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "").upper()
        if path != "/prospect-discovery" or method != "POST":
            await self.app(scope, receive, send)
            return

        now = time.monotonic()
        window = settings.RATE_LIMIT_WINDOW_SECONDS
        limit = settings.RATE_LIMIT_REQUESTS
        ip = self._client_ip(scope)

        bucket = self._hits[ip]
        while bucket and (now - bucket[0]) >= window:
            bucket.popleft()

        if len(bucket) >= limit:
            retry_after = max(1, int(window - (now - bucket[0])))
            request_id = scope.get("state", {}).get("request_id", "unknown")
            resp = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
                content={
                    "success": False,
                    "error": "Rate limit exceeded. Please try again later.",
                    "request_id": request_id,
                },
            )
            await resp(scope, receive, send)
            return

        bucket.append(now)
        remaining = max(0, limit - len(bucket))

        async def send_with_ratelimit(message: Message) -> None:
            if message["type"] == "http.response.start":
                raw_headers = list(message.get("headers", []))
                raw_headers.extend(
                    [
                        (b"x-ratelimit-limit", str(limit).encode("ascii")),
                        (b"x-ratelimit-remaining", str(remaining).encode("ascii")),
                    ]
                )
                message["headers"] = raw_headers
            await send(message)

        await self.app(scope, receive, send_with_ratelimit)


# ---------------------------------------------------------------------------
# Pure ASGI Middleware: body-size limiter (handles declared + chunked streams)
# ---------------------------------------------------------------------------
class _BodyTooLargeError(Exception):
    """Raised internally when a streamed body exceeds MAX_BYTES."""


class BodySizeLimitMiddleware:
    """Rejects requests whose Content-Length or streamed body exceeds MAX_BYTES."""

    MAX_BYTES: int = 64 * 1024  # 64 KB

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        raw_cl = headers.get(b"content-length")
        if raw_cl is not None:
            try:
                content_length = int(raw_cl)
            except ValueError:
                resp = JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"success": False, "error": "Invalid Content-Length header."},
                )
                await resp(scope, receive, send)
                return

            if content_length < 0 or content_length > self.MAX_BYTES:
                resp = JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "success": False,
                        "error": "Request body too large.",
                        "max_bytes": self.MAX_BYTES,
                    },
                )
                await resp(scope, receive, send)
                return

        bytes_seen = 0

        async def bounded_receive() -> Message:
            nonlocal bytes_seen
            message = await receive()
            if message["type"] == "http.request":
                bytes_seen += len(message.get("body", b""))
                if bytes_seen > self.MAX_BYTES:
                    raise _BodyTooLargeError()
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except _BodyTooLargeError:
            resp = JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "success": False,
                    "error": "Request body too large.",
                    "max_bytes": self.MAX_BYTES,
                },
            )
            await resp(scope, receive, send)


# ---------------------------------------------------------------------------
# Pure ASGI Middleware: request-ID injection (runs around limiter middlewares)
# ---------------------------------------------------------------------------
class RequestIDMiddleware:
    """Attaches a UUID to each HTTP request and echoes it in X-Request-ID."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                raw_headers = list(message.get("headers", []))
                raw_headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = raw_headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)


# Registration order (last added = outermost):
#   CORSMiddleware -> RequestIDMiddleware -> BodySizeLimitMiddleware -> RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-Request-ID"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    safe_errors: list[Dict[str, Any]] = []
    for err in exc.errors():
        safe_err: Dict[str, Any] = {
            "type": err.get("type", ""),
            "loc": list(err.get("loc", [])),
            "msg": err.get("msg", ""),
        }
        if "input" in err:
            safe_err["input"] = str(err["input"])[:200]
        safe_errors.append(safe_err)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation Error",
            "details": safe_errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": str(exc.detail),
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(
        "Unhandled error [request_id=%s] on %s", request_id, request.url.path
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "An internal error occurred. Please try again later.",
            "request_id": request_id,
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"], summary="Health check")
async def root() -> dict:
    return {
        "status": "online",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "discovery_endpoint": "POST /prospect-discovery",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"], summary="Liveness probe")
async def health() -> dict:
    return {"status": "healthy"}


@app.post(
    "/prospect-discovery",
    response_model=ProspectDiscoveryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prospect Discovery"],
    summary="Discover prospects across public platforms",
    description=(
        "Accepts a natural-language query, extracts audience/location intent via LLM, "
        "concurrently discovers public profiles across Google, LinkedIn, X, YouTube, "
        "Instagram, Facebook, Threads, WhatsApp, and Email, deduplicates cross-platform "
        "matches, scores relevance strictly, and returns consolidated prospects plus "
        "provider health telemetry."
    ),
)
async def discover_prospects(
    request: ProspectQueryRequest,
    http_request: Request,
) -> ProspectDiscoveryResponse:
    request_id = getattr(http_request.state, "request_id", "unknown")
    logger.info(
        "Prospect discovery request [request_id=%s] query=%r",
        request_id,
        request.query,
    )
    try:
        result = await prospect_service.discover_prospects(request)
        logger.info(
            "Prospect discovery complete [request_id=%s] prospects=%d",
            request_id,
            result.total_prospects,
        )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Discovery pipeline error [request_id=%s]", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prospect discovery failed. Please try again later.",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        access_log=True,
    )
