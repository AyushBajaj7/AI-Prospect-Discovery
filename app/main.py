"""
FastAPI application entry-point.

Security hardening:
- CORS origins are configurable (not open wildcard with credentials).
- allow_credentials is False by default to prevent cookie-based CSRF.
- Internal exception details are never exposed to API callers.
- A unique request-ID header is added to every response for traceability.
- Request body size is bounded to 64 KB via a custom middleware.
"""
from __future__ import annotations

import logging
import uuid
from typing import Callable

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.schemas import ProspectDiscoveryResponse, ProspectQueryRequest
from app.services.prospect_service import prospect_service

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
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
)


# ---------------------------------------------------------------------------
# Middleware: request-ID injection
# ---------------------------------------------------------------------------
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attaches a UUID to each request and echoes it in the response header."""

    async def dispatch(self, request: Request, call_next: Callable):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------------------------
# Middleware: body-size limiter (prevent memory exhaustion)
# ---------------------------------------------------------------------------
class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects requests whose body exceeds MAX_BYTES."""

    MAX_BYTES: int = 64 * 1024  # 64 KB

    async def dispatch(self, request: Request, call_next: Callable):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_BYTES:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "success": False,
                    "error": "Request body too large.",
                    "max_bytes": self.MAX_BYTES,
                },
            )
        return await call_next(request)


app.add_middleware(BodySizeLimitMiddleware)


# ---------------------------------------------------------------------------
# Middleware: CORS
# Security note: allow_credentials=True is ONLY safe when allow_origins is
# NOT a wildcard.  We use an explicit list from configuration.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins(),
    allow_credentials=False,   # No cookie / session credentials needed
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
    # exc.errors() may contain non-serialisable ValueError objects inside the
    # 'ctx' key (e.g. when a field_validator raises ValueError).
    # Sanitise each error dict so the response is always valid JSON.
    safe_errors: list[dict] = []
    for err in exc.errors():
        safe_err: dict = {
            "type": err.get("type", ""),
            "loc": list(err.get("loc", [])),
            "msg": err.get("msg", ""),
        }
        # 'input' is safe (it's the user-supplied value).
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


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    # Log full details internally…
    logger.exception(
        "Unhandled error [request_id=%s] on %s", request_id, request.url.path
    )
    # …but never expose internal stack traces or raw exception messages to callers.
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
        "matches, scores relevance, and returns a consolidated JSON list of prospects."
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
        logger.exception(
            "Discovery pipeline error [request_id=%s]", request_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prospect discovery failed. Please try again later.",
        )


# ---------------------------------------------------------------------------
# Entry-point for direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        access_log=True,
    )
