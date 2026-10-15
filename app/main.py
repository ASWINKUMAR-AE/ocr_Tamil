"""
Tamil OCR REST API Server.
Production-ready FastAPI application for Tamil & mixed-language text recognition,
table detection, and cross-platform client integration.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings
from app.services.ocr_service import ocr_service
from app.utils.network import print_startup_banner
from app.utils.security import generate_request_id

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tamil_ocr.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown procedures."""
    logger.info("Starting Tamil OCR API Server...")

    # Display dynamic network addresses and accessibility guide
    print_startup_banner(settings.HOST, settings.PORT)

    # Initialize OCR engine models
    try:
        ocr_service.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize OCR engine at startup: {e}")

    yield

    logger.info("Shutting down Tamil OCR API Server...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-ready Tamil Optical Character Recognition (OCR) REST API. "
        "Extracts Tamil and mixed-language text from natural scenes and documents, "
        "detects tabular structures, and returns structured JSON responses for mobile, "
        "web, and desktop applications."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ============================================================================
# CORS Middleware Configuration
# ============================================================================
allow_creds = False
if "*" not in settings.cors_origins_list:
    allow_creds = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=allow_creds,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Standardize HTTP error responses."""
    req_id = getattr(request.state, "request_id", generate_request_id())
    detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
    detail["request_id"] = detail.get("request_id", req_id)
    detail["success"] = False

    return JSONResponse(
        status_code=exc.status_code,
        content=detail,
        headers=getattr(exc, "headers", None)
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Standardize 422 validation errors."""
    req_id = generate_request_id()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "request_id": req_id,
            "error": "Validation error in request parameters",
            "details": exc.errors(),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all internal server error handler with sanitized response."""
    req_id = generate_request_id()
    logger.error(f"Unhandled exception [req_id: {req_id}]: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "request_id": req_id,
            "error": "An unexpected internal server error occurred.",
            "error_code": "INTERNAL_SERVER_ERROR"
        }
    )


# ============================================================================
# Include API Routes
# ============================================================================
app.include_router(router)
