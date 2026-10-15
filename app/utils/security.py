"""
Security and validation utilities for Tamil OCR API.
Includes API Key authentication, payload size checks, MIME magic byte inspection,
and request tracking identifiers.
"""

import uuid
import secrets
from typing import Optional, Tuple
from fastapi import Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings

api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
http_bearer_scheme = HTTPBearer(auto_error=False)

ALLOWED_MIME_TYPES = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/jpg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/webp": [b"RIFF"],
}


def generate_request_id() -> str:
    """Generate a clean UUIDv4 for request tracking."""
    return str(uuid.uuid4())


async def verify_api_key(
    x_api_key: Optional[str] = Security(api_key_header_scheme),
    bearer_auth: Optional[HTTPAuthorizationCredentials] = Security(http_bearer_scheme),
) -> Optional[str]:
    """
    FastAPI security dependency for optional/enforced API key verification.
    If API_KEY is not set in configuration, authentication is bypassed.
    If set, accepts API Key via either `X-API-Key` header or `Authorization: Bearer <token>`.
    """
    if not settings.is_auth_enabled:
        return None

    provided_key = None
    if x_api_key:
        provided_key = x_api_key.strip()
    elif bearer_auth and bearer_auth.credentials:
        provided_key = bearer_auth.credentials.strip()

    if not provided_key or not secrets.compare_digest(provided_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Missing or invalid API Key.",
            headers={"WWW-Authenticate": "ApiKey or Bearer"},
        )

    return provided_key


def validate_image_bytes(data: bytes, filename: str, content_type: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate raw uploaded bytes for safety, non-empty size, maximum limits,
    and actual image magic byte signatures.
    """
    if not data or len(data) == 0:
        return False, "Uploaded file is empty (0 bytes)."

    if len(data) > settings.max_upload_size_bytes:
        return False, (
            f"File size {len(data) / (1024 * 1024):.2f}MB exceeds the maximum allowed limit "
            f"of {settings.MAX_UPLOAD_SIZE_MB}MB."
        )

    # Magic byte signature check
    is_jpeg = data.startswith(b"\xff\xd8\xff")
    is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = len(data) > 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"

    if not (is_jpeg or is_png or is_webp):
        return False, (
            f"Unsupported or invalid image format for file '{filename}'. "
            "Supported formats are JPEG, PNG, and WebP."
        )

    return True, None
