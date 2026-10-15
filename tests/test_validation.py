"""
Tests for input validation, corruption handling, and size constraints.
"""

from fastapi import status
from unittest.mock import patch
from app.config import settings


def test_invalid_file_type_rejected(client):
    """Test non-image file uploads (e.g. plain text or python files) are rejected with 400."""
    fake_txt = b"Hello, this is not an image file at all."
    response = client.post(
        "/api/v1/ocr",
        files={"file": ("document.txt", fake_txt, "text/plain")}
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["success"] is False
    assert "error" in data


def test_corrupted_image_rejected(client):
    """Test image with valid extension but corrupted payload is rejected with 400."""
    corrupted_data = b"\x89PNG\r\n\x1a\nCORRUPTED_BYTES_HERE_NOT_A_VALID_IMAGE"
    response = client.post(
        "/api/v1/ocr",
        files={"file": ("corrupted.png", corrupted_data, "image/png")}
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["success"] is False
    assert "error" in data


def test_oversized_image_rejected(client):
    """Test files exceeding maximum upload limit are rejected with 413 Payload Too Large."""
    # Temporarily set max size to 0.001 MB (~1KB) to test without creating gigabytes
    with patch.object(settings, "MAX_UPLOAD_SIZE_MB", 0.001):
        fake_large_jpeg = b"\xff\xd8\xff" + b"A" * 5000
        response = client.post(
            "/api/v1/ocr",
            files={"file": ("large.jpg", fake_large_jpeg, "image/jpeg")}
        )

        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        data = response.json()
        assert data["success"] is False
        assert "exceeds" in data["error"]
