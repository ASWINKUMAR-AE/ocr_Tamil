"""
Tests for main OCR endpoints: /api/v1/ocr and /api/v1/ocr/text.
"""

import os
from fastapi import status


def test_valid_tamil_image_ocr(client, sample_image_path):
    """Test full OCR on a real Tamil image file."""
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/api/v1/ocr",
            files={"file": ("1.jpg", f, "image/jpeg")},
            data={"extract_tables": "true", "preprocess": "true"}
        )

    assert response.status_code == status.HTTP_200_OK
    assert "X-Request-ID" in response.headers

    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "1.jpg"
    assert data["content_type"] == "image/jpeg"
    assert isinstance(data["request_id"], str)
    assert len(data["request_id"]) > 0

    # Verify extracted text contains recognized Tamil text
    assert isinstance(data["extracted_text"], str)
    assert len(data["extracted_text"]) > 0
    # In sample_images/1.jpg, "நெடுஞ்சாலைத்" is expected
    assert "நெடுஞ்சாலைத்" in data["extracted_text"]

    # Verify metadata structure
    meta = data["metadata"]
    assert meta["ocr_engine"] == "ocr-tamil"
    assert meta["preprocessing_applied"] is True
    assert meta["processing_time_ms"] > 0
    assert "image_dimensions" in meta

    assert isinstance(data["tables"], list)
    assert isinstance(data["warnings"], list)


def test_plain_text_only_endpoint(client, sample_image_path):
    """Test /api/v1/ocr/text returns only text and metadata."""
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/api/v1/ocr/text",
            files={"file": ("1.jpg", f, "image/jpeg")},
            data={"preprocess": "false"}
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "நெடுஞ்சாலைத்" in data["extracted_text"]
    assert "tables" not in data  # Text only endpoint does not have tables
    assert data["metadata"]["preprocessing_applied"] is False
