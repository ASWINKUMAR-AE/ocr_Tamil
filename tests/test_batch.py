"""
Tests for Batch OCR endpoint /api/v1/ocr/batch.
"""

from fastapi import status
from unittest.mock import patch
from app.config import settings


def test_batch_ocr_multiple_files(client, sample_image_path):
    """Test batch OCR endpoint with 2 valid images."""
    with open(sample_image_path, "rb") as f1, open(sample_image_path, "rb") as f2:
        response = client.post(
            "/api/v1/ocr/batch",
            files=[
                ("files", ("image1.jpg", f1.read(), "image/jpeg")),
                ("files", ("image2.jpg", f2.read(), "image/jpeg")),
            ],
            data={"preprocess": "false", "extract_tables": "false"}
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert data["total_images"] == 2
    assert data["successful_images"] == 2
    assert data["failed_images"] == 0
    assert len(data["results"]) == 2

    for res in data["results"]:
        assert res["success"] is True
        assert "நெடுஞ்சாலைத்" in res["extracted_text"]


def test_batch_ocr_exceeds_max_limit(client, sample_image_path):
    """Test batch request exceeding MAX_BATCH_SIZE is rejected with 400."""
    with patch.object(settings, "MAX_BATCH_SIZE", 2):
        files = [
            ("files", (f"img_{i}.jpg", b"\xff\xd8\xffFAKE", "image/jpeg"))
            for i in range(3)
        ]
        response = client.post("/api/v1/ocr/batch", files=files)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "exceeds" in data["error"]


def test_batch_ocr_partial_failure_handling(client, sample_image_path):
    """Test batch request with one valid and one corrupted image handles each independently."""
    with open(sample_image_path, "rb") as f:
        valid_bytes = f.read()

    corrupt_bytes = b"NOT_AN_IMAGE_CONTENT"

    response = client.post(
        "/api/v1/ocr/batch",
        files=[
            ("files", ("valid.jpg", valid_bytes, "image/jpeg")),
            ("files", ("invalid.png", corrupt_bytes, "image/png")),
        ],
        data={"preprocess": "false"}
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total_images"] == 2
    assert data["successful_images"] == 1
    assert data["failed_images"] == 1

    results = data["results"]
    assert results[0]["success"] is True
    assert results[1]["success"] is False
    assert results[1]["error"] is not None
