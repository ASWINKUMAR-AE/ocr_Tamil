"""
Tests for the Health Check endpoint.
"""

from fastapi import status
from unittest.mock import patch
from app.services.ocr_service import ocr_service


def test_health_check_success(client):
    """Test /health returns 200 OK with expected JSON structure when engine is healthy."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Tamil OCR API"
    assert data["ocr_engine"] == "ocr-tamil"
    assert data["version"] == "0.1.4"


def test_health_check_failure_when_unhealthy(client):
    """Test /health returns 503 Service Unavailable when engine initialization has failed."""
    with patch.object(ocr_service, "is_healthy", return_value=False):
        with patch.object(ocr_service, "_init_error", "Mock model failure"):
            response = client.get("/health")
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            data = response.json()
            assert data["status"] == "error"
            assert "Mock model failure" in data["error"]
            assert data["success"] is False
