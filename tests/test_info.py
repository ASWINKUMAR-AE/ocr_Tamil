"""
Tests for API Information & Capabilities endpoint.
"""

from fastapi import status


def test_api_info_endpoint(client):
    """Test /api/v1/info returns capabilities, supported formats, limits, and dynamic network metadata."""
    response = client.get("/api/v1/info")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["service"] == "Tamil OCR API"
    assert data["version"] == "0.1.4"
    assert data["ocr_engine"] == "ocr-tamil"
    assert "image/jpeg" in data["supported_image_formats"]
    assert "image/png" in data["supported_image_formats"]
    assert "image/webp" in data["supported_image_formats"]
    assert data["max_upload_size_mb"] >= 1
    assert data["max_image_dimension"] >= 512
    assert data["max_batch_size"] >= 1
    assert "tamil_text_recognition" in data["capabilities"]
    assert "table_structure_detection" in data["capabilities"]

    # Network metadata
    net = data["network"]
    assert "primary_lan_ip" in net
    assert "lan_url" in net
    assert "local_url" in net
    assert "docs_url" in net
    assert "interfaces" in net
    assert len(net["interfaces"]) > 0
