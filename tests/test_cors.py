"""
Tests for CORS Headers and Configuration.
"""

from fastapi import status


def test_cors_headers_present(client):
    """Test CORS preflight and response headers."""
    # Preflight OPTIONS request
    response = client.options(
        "/api/v1/ocr",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type, X-API-Key"
        }
    )

    assert response.status_code == status.HTTP_200_OK
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] in ("*", "http://localhost:3000")
