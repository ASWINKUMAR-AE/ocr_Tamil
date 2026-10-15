"""
Tests for API Key Authentication and Security Controls.
"""

from fastapi import status
from unittest.mock import patch
from app.config import settings


def test_auth_disabled_by_default(client):
    """When API_KEY is empty/unset, endpoints are accessible without authentication headers."""
    response = client.get("/api/v1/info")
    assert response.status_code == status.HTTP_200_OK


def test_auth_enforced_when_api_key_configured(client):
    """When API_KEY is set in configuration, missing or invalid keys return 401 Unauthorized."""
    test_key = "secret-test-token-789"
    with patch.object(settings, "API_KEY", test_key):
        # 1. Missing API key -> 401
        res1 = client.get("/api/v1/info")
        assert res1.status_code == status.HTTP_401_UNAUTHORIZED

        # 2. Invalid API key -> 401
        res2 = client.get("/api/v1/info", headers={"X-API-Key": "wrong-key"})
        assert res2.status_code == status.HTTP_401_UNAUTHORIZED

        # 3. Valid X-API-Key header -> 200
        res3 = client.get("/api/v1/info", headers={"X-API-Key": test_key})
        assert res3.status_code == status.HTTP_200_OK

        # 4. Valid Authorization: Bearer token -> 200
        res4 = client.get("/api/v1/info", headers={"Authorization": f"Bearer {test_key}"})
        assert res4.status_code == status.HTTP_200_OK
