"""
Pytest configuration and shared fixtures for Tamil OCR API tests.
"""

import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.ocr_service import ocr_service


@pytest.fixture(scope="session", autouse=True)
def initialize_engine():
    """Ensure the OCR engine is initialized once for the entire test session."""
    ocr_service.initialize()
    yield


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_path():
    """Path to a verified real Tamil test image."""
    path = os.path.join("sample_images", "1.jpg")
    assert os.path.exists(path), f"Test image missing: {path}"
    return path


@pytest.fixture
def sample_table_path():
    """Path to a verified table test image."""
    path = os.path.join("sample_images", "table_test.png")
    assert os.path.exists(path), f"Table image missing: {path}"
    return path
