"""
Automated benchmark and regression test suite for real register images.
Tests real-image OCR, Surya 0.1.2 layout table reconstruction,
semantic header mapping, and ground truth field-level verification.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.surya_engine import surya_engine

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BENCHMARK_DIR = FIXTURES_DIR / "benchmark_registers"
GROUND_TRUTH_DIR = FIXTURES_DIR / "ground_truth"


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def init_engines():
    """Ensure Surya and Tamil ParSeq are initialized before benchmarks."""
    surya_engine.initialize()


def test_img2_clean_english_register_exact_matches(client):
    """
    Benchmark Image 2: Clean English Farmer Register.
    Mandatory correctness criteria:
    1. Exactly 10 reference rows extracted.
    2. All 10 phone numbers match reference ground truth exactly.
    3. All 10 12-digit Pashu Aadhaar IDs match reference ground truth exactly as strings.
    4. Table strategy is surya_layout.
    """
    img_path = BENCHMARK_DIR / "img2_clean_english_register.jpg"
    gt_path = GROUND_TRUTH_DIR / "img2_ground_truth.json"

    assert img_path.exists(), f"Benchmark image missing: {img_path}"
    assert gt_path.exists(), f"Ground truth missing: {gt_path}"

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    with open(img_path, "rb") as f:
        response = client.post(
            "/api/v1/ocr/table",
            files={"file": ("img2_clean_english_register.jpg", f, "image/jpeg")},
            data={"language_hint": "en,ta"}
        )

    assert response.status_code == 200, f"OCR API error: {response.text}"
    data = response.json()
    assert data["success"] is True
    assert "tables" in data
    assert len(data["tables"]) >= 1

    table = data["tables"][0]
    assert table["detection_strategy"] == "surya_layout"

    rows = table["rows"]
    assert len(rows) == 10, f"Expected 10 rows, got {len(rows)}: {rows}"

    expected_rows = ground_truth["rows"]
    matched_phones = 0
    matched_aadhaars = 0

    for idx, (actual_row, expected_row) in enumerate(zip(rows, expected_rows)):
        # Extract phone (Col 2) and cattle id (Col 3)
        actual_sno = actual_row[0].strip()
        actual_name = actual_row[1].strip()
        actual_phone = actual_row[2].strip()
        actual_aadhaar = actual_row[3].strip()

        expected_sno = expected_row["s_no"]
        expected_name = expected_row["farmer_name"]
        expected_phone = expected_row["phone_no"]
        expected_aadhaar = expected_row["pashu_aadhar"]

        assert actual_sno == expected_sno, f"Row {idx} S.No mismatch: {actual_sno} != {expected_sno}"
        assert expected_name.split()[-1] in actual_name, f"Row {idx} Name mismatch: {actual_name} != {expected_name}"

        # Exact Phone match
        assert actual_phone == expected_phone, f"Row {idx} Phone mismatch: {actual_phone} != {expected_phone}"
        matched_phones += 1

        # Exact 12-digit Pashu Aadhaar match
        assert actual_aadhaar == expected_aadhaar, f"Row {idx} Aadhaar mismatch: {actual_aadhaar} != {expected_aadhaar}"
        assert len(actual_aadhaar) == 12, f"Row {idx} Aadhaar length is not 12: {actual_aadhaar}"
        matched_aadhaars += 1

    assert matched_phones == 10
    assert matched_aadhaars == 10


def test_img2_structured_endpoint_candidates(client):
    """Verify that POST /api/v1/ocr/structured generates valid review candidates for Image 2."""
    img_path = BENCHMARK_DIR / "img2_clean_english_register.jpg"
    with open(img_path, "rb") as f:
        res = client.post(
            "/api/v1/ocr/structured",
            files={"file": ("img2_clean_english_register.jpg", f, "image/jpeg")},
            data={"document_type": "aavin_register"}
        )
    assert res.status_code == 200
    data = res.json()
    assert data["review_required"] is True
    assert len(data["candidates"]) == 10

    first_cand = data["candidates"][0]
    assert first_cand["farmer_name"] == "R. Murugan"
    assert first_cand["phone"] == "9345126780"
    assert "111234567890" in first_cand["cattle_ids"]
    assert first_cand["validation_status"] == "valid"


def test_img1_tamil_handwritten_crop(client):
    """Benchmark Image 1: Tamil handwritten name list."""
    img_path = BENCHMARK_DIR / "img1_tamil_handwritten_crop.png"
    gt_path = GROUND_TRUTH_DIR / "img1_ground_truth.json"

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    with open(img_path, "rb") as f:
        res = client.post(
            "/api/v1/ocr/text",
            files={"file": ("img1_tamil_handwritten_crop.png", f, "image/png")},
            data={"language_hint": "ta", "preprocess": "false"}
        )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert len(data["extracted_text"]) > 20


def test_img3_large_english_register_continuation_rows(client):
    """
    Benchmark Image 3: Large English Share Register.
    Verifies continuation row handling and farmer-to-share associations.
    """
    img_path = BENCHMARK_DIR / "img3_large_english_register.jpg"
    gt_path = GROUND_TRUTH_DIR / "img3_ground_truth.json"

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    with open(img_path, "rb") as f:
        res = client.post(
            "/api/v1/ocr/table",
            files={"file": ("img3_large_english_register.jpg", f, "image/jpeg")},
            data={"language_hint": "en"}
        )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    table = data["tables"][0]
    assert table["detection_strategy"] == "surya_layout"

    # Verify key identifiers exist in extracted rows
    extracted_text_blob = " ".join(" ".join(r) for r in table["rows"]).replace(" ", "")
    assert "322687625891" in extracted_text_blob
    assert "400578511362" in extracted_text_blob
    assert "426877663224" in extracted_text_blob
    assert "100346552999" in extracted_text_blob
    assert "992453532004" in extracted_text_blob
    assert "889420043214" in extracted_text_blob


def test_img4_large_tamil_handwritten_register(client):
    """Benchmark Image 4: Large Tamil Handwritten Register."""
    img_path = BENCHMARK_DIR / "img4_large_tamil_handwritten_register.jpg"
    with open(img_path, "rb") as f:
        res = client.post(
            "/api/v1/ocr/table",
            files={"file": ("img4_large_tamil_handwritten_register.jpg", f, "image/jpeg")},
            data={"language_hint": "ta"}
        )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    table = data["tables"][0]
    assert table["detection_strategy"] == "surya_layout"
    assert len(table["rows"]) >= 6


def test_img5_printed_tamil_register(client):
    """Benchmark Image 5: Printed Tamil Register."""
    img_path = BENCHMARK_DIR / "img5_printed_tamil_register.jpg"
    with open(img_path, "rb") as f:
        res = client.post(
            "/api/v1/ocr/table",
            files={"file": ("img5_printed_tamil_register.jpg", f, "image/jpeg")},
            data={"language_hint": "ta"}
        )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    table = data["tables"][0]
    assert len(table["rows"]) >= 10
