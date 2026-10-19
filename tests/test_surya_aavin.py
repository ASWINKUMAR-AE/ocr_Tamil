"""Tests for Surya OCR 0.1.2 integration, Aavin domain mapping intelligence,
and the review-first structured OCR endpoint.
"""
import io
import pytest
from PIL import Image, ImageDraw
from fastapi import status
from app.services.surya_engine import SuryaEngine
from app.schemas.ocr import TableResult, TableColumn
from app.services.aavin_mapping import (
    AavinTableMapper,
    is_valid_12digit_cattle_id,
    extract_cattle_ids_from_text,
    extract_cattle_ids_with_raw_from_text,
    normalize_indian_phone,
    parse_milk_quantity,
    parse_cow_count,
)
from app.services.table_extraction import table_extractor


def test_surya_engine_singleton_and_detection():
    """Verify Surya 0.1.2 engine initializes and detects lines on an image."""
    engine = SuryaEngine.get_instance()
    assert engine is not None

    # Create a synthetic image with horizontal and vertical lines simulating a register
    img = Image.new("RGB", (300, 200), color="white")
    draw = ImageDraw.Draw(img)
    # Draw horizontal table grid lines
    draw.line([(20, 40), (280, 40)], fill="black", width=2)
    draw.line([(20, 80), (280, 80)], fill="black", width=2)
    draw.line([(20, 120), (280, 120)], fill="black", width=2)
    # Draw vertical table column lines
    draw.line([(20, 40), (20, 120)], fill="black", width=2)
    draw.line([(100, 40), (100, 120)], fill="black", width=2)
    draw.line([(200, 40), (200, 120)], fill="black", width=2)
    draw.line([(280, 40), (280, 120)], fill="black", width=2)

    res = engine.detect_lines(img)
    assert isinstance(res, dict)
    assert "bboxes" in res
    assert "vertical_lines" in res
    assert "horizontal_lines" in res
    assert isinstance(res["bboxes"], list)
    assert isinstance(res["vertical_lines"], list)
    assert isinstance(res["horizontal_lines"], list)


def test_aavin_mapper_pashu_aadhaar_preservation():
    """Verify 12-digit cattle IDs are preserved as strings with leading zeros."""
    # Valid 12-digit ID with leading zeros
    valid_id = "000123456789"
    is_valid, digits = is_valid_12digit_cattle_id(valid_id)
    assert is_valid is True
    assert digits == "000123456789"
    assert isinstance(digits, str)

    # Text containing spaced or tagged cattle IDs
    spaced_id = "Tag: 000123456789 / 999888777666"
    extracted = extract_cattle_ids_from_text(spaced_id)
    assert len(extracted) == 2
    assert "000123456789" in extracted
    assert "999888777666" in extracted

    # Invalid length (e.g. 8 digits)
    invalid_id = "12345678"
    is_valid, digits = is_valid_12digit_cattle_id(invalid_id)
    assert is_valid is False


def test_aavin_mapper_phone_normalization():
    """Verify Indian mobile numbers are normalized to 10 digits."""
    is_valid, phone = normalize_indian_phone("+91 98401-23456")
    assert is_valid is True
    assert phone == "9840123456"

    is_valid, phone = normalize_indian_phone("09840123456")
    assert is_valid is True
    assert phone == "9840123456"

    # Invalid (starts with 1)
    is_valid, phone = normalize_indian_phone("1234567890")
    assert is_valid is False
    assert phone is None


def test_aavin_mapper_milk_quantity():
    """Verify milk quantities are parsed as floats without guessing zero for blanks."""
    # Normal decimal
    val = parse_milk_quantity("12.5")
    assert val == 12.5

    # Integer representation
    val = parse_milk_quantity("10")
    assert val == 10.0

    # Empty string should NOT be assumed to be 0
    val = parse_milk_quantity("")
    assert val is None

    # Unrealistic outlier (e.g. 3500 litres per cow single session)
    val = parse_milk_quantity("3500")
    assert val is None


def test_aavin_mapper_header_synonyms():
    """Verify Tamil and English header names are mapped to semantic fields."""
    # Tamil headers
    assert AavinTableMapper.match_header_semantic("விவசாயி பெயர்") == "farmer_name"
    assert AavinTableMapper.match_header_semantic("மாடு எண்") == "cattle_id"
    assert AavinTableMapper.match_header_semantic("பால் அளவு") == "milk_quantity"
    assert AavinTableMapper.match_header_semantic("கைபேசி எண்") == "phone"

    # English headers
    assert AavinTableMapper.match_header_semantic("Farmer Name") == "farmer_name"
    assert AavinTableMapper.match_header_semantic("Tag ID") == "cattle_id"
    assert AavinTableMapper.match_header_semantic("Litres") == "milk_quantity"
    assert AavinTableMapper.match_header_semantic("Mobile") == "phone"


def test_aavin_continuation_rows_and_multi_cattle():
    """Verify continuation rows group multiple cattle IDs under the correct farmer."""
    headers = ["விவசாயி பெயர்", "கைபேசி", "மாடு எண்", "பால் அளவு"]

    # Table simulation:
    # Row 0: Farmer Murugan with first cow
    # Row 1: Continuation row with second cow for Murugan (no farmer name repeated)
    # Row 2: Farmer Kannan with cow
    rows = [
        ["முருகன்", "9840123456", "000111222333", "8.5"],
        ["", "", "000444555666", "7.0"],
        ["கண்ணன்", "9840999888", "000777888999", "10.0"],
    ]

    mock_table = TableResult(
        table_index=0,
        headers=headers,
        rows=rows,
        columns=[
            TableColumn(key=f"col_{i}", label=h) for i, h in enumerate(headers)
        ],
        confidence=0.95
    )

    farmers, unassigned = AavinTableMapper.extract_candidate_records(mock_table)

    # There should be exactly 2 farmers (Murugan and Kannan), NOT 3
    assert len(farmers) == 2
    assert farmers[0].farmer_name == "முருகன்"
    assert farmers[0].phone == "9840123456"
    # Murugan should have both cattle IDs
    assert len(farmers[0].cattle_ids) == 2
    assert "000111222333" in farmers[0].cattle_ids
    assert "000444555666" in farmers[0].cattle_ids

    # Kannan should have 1 cattle ID
    assert len(farmers[1].cattle_ids) == 1
    assert "000777888999" in farmers[1].cattle_ids

    # Milk records check
    assert len(farmers[0].milk_records) == 2
    assert farmers[0].milk_records[0].litres == 8.5
    assert farmers[0].milk_records[1].litres == 7.0

    assert len(farmers[1].milk_records) == 1
    assert farmers[1].milk_records[0].litres == 10.0


def test_structured_ocr_endpoint_api(client, sample_image_path):
    """Test POST /api/v1/ocr/structured endpoint."""
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/api/v1/ocr/structured",
            files={"file": ("1.jpg", f, "image/jpeg")},
            data={"document_type": "aavin_register", "language_hint": "ta,en"}
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["success"] is True
    assert "request_id" in data
    assert data["api_version"] == "v1"
    assert data["engine"]["name"] == "surya-ocr"
    assert data["engine"]["version"] == "0.1.2"
    assert isinstance(data["tables"], list)
    assert isinstance(data["records"], list)
    assert isinstance(data["warnings"], list)


def test_table_endpoint_backward_compatibility(client, sample_image_path):
    """Verify POST /api/v1/ocr/table retains all legacy response fields."""
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/api/v1/ocr/table",
            files={"file": ("1.jpg", f, "image/jpeg")}
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    # Legacy fields must be present
    assert data["success"] is True
    assert "table_id" in data
    assert "headers" in data
    assert "headers_detail" in data
    assert "rows" in data
    assert "columns" in data
    assert "confidence" in data
    assert "detection_strategy" in data
    assert "metadata" in data
    assert data["metadata"]["ocr_engine"] == "surya-ocr"


def test_comma_separated_cattle_ids_extraction():
    """Verify comma-separated 12-digit cattle IDs in a cell are parsed into distinct tag IDs."""
    # 1. 3 comma-separated IDs with spaces inside 4-4-4 format
    text1 = "4006 7854 1362, 1567 9286 1041, 2965 4105 0723"
    ids1 = extract_cattle_ids_from_text(text1)
    assert len(ids1) == 3
    assert ids1 == ["400678541362", "156792861041", "296541050723"]

    # 2. 5 comma-separated continuous 12-digit IDs
    text2 = "100346652999, 100346652994, 143216789555, 122643227894, 467898443201"
    ids2 = extract_cattle_ids_from_text(text2)
    assert len(ids2) == 5
    assert ids2[0] == "100346652999"
    assert ids2[4] == "467898443201"

    # 3. Leading zeros preserved across all comma-separated IDs
    text3 = "000111222333, 000444555666, 000777888999"
    ids3 = extract_cattle_ids_from_text(text3)
    assert len(ids3) == 3
    assert ids3 == ["000111222333", "000444555666", "000777888999"]
    for cid in ids3:
        assert isinstance(cid, str)
        assert cid.startswith("000")
        assert len(cid) == 12

    # 4. Mixed delimiters: semicolon, newline, slash
    text4 = "100346652999; 100346652994\n143216789555 / 122643227894"
    ids4 = extract_cattle_ids_from_text(text4)
    assert len(ids4) == 4
    assert ids4 == ["100346652999", "100346652994", "143216789555", "122643227894"]

    # 5. Raw substring tracking
    pairs = extract_cattle_ids_with_raw_from_text("4006 7854 1362, 1567 9286 1041")
    assert len(pairs) == 2
    assert pairs[0][0] == "400678541362"
    assert "4006 7854 1362" in pairs[0][1]


def test_comma_separated_n_cows_for_n_ids_mapping():
    """Verify N comma-separated IDs in a cell map to exactly N individual cows with cow_index and synchronized cow_count."""
    headers = ["விவசாயி பெயர்", "கைபேசி எண்", "12 Digit Aadhar ID", "இனம்"]
    rows = [
        ["செந்தில் முருகன்", "9840123456", "4006 7854 1362, 1567 9286 1041, 2965 4105 0723", "HF, Jersey, சிந்தி"],
        ["கார்த்திக்", "9444123456", "100346652999", "நாட்டு மாடு"],
    ]

    mock_table = TableResult(
        table_index=0,
        headers=headers,
        rows=rows,
        columns=[TableColumn(key=f"col_{i}", label=h) for i, h in enumerate(headers)],
        confidence=0.96
    )

    farmers, unassigned = AavinTableMapper.extract_candidate_records(mock_table)

    assert len(farmers) == 2
    farmer1 = farmers[0]
    farmer2 = farmers[1]

    # Farmer 1 has 3 IDs in one cell -> exactly 3 cows!
    assert len(farmer1.cattle) == 3
    assert len(farmer1.cattle_ids) == 3
    assert farmer1.cow_count == 3

    # Check each individual cow
    cow1 = farmer1.cattle[0]
    cow2 = farmer1.cattle[1]
    cow3 = farmer1.cattle[2]

    assert cow1.cow_index == 1
    assert cow1.tag_id == "400678541362"
    assert cow1.breed == "HF"
    assert cow1.status == "candidate"

    assert cow2.cow_index == 2
    assert cow2.tag_id == "156792861041"
    assert cow2.breed == "Jersey"

    assert cow3.cow_index == 3
    assert cow3.tag_id == "296541050723"
    assert cow3.breed == "சிந்தி"

    # Farmer 2 has 1 ID -> exactly 1 cow
    assert len(farmer2.cattle) == 1
    assert farmer2.cow_count == 1
    assert farmer2.cattle[0].cow_index == 1
    assert farmer2.cattle[0].tag_id == "100346652999"
    assert farmer2.cattle[0].breed == "நாட்டு மாடு"


def test_continuation_row_with_comma_separated_ids():
    """Verify continuation rows with comma-separated IDs sequence cow_index 1..N seamlessly."""
    headers = ["விவசாயி பெயர்", "12 Digit Pashu Aadhaar"]
    rows = [
        # Farmer row with 2 IDs
        ["முத்து", "000111222333, 000444555666"],
        # Continuation row with 2 more IDs
        ["", "000777888999, 000999888777"],
    ]

    mock_table = TableResult(
        table_index=0,
        headers=headers,
        rows=rows,
        columns=[TableColumn(key=f"col_{i}", label=h) for i, h in enumerate(headers)],
        confidence=0.95
    )

    farmers, _ = AavinTableMapper.extract_candidate_records(mock_table)

    assert len(farmers) == 1
    muthu = farmers[0]

    # Total 4 cows across primary + continuation rows
    assert len(muthu.cattle) == 4
    assert muthu.cow_count == 4
    assert muthu.cattle_ids == ["000111222333", "000444555666", "000777888999", "000999888777"]

    # Verify 1-indexed sequence cow_index: 1, 2, 3, 4
    for idx, cow in enumerate(muthu.cattle):
        assert cow.cow_index == idx + 1
        assert cow.tag_id == muthu.cattle_ids[idx]


def test_expand_multi_cattle_columns_table_transformation():
    """
    Verify that cells containing N Aadhaar numbers are NOT left in a single cell,
    but expanded so each Aadhaar number is placed into its own distinct column.
    """
    raw_table = TableResult(
        table_index=0,
        headers=["S.No", "Farmer Name", "Phone No.", "Pashu Aadhar (12 Digit)", "Breed"],
        rows=[
            ["1", "Murugan", "9840123456", "4006 7854 1362, 1567 9286 1041, 2965 4105 0723", "HF, Jersey, Sindhi"],
            ["2", "Kannan", "9840654321", "100346652999", "Jersey"],
        ],
        columns=[
            TableColumn(key="s_no", label="S.No"),
            TableColumn(key="farmer_name", label="Farmer Name"),
            TableColumn(key="phone_no", label="Phone No."),
            TableColumn(key="pashu_aadhar", label="Pashu Aadhar (12 Digit)"),
            TableColumn(key="breed", label="Breed"),
        ],
        confidence=0.95
    )

    expanded = table_extractor.expand_multi_cattle_columns(raw_table)

    # 1. Headers must have 3 distinct Pashu Aadhar columns and 3 distinct Breed columns
    expected_headers = [
        "S.No", "Farmer Name", "Phone No.",
        "Pashu Aadhar 1", "Pashu Aadhar 2", "Pashu Aadhar 3",
        "Breed 1", "Breed 2", "Breed 3"
    ]
    assert expanded.headers == expected_headers
    assert expanded.column_count == len(expected_headers)

    # 2. Row 1: Murugan has 3 cows - each Aadhaar ID is in its own column!
    row1 = expanded.rows[0]
    assert row1[0] == "1"
    assert row1[1] == "Murugan"
    assert row1[2] == "9840123456"
    assert row1[3] == "400678541362"
    assert row1[4] == "156792861041"
    assert row1[5] == "296541050723"
    assert row1[6] == "HF"
    assert row1[7] == "Jersey"
    assert row1[8] == "Sindhi"

    # Verify no cell contains multiple IDs separated by comma
    for cell in row1:
        assert "," not in cell

    # 3. Row 2: Kannan has 1 cow - placed in Pashu Aadhar 1, columns 2 & 3 are empty
    row2 = expanded.rows[1]
    assert row2[0] == "2"
    assert row2[1] == "Kannan"
    assert row2[2] == "9840654321"
    assert row2[3] == "100346652999"
    assert row2[4] == ""
    assert row2[5] == ""
    assert row2[6] == "Jersey"
    assert row2[7] == ""
    assert row2[8] == ""

    # 4. Structured data key-value dictionary check
    assert expanded.data[0]["pashu_aadhar_1"] == "400678541362"
    assert expanded.data[0]["pashu_aadhar_2"] == "156792861041"
    assert expanded.data[0]["pashu_aadhar_3"] == "296541050723"

    # 5. HTML table check
    assert ">Pashu Aadhar 1</th>" in expanded.html
    assert ">Pashu Aadhar 2</th>" in expanded.html
    assert ">Pashu Aadhar 3</th>" in expanded.html
    assert ">400678541362</td>" in expanded.html


