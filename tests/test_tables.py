"""
Tests for Table Extraction endpoint /api/v1/ocr/table.
"""

from fastapi import status


def test_table_extraction_endpoint_schema(client, sample_table_path):
    """Test table extraction endpoint returns valid structured table schema."""
    with open(sample_table_path, "rb") as f:
        response = client.post(
            "/api/v1/ocr/table",
            files={"file": ("table_test.png", f, "image/png")},
            data={"preprocess": "false"}
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "tables" in data
    assert isinstance(data["tables"], list)

    if len(data["tables"]) > 0:
        table = data["tables"][0]
        assert "table_index" in table
        assert "headers" in table
        assert "rows" in table
        assert "row_count" in table
        assert "column_count" in table
        assert "columns" in table
        assert "data" in table

        # Column definitions check
        for col in table["columns"]:
            assert "key" in col
            assert "label" in col

        # Data rows check
        assert isinstance(table["data"], list)


def test_table_extraction_on_non_table_image(client, sample_image_path):
    """Test table extraction on a non-table image returns warnings rather than fabricating data."""
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/api/v1/ocr/table",
            files={"file": ("1.jpg", f, "image/jpeg")}
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert len(data["tables"]) == 0
    assert len(data["warnings"]) > 0


def test_strategy_a_bbox_clustering_direct():
    """Verify Strategy A correctly extracts table from OCR word coordinates without borders."""
    from app.services.table_extraction import table_extractor

    mock_words = [
        {"text": "வரிசை", "bbox": [50, 20, 60, 30], "line": 1, "center_x": 80, "center_y": 35},
        {"text": "பெயர்", "bbox": [200, 20, 70, 30], "line": 1, "center_x": 235, "center_y": 35},
        {"text": "தேதி", "bbox": [400, 20, 60, 30], "line": 1, "center_x": 430, "center_y": 35},
        {"text": "18092", "bbox": [45, 80, 70, 28], "line": 2, "center_x": 80, "center_y": 94},
        {"text": "ஆதிரன்", "bbox": [195, 80, 80, 28], "line": 2, "center_x": 235, "center_y": 94},
        {"text": "9.8.2023", "bbox": [390, 80, 85, 28], "line": 2, "center_x": 432, "center_y": 94},
        {"text": "18024", "bbox": [48, 140, 68, 28], "line": 3, "center_x": 82, "center_y": 154},
        {"text": "பைப்பு", "bbox": [190, 140, 60, 28], "line": 3, "center_x": 220, "center_y": 154},
        {"text": "பழப்பு", "bbox": [260, 140, 60, 28], "line": 3, "center_x": 290, "center_y": 154},
        {"text": "12.10.2023", "bbox": [395, 140, 90, 28], "line": 3, "center_x": 440, "center_y": 154},
    ]

    import numpy as np
    mock_img = np.zeros((300, 600, 3), dtype=np.uint8)
    tables, warnings = table_extractor.extract_tables(mock_img, words=mock_words, extract_text=False)

    assert len(tables) == 1
    t = tables[0]
    assert t.column_count == 3
    assert t.row_count == 2
    assert t.headers == ["வரிசை", "பெயர்", "தேதி"]
    assert t.rows[0] == ["18092", "ஆதிரன்", "9.8.2023"]
    assert t.rows[1] == ["18024", "பைப்பு பழப்பு", "12.10.2023"]
    assert t.detection_strategy == "bbox_clustering"
    assert "<table" in t.html
    assert "வரிசை" in t.html


def test_strategy_c_text_pattern_alignment():
    """Verify Strategy C fallback reconstructs tabular structure from text patterns and dates."""
    from app.services.table_extraction import table_extractor
    import numpy as np

    sample_ledger_text = (
        "வரிசை பதிவு\n"
        "18092 ஆதிரன் 9.8.2023\n"
        "18024 பைப்பு Ife LITTOUROST பழப்பு\n"
        "குடினின் 12.102023 டுப்பூர் 88709 75954\n"
    )

    mock_img = np.zeros((300, 600, 3), dtype=np.uint8)
    tables, warnings = table_extractor.extract_tables(
        mock_img,
        words=None,
        extracted_text=sample_ledger_text,
        extract_text=False
    )

    assert len(tables) == 1
    t = tables[0]
    assert t.column_count >= 2
    assert t.row_count >= 2
    assert t.detection_strategy == "text_alignment"
    assert t.is_uncertain is True
    assert "<table" in t.html


def test_html_table_generation_xss_safety():
    """Verify HTML table generation escapes special characters and prevents XSS."""
    from app.services.table_extraction import generate_html_table

    headers = ["<script>alert(1)</script>", "Tamil பெயர்"]
    rows = [["<b>Malicious</b>", "Normal தமிழ்"]]

    html_out = generate_html_table(headers, rows)
    assert "<script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
    assert "&lt;b&gt;Malicious&lt;/b&gt;" in html_out
    assert "Tamil பெயர்" in html_out
    assert "Normal தமிழ்" in html_out
    assert "<table" in html_out
    assert "<thead>" in html_out
    assert "<tbody>" in html_out


# =============================================================================
# Dynamic Schema Discovery Tests
# =============================================================================

def test_dynamic_eight_column_student_register():
    """Verify 8-column student register is dynamically discovered without forcing a 4-column schema."""
    import numpy as np
    from app.services.table_extraction import table_extractor

    mock_words = [
        # Line 1: Header Row (8 columns)
        {"text": "வடஎண்", "bbox": [50, 40, 50, 25], "line": 1, "center_x": 75, "center_y": 52},
        {"text": "மாணவர்", "bbox": [150, 40, 60, 25], "line": 1, "center_x": 180, "center_y": 52},
        {"text": "பெயர்", "bbox": [215, 40, 50, 25], "line": 1, "center_x": 240, "center_y": 52},
        {"text": "சேர்க்கை", "bbox": [300, 40, 60, 25], "line": 1, "center_x": 330, "center_y": 52},
        {"text": "எண்", "bbox": [365, 40, 40, 25], "line": 1, "center_x": 385, "center_y": 52},
        {"text": "பிறந்த", "bbox": [430, 40, 50, 25], "line": 1, "center_x": 455, "center_y": 52},
        {"text": "தேதி", "bbox": [485, 40, 40, 25], "line": 1, "center_x": 505, "center_y": 52},
        {"text": "பாலினம்", "bbox": [550, 40, 60, 25], "line": 1, "center_x": 580, "center_y": 52},
        {"text": "பெற்றோர்", "bbox": [640, 40, 65, 25], "line": 1, "center_x": 672, "center_y": 52},
        {"text": "பெயர்", "bbox": [710, 40, 45, 25], "line": 1, "center_x": 732, "center_y": 52},
        {"text": "கைபேசி", "bbox": [780, 40, 55, 25], "line": 1, "center_x": 807, "center_y": 52},
        {"text": "எண்", "bbox": [840, 40, 35, 25], "line": 1, "center_x": 857, "center_y": 52},
        {"text": "முகவரி", "bbox": [920, 40, 60, 25], "line": 1, "center_x": 950, "center_y": 52},

        # Line 2: Student Row 1
        {"text": "1.", "bbox": [60, 100, 25, 25], "line": 2, "center_x": 72, "center_y": 112},
        {"text": "செல்வன்", "bbox": [160, 100, 70, 25], "line": 2, "center_x": 195, "center_y": 112},
        {"text": "15A01", "bbox": [310, 100, 60, 25], "line": 2, "center_x": 340, "center_y": 112},
        {"text": "12/05/2009", "bbox": [440, 100, 80, 25], "line": 2, "center_x": 480, "center_y": 112},
        {"text": "ஆண்", "bbox": [560, 100, 40, 25], "line": 2, "center_x": 580, "center_y": 112},
        {"text": "கண்ணன்", "bbox": [650, 100, 65, 25], "line": 2, "center_x": 682, "center_y": 112},
        {"text": "9843123456", "bbox": [790, 100, 85, 25], "line": 2, "center_x": 832, "center_y": 112},
        {"text": "சென்னை", "bbox": [920, 100, 70, 25], "line": 2, "center_x": 955, "center_y": 112},

        # Line 3: Student Row 2
        {"text": "2.", "bbox": [60, 150, 25, 25], "line": 3, "center_x": 72, "center_y": 162},
        {"text": "மாலதி", "bbox": [160, 150, 65, 25], "line": 3, "center_x": 192, "center_y": 162},
        {"text": "15A02", "bbox": [310, 150, 60, 25], "line": 3, "center_x": 340, "center_y": 162},
        {"text": "03/08/2009", "bbox": [440, 150, 80, 25], "line": 3, "center_x": 480, "center_y": 162},
        {"text": "பெண்", "bbox": [560, 150, 40, 25], "line": 3, "center_x": 580, "center_y": 162},
        {"text": "முருகன்", "bbox": [650, 150, 65, 25], "line": 3, "center_x": 682, "center_y": 162},
        {"text": "9500678901", "bbox": [790, 150, 85, 25], "line": 3, "center_x": 832, "center_y": 162},
        {"text": "மதுரை", "bbox": [920, 150, 65, 25], "line": 3, "center_x": 952, "center_y": 162},
    ]

    mock_img = np.zeros((300, 1100, 3), dtype=np.uint8)
    tables, warnings = table_extractor.extract_tables(mock_img, words=mock_words, extract_text=False)

    assert len(tables) == 1
    t = tables[0]
    assert t.column_count == 8, f"Expected 8 columns, got {t.column_count}"
    assert t.row_count == 2
    assert "வ.எண்" in t.headers[0]
    assert "மாணவர் பெயர்" in t.headers[1]
    assert "சேர்க்கை எண்" in t.headers[2]
    assert "பிறந்த தேதி" in t.headers[3]
    assert "பாலினம்" in t.headers[4]
    assert "பெற்றோர் பெயர்" in t.headers[5]
    assert "கைபேசி எண்" in t.headers[6]
    assert "முகவரி" in t.headers[7]
    assert t.rows[0][0] == "1."
    assert t.rows[0][1] == "செல்வன்"
    assert t.rows[0][2] == "15A01"
    assert t.rows[0][3] == "12/05/2009"
    assert t.rows[0][6] == "9843123456"
    assert t.table_id == 1
    assert len(t.headers_detail) == 8


def test_farmer_register_different_schema():
    """Verify a farmer register with a completely different 4-column schema is dynamically discovered."""
    import numpy as np
    from app.services.table_extraction import table_extractor

    mock_words = [
        # Line 1: Header Row
        {"text": "விவசாயி", "bbox": [40, 30, 60, 25], "line": 1, "center_x": 70, "center_y": 42},
        {"text": "பெயர்", "bbox": [105, 30, 45, 25], "line": 1, "center_x": 127, "center_y": 42},
        {"text": "சர்வே", "bbox": [220, 30, 50, 25], "line": 1, "center_x": 245, "center_y": 42},
        {"text": "எண்", "bbox": [275, 30, 35, 25], "line": 1, "center_x": 292, "center_y": 42},
        {"text": "பயிர்", "bbox": [370, 30, 45, 25], "line": 1, "center_x": 392, "center_y": 42},
        {"text": "விவரம்", "bbox": [420, 30, 55, 25], "line": 1, "center_x": 447, "center_y": 42},
        {"text": "பரப்பளவு", "bbox": [530, 30, 70, 25], "line": 1, "center_x": 565, "center_y": 42},

        # Line 2: Farmer Data Row 1
        {"text": "முருகேசன்", "bbox": [45, 90, 80, 25], "line": 2, "center_x": 85, "center_y": 102},
        {"text": "142/2A", "bbox": [230, 90, 60, 25], "line": 2, "center_x": 260, "center_y": 102},
        {"text": "நெல்", "bbox": [380, 90, 45, 25], "line": 2, "center_x": 402, "center_y": 102},
        {"text": "2.5 ஏக்கர்", "bbox": [535, 90, 65, 25], "line": 2, "center_x": 567, "center_y": 102},

        # Line 3: Farmer Data Row 2
        {"text": "காளிமுத்து", "bbox": [45, 140, 85, 25], "line": 3, "center_x": 87, "center_y": 152},
        {"text": "98/1B", "bbox": [235, 140, 55, 25], "line": 3, "center_x": 262, "center_y": 152},
        {"text": "கரும்பு", "bbox": [380, 140, 55, 25], "line": 3, "center_x": 407, "center_y": 152},
        {"text": "1.0 ஏக்கர்", "bbox": [535, 140, 65, 25], "line": 3, "center_x": 567, "center_y": 152},
    ]

    mock_img = np.zeros((260, 700, 3), dtype=np.uint8)
    tables, warnings = table_extractor.extract_tables(mock_img, words=mock_words, extract_text=False)

    assert len(tables) == 1
    t = tables[0]
    assert t.column_count == 4
    assert t.row_count == 2
    assert "விவசாயி பெயர்" in t.headers[0]
    assert "சர்வே எண்" in t.headers[1]
    assert "பயிர் விவரம்" in t.headers[2]
    assert "பரப்பளவு" in t.headers[3]
    assert t.rows[0] == ["முருகேசன்", "142/2A", "நெல்", "2.5 ஏக்கர்"]
    assert t.rows[1] == ["காளிமுத்து", "98/1B", "கரும்பு", "1.0 ஏக்கர்"]


def test_milk_collection_table_schema():
    """Verify milk collection table with quantities and amounts."""
    import numpy as np
    from app.services.table_extraction import table_extractor

    mock_words = [
        # Line 1: Header Row (5 columns)
        {"text": "வ.எண்", "bbox": [40, 25, 45, 22], "line": 1, "center_x": 62, "center_y": 36},
        {"text": "உறுப்பினர்", "bbox": [130, 25, 75, 22], "line": 1, "center_x": 167, "center_y": 36},
        {"text": "பெயர்", "bbox": [210, 25, 45, 22], "line": 1, "center_x": 232, "center_y": 36},
        {"text": "பால்", "bbox": [300, 25, 40, 22], "line": 1, "center_x": 320, "center_y": 36},
        {"text": "அளவு", "bbox": [345, 25, 40, 22], "line": 1, "center_x": 365, "center_y": 36},
        {"text": "கொழுப்பு", "bbox": [440, 25, 60, 22], "line": 1, "center_x": 470, "center_y": 36},
        {"text": "தொகை", "bbox": [550, 25, 50, 22], "line": 1, "center_x": 575, "center_y": 36},

        # Line 2: Data Row 1
        {"text": "1", "bbox": [55, 75, 20, 22], "line": 2, "center_x": 65, "center_y": 86},
        {"text": "ராமசாமி", "bbox": [140, 75, 70, 22], "line": 2, "center_x": 175, "center_y": 86},
        {"text": "8.5 லி", "bbox": [310, 75, 55, 22], "line": 2, "center_x": 337, "center_y": 86},
        {"text": "4.2", "bbox": [450, 75, 30, 22], "line": 2, "center_x": 465, "center_y": 86},
        {"text": "₹280.50", "bbox": [545, 75, 65, 22], "line": 2, "center_x": 577, "center_y": 86},

        # Line 3: Data Row 2
        {"text": "2", "bbox": [55, 125, 20, 22], "line": 3, "center_x": 65, "center_y": 136},
        {"text": "மாரிமுத்து", "bbox": [140, 125, 75, 22], "line": 3, "center_x": 177, "center_y": 136},
        {"text": "12.0 லி", "bbox": [308, 125, 60, 22], "line": 3, "center_x": 338, "center_y": 136},
        {"text": "4.5", "bbox": [450, 125, 30, 22], "line": 3, "center_x": 465, "center_y": 136},
        {"text": "₹402.00", "bbox": [545, 125, 65, 22], "line": 3, "center_x": 577, "center_y": 136},
    ]

    mock_img = np.zeros((220, 680, 3), dtype=np.uint8)
    tables, warnings = table_extractor.extract_tables(mock_img, words=mock_words, extract_text=False)

    assert len(tables) == 1
    t = tables[0]
    assert t.column_count == 5
    assert "பால் அளவு" in t.headers[2]
    assert "தொகை" in t.headers[4]
    assert t.rows[0] == ["1", "ராமசாமி", "8.5 லி", "4.2", "₹280.50"]


def test_table_without_headers_level_c_fallback():
    """Verify tables without headers fall back to neutral Column N headers with semantic inference."""
    import numpy as np
    from app.services.table_extraction import table_extractor

    mock_words = [
        # Line 1: Data Row 1 (IDs, Dates, Phone)
        {"text": "18001", "bbox": [50, 40, 60, 25], "line": 1, "center_x": 80, "center_y": 52},
        {"text": "15/06/2021", "bbox": [200, 40, 85, 25], "line": 1, "center_x": 242, "center_y": 52},
        {"text": "9840112233", "bbox": [360, 40, 90, 25], "line": 1, "center_x": 405, "center_y": 52},

        # Line 2: Data Row 2
        {"text": "18002", "bbox": [50, 90, 60, 25], "line": 2, "center_x": 80, "center_y": 102},
        {"text": "22/07/2021", "bbox": [200, 90, 85, 25], "line": 2, "center_x": 242, "center_y": 102},
        {"text": "9840223344", "bbox": [360, 90, 90, 25], "line": 2, "center_x": 405, "center_y": 102},

        # Line 3: Data Row 3
        {"text": "18003", "bbox": [50, 140, 60, 25], "line": 3, "center_x": 80, "center_y": 152},
        {"text": "05/08/2021", "bbox": [200, 140, 85, 25], "line": 3, "center_x": 242, "center_y": 152},
        {"text": "9840334455", "bbox": [360, 140, 90, 25], "line": 3, "center_x": 405, "center_y": 152},
    ]

    mock_img = np.zeros((220, 520, 3), dtype=np.uint8)
    tables, warnings = table_extractor.extract_tables(mock_img, words=mock_words, extract_text=False)

    assert len(tables) == 1
    t = tables[0]
    assert t.column_count == 3
    assert t.row_count == 3
    # Headers inferred from data distribution
    assert t.headers_detail[0].semantic_type == "identifier"
    assert t.headers_detail[1].semantic_type == "date"
    assert t.headers_detail[2].semantic_type in ["phone", "phone_number"]


def test_empty_cells_preserved_no_left_shift():
    """Verify that empty cells are preserved as empty strings and do not shift neighboring values."""
    import numpy as np
    from app.services.table_extraction import table_extractor

    mock_words = [
        # Line 1: Header Row (3 columns: Code, Middle Name, Final Note)
        {"text": "குறியீடு", "bbox": [40, 30, 60, 25], "line": 1, "center_x": 70, "center_y": 42},
        {"text": "பெயர்", "bbox": [180, 30, 50, 25], "line": 1, "center_x": 205, "center_y": 42},
        {"text": "குறிப்பு", "bbox": [320, 30, 55, 25], "line": 1, "center_x": 347, "center_y": 42},

        # Line 2: Full Row
        {"text": "A101", "bbox": [45, 80, 50, 25], "line": 2, "center_x": 70, "center_y": 92},
        {"text": "கார்த்திக்", "bbox": [170, 80, 70, 25], "line": 2, "center_x": 205, "center_y": 92},
        {"text": "முடிக்கப்பட்டது", "bbox": [310, 80, 95, 25], "line": 2, "center_x": 357, "center_y": 92},

        # Line 3: Row with MISSING middle cell (Empty Name)
        {"text": "B202", "bbox": [45, 130, 50, 25], "line": 3, "center_x": 70, "center_y": 142},
        # Col 2 is empty!
        {"text": "நிலுவை", "bbox": [315, 130, 60, 25], "line": 3, "center_x": 345, "center_y": 142},
    ]

    mock_img = np.zeros((200, 480, 3), dtype=np.uint8)
    tables, warnings = table_extractor.extract_tables(mock_img, words=mock_words, extract_text=False)

    assert len(tables) == 1
    t = tables[0]
    assert t.column_count == 3
    assert t.row_count == 2
    # Verify second row: middle cell MUST be empty and note must NOT shift left to Col 2
    row2 = t.rows[1]
    assert row2[0] == "B202"
    assert row2[1] == ""  # Empty cell preserved!
    assert row2[2] == "நிலுவை"  # Not shifted left!


def test_tamil_english_mixed_table():
    """Verify table with mixed Tamil and English tokens."""
    import numpy as np
    from app.services.table_extraction import table_extractor

    mock_words = [
        {"text": "ID", "bbox": [40, 20, 30, 20], "line": 1, "center_x": 55, "center_y": 30},
        {"text": "Student", "bbox": [120, 20, 60, 20], "line": 1, "center_x": 150, "center_y": 30},
        {"text": "Name", "bbox": [185, 20, 45, 20], "line": 1, "center_x": 207, "center_y": 30},
        {"text": "Score", "bbox": [280, 20, 50, 20], "line": 1, "center_x": 305, "center_y": 30},

        {"text": "101", "bbox": [45, 60, 30, 20], "line": 2, "center_x": 60, "center_y": 70},
        {"text": "அரவிந்தன்", "bbox": [130, 60, 75, 20], "line": 2, "center_x": 167, "center_y": 70},
        {"text": "95.5", "bbox": [285, 60, 40, 20], "line": 2, "center_x": 305, "center_y": 70},

        {"text": "102", "bbox": [45, 100, 30, 20], "line": 3, "center_x": 60, "center_y": 110},
        {"text": "பிரியா", "bbox": [130, 100, 55, 20], "line": 3, "center_x": 157, "center_y": 110},
        {"text": "98.0", "bbox": [285, 100, 40, 20], "line": 3, "center_x": 305, "center_y": 110},
    ]

    mock_img = np.zeros((180, 400, 3), dtype=np.uint8)
    tables, warnings = table_extractor.extract_tables(mock_img, words=mock_words, extract_text=False)

    assert len(tables) == 1
    t = tables[0]
    assert t.column_count == 3
    assert "Student Name" in t.headers[1]
    assert t.rows[0] == ["101", "அரவிந்தன்", "95.5"]
    assert t.rows[1] == ["102", "பிரியா", "98.0"]


def test_pure_english_table_discovery():
    """Verify dynamic discovery on pure English invoices/tables."""
    import numpy as np
    from app.services.table_extraction import table_extractor

    mock_words = [
        {"text": "Item", "bbox": [30, 20, 40, 20], "line": 1, "center_x": 50, "center_y": 30},
        {"text": "Description", "bbox": [120, 20, 80, 20], "line": 1, "center_x": 160, "center_y": 30},
        {"text": "Qty", "bbox": [250, 20, 30, 20], "line": 1, "center_x": 265, "center_y": 30},
        {"text": "Rate", "bbox": [330, 20, 35, 20], "line": 1, "center_x": 347, "center_y": 30},
        {"text": "Total", "bbox": [410, 20, 40, 20], "line": 1, "center_x": 430, "center_y": 30},

        {"text": "1", "bbox": [35, 60, 20, 20], "line": 2, "center_x": 45, "center_y": 70},
        {"text": "Wireless Mouse", "bbox": [110, 60, 95, 20], "line": 2, "center_x": 157, "center_y": 70},
        {"text": "5", "bbox": [255, 60, 20, 20], "line": 2, "center_x": 265, "center_y": 70},
        {"text": "450.00", "bbox": [325, 60, 50, 20], "line": 2, "center_x": 350, "center_y": 70},
        {"text": "2250.00", "bbox": [405, 60, 55, 20], "line": 2, "center_x": 432, "center_y": 70},
    ]

    mock_img = np.zeros((160, 500, 3), dtype=np.uint8)
    tables, warnings = table_extractor.extract_tables(mock_img, words=mock_words, extract_text=False)

    assert len(tables) == 1
    t = tables[0]
    assert t.column_count == 5
    assert t.row_count == 1
    assert "Item" in t.headers[0]
    assert "Description" in t.headers[1]
    assert "Qty" in t.headers[2]
    assert "Rate" in t.headers[3]
    assert "Total" in t.headers[4]
    assert t.rows[0] == ["1", "Wireless Mouse", "5", "450.00", "2250.00"]
    assert "en" in t.detected_languages


def test_cattle_aadhaar_in_cattle_context():
    """Verify 12-digit number in cattle context is classified as cattle_aadhaar_id."""
    import numpy as np
    from app.services.table_extraction import table_extractor

    mock_words = [
        # Table with explicit cattle/pashu context
        {"text": "வ.எண்", "bbox": [30, 20, 40, 20], "line": 1, "center_x": 50, "center_y": 30},
        {"text": "கால்நடை", "bbox": [110, 20, 65, 20], "line": 1, "center_x": 142, "center_y": 30},
        {"text": "ஆதார்", "bbox": [180, 20, 45, 20], "line": 1, "center_x": 202, "center_y": 30},
        {"text": "பால் அளவு", "bbox": [280, 20, 70, 20], "line": 1, "center_x": 315, "center_y": 30},

        # Data Row
        {"text": "1", "bbox": [35, 60, 20, 20], "line": 2, "center_x": 45, "center_y": 70},
        {"text": "123456789012", "bbox": [115, 60, 110, 20], "line": 2, "center_x": 170, "center_y": 70},
        {"text": "15.5 லிட்டர்", "bbox": [280, 60, 80, 20], "line": 2, "center_x": 320, "center_y": 70},
    ]

    mock_img = np.zeros((150, 420, 3), dtype=np.uint8)
    tables, warnings = table_extractor.extract_tables(mock_img, words=mock_words, extract_text=False)

    assert len(tables) == 1
    t = tables[0]
    assert t.column_count == 3
    assert t.headers_detail[1].semantic_type == "cattle_aadhaar_id"
    assert t.rows[0][1] == "123456789012"  # Exact string representation preserved


def test_twelve_digit_identifier_outside_cattle_context():
    """Verify 12-digit number in non-cattle document is NOT classified as cattle_aadhaar_id."""
    from app.services.pattern_classifier import pattern_classifier

    # In general non-cattle document
    res = pattern_classifier.classify_single_value("987654321098", column_context="UID", document_context="Invoice Report")
    assert res.semantic_type != "cattle_aadhaar_id"
    assert res.semantic_type == "identifier"


def test_conflict_resolution_account_number_with_ten_digits():
    """Verify column named 'Account Number' containing 10-digit numbers retains explicit header with warning."""
    from app.services.pattern_classifier import pattern_classifier

    # Explicit header 'Account Number' with 10-digit values
    vals = ["9840112233", "9840223344", "9840334455"]
    res = pattern_classifier.analyze_column_distribution(
        column_values=vals,
        header_text="Account Number",
        preferred_language="en"
    )

    # Must preserve explicit header and record conflict warning
    assert res.display_text == "Account Number"
    assert res.semantic_type in ["account_number", "identifier", "explicit_text"]
    assert len(res.warnings) > 0
    assert "phone" in res.warnings[0].lower()


def test_preservation_of_leading_zeros():
    """Verify identifiers with leading zeros retain zeros as strings and are never truncated."""
    from app.services.pattern_classifier import pattern_classifier

    res1 = pattern_classifier.classify_single_value("012345678901", column_context="மாடு ஆதார்")
    assert res1.normalized_value == "012345678901"
    assert res1.normalized_value.startswith("0")

    res2 = pattern_classifier.classify_single_value("00452", column_context="Reg No")
    assert res2.original_text == "00452"


def test_unassigned_text_extraction():
    """Verify text outside table bounding boundaries is captured in unassigned_text."""
    import numpy as np
    from app.services.table_extraction import table_extractor

    mock_words = [
        # Title outside table at top
        {"text": "தலைமைச் செயலக அறிக்கை 2026", "bbox": [50, 10, 200, 25], "line": 1, "center_x": 150, "center_y": 22},

        # Table rows (well below title)
        {"text": "வ.எண்", "bbox": [50, 100, 50, 25], "line": 2, "center_x": 75, "center_y": 112},
        {"text": "விவரம்", "bbox": [150, 100, 60, 25], "line": 2, "center_x": 180, "center_y": 112},
        {"text": "1", "bbox": [55, 140, 20, 25], "line": 3, "center_x": 65, "center_y": 152},
        {"text": "பராமரிப்பு", "bbox": [140, 140, 80, 25], "line": 3, "center_x": 180, "center_y": 152},
    ]

    mock_img = np.zeros((220, 400, 3), dtype=np.uint8)
    tables, warnings = table_extractor.extract_tables(mock_img, words=mock_words, extract_text=False)

    assert len(tables) == 1
    t = tables[0]
    assert t.column_count == 2
    assert t.unassigned_text is not None
    assert "தலைமைச்" in t.unassigned_text

