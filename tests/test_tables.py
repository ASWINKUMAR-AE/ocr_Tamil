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
