#!/bin/bash
# Tamil OCR REST API - cURL Examples
# Upgraded with Surya OCR 0.1.2 & Aavin Smart Cattle Passport Register Intelligence
# Works with Local (http://127.0.0.1:8000) or LAN / Remote URLs

SERVER="http://127.0.0.1:8000"

echo "=== 1. Health Check ==="
curl -s -X GET "${SERVER}/health" | jq .

echo -e "\n=== 2. API Info & Capabilities (Surya OCR 0.1.2) ==="
curl -s -X GET "${SERVER}/api/v1/info" | jq .

echo -e "\n=== 3. Full OCR in JSON Format (Includes .json, .html, .text, .formats) ==="
curl -s -X POST "${SERVER}/api/v1/ocr" \
  -F "file=@sample_images/0.jpg" \
  -F "extract_tables=true" \
  -F "preprocess=true" \
  -F "format=json" | jq .

echo -e "\n=== 4. Direct HTML Output Format ==="
curl -s -X POST "${SERVER}/api/v1/ocr" \
  -F "file=@sample_images/0.jpg" \
  -F "format=html"

echo -e "\n\n=== 5. Direct Plain Text Output Format ==="
curl -s -X POST "${SERVER}/api/v1/ocr/text" \
  -F "file=@sample_images/0.jpg" \
  -F "format=text"

echo -e "\n\n=== 6. Table Extraction (Backward-Compatible Schema) ==="
curl -s -X POST "${SERVER}/api/v1/ocr/table" \
  -F "file=@sample_images/table_test.png" \
  -F "format=json" | jq .

echo -e "\n=== 7. Structured Aavin Register Intelligence (Surya 0.1.2 + Domain Entities) ==="
curl -s -X POST "${SERVER}/api/v1/ocr/structured" \
  -F "file=@sample_images/table_test.png" \
  -F "document_type=aavin_register" \
  -F "language_hint=ta,en" | jq .

echo -e "\n=== 8. Batch OCR (Multiple Files) ==="
curl -s -X POST "${SERVER}/api/v1/ocr/batch" \
  -F "files=@sample_images/0.jpg" \
  -F "files=@sample_images/1.jpg" | jq .

echo -e "\n=== 9. With API Key Authentication (When enabled) ==="
curl -s -X POST "${SERVER}/api/v1/ocr/structured" \
  -H "X-API-Key: secret-token-123" \
  -F "file=@sample_images/table_test.png" | jq .
