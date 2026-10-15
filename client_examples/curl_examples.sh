#!/bin/bash
# Tamil OCR REST API - cURL Examples
# Replace 10.110.44.105 with your server's actual LAN IP address.

SERVER="http://10.110.44.105:8000"

echo "=== 1. Health Check ==="
curl -s -X GET "${SERVER}/health" | jq .

echo -e "\n=== 2. API Info & Capabilities ==="
curl -s -X GET "${SERVER}/api/v1/info" | jq .

echo -e "\n=== 3. Full OCR (Text + Tables) ==="
curl -s -X POST "${SERVER}/api/v1/ocr" \
  -F "file=@sample_images/0.jpg" \
  -F "extract_tables=true" \
  -F "preprocess=true" | jq .

echo -e "\n=== 4. Table Extraction ==="
curl -s -X POST "${SERVER}/api/v1/ocr/table" \
  -F "file=@sample_images/table_test.png" | jq .

echo -e "\n=== 5. Plain Text Only ==="
curl -s -X POST "${SERVER}/api/v1/ocr/text" \
  -F "file=@sample_images/1.jpg" \
  -F "preprocess=true" | jq .

echo -e "\n=== 6. Batch OCR (Multiple Files) ==="
curl -s -X POST "${SERVER}/api/v1/ocr/batch" \
  -F "files=@sample_images/0.jpg" \
  -F "files=@sample_images/1.jpg" | jq .

echo -e "\n=== 7. With API Key Authentication (When enabled) ==="
curl -s -X POST "${SERVER}/api/v1/ocr" \
  -H "X-API-Key: secret-token-123" \
  -F "file=@sample_images/0.jpg" | jq .
