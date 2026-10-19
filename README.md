# Tamil OCR REST API & Testing Web Application 🚀

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ocr-tamil](https://img.shields.io/badge/OCR%20Engine-ocr--tamil%20v0.1.4-blue)](https://pypi.org/project/ocr-tamil/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, production-ready **Tamil Optical Character Recognition (OCR) REST API Server** paired with a standalone **HTML Testing Web Application**.

Built using **FastAPI**, **Uvicorn**, **`ocr-tamil==0.1.4`** (PARSeq + CRAFT), **OpenCV**, and **Pillow**.

---

## 📌 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Project Structure](#-project-structure)
- [Key Features](#-key-features)
- [Quick Start Guide](#-quick-start-guide)
  - [1. Backend Setup & Run](#1-backend-setup--run)
  - [2. Frontend Web Tester Run](#2-frontend-web-tester-run)
- [Dynamic LAN IP & Network Accessibility](#-dynamic-lan-ip--network-accessibility)
- [API Endpoints Reference](#-api-endpoints-reference)
- [HTML Testing Web Application](#-html-testing-web-application)
- [Client Integration Examples](#-client-integration-examples)
- [Automated Testing](#-automated-testing)
- [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🌟 Overview & Architecture

This repository provides an end-to-end OCR solution designed specifically for the Tamil script (including mixed Tamil-English text, printed signboards, documents, and structured tables).

```text
 ┌─────────────────────────────────────────────────────────────┐
 │                     Client Applications                     │
 │  (React Native / Expo • Web Browser • Desktop Python • cURL)│
 └──────────────────────────────┬──────────────────────────────┘
                                │ HTTP POST (Multipart Form-Data)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                  FastAPI REST API Server                    │
 │               (Host: 0.0.0.0, Default Port: 8000)           │
 └──────────────┬───────────────────────────────┬──────────────┘
                │                               │
                ▼                               ▼
 ┌──────────────────────────────┐ ┌────────────────────────────┐
 │     Image Preprocessor       │ │    Table Grid Detector     │
 │ • EXIF Orientation Transpose │ │ • OpenCV Morphological     │
 │ • Safe Resizing & Deskew     │ │ • Row/Column Clustering    │
 │ • Bilateral Noise Filtering  │ │ • Cell Boundary Extraction │
 │ • CLAHE Contrast Enhancement │ │ • Inner Margin Cropping    │
 └──────────────┬───────────────┘ └─────────────┬──────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                     ocr-tamil Engine                        │
 │ • CRAFT Text Detection (Multi-lingual scene & line boxes)   │
 │ • PARSeq Tamil Recognition (Sequence-to-sequence decoder)   │
 │ • PARSeq English Recognition (Fallback for mixed Latin text)│
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                   Structured JSON Output                    │
 │   • Preserved Line Order Text • 2D Table Arrays (CSV/TSV)   │
 │   • Key-Value Row Objects • Processing Metadata             │
 └─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
ocr_Tamil/
├── app/                                 # Backend FastAPI Application
│   ├── __init__.py
│   ├── main.py                         # FastAPI setup, lifespan, CORS, and error handlers
│   ├── config.py                       # Pydantic BaseSettings loading from .env
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                   # REST endpoints (/health, /info, /ocr, /table, /text, /batch)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── ocr.py                      # Pydantic request and response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ocr_service.py              # OCR service wrapper for ocr-tamil, CRAFT & PARSeq
│   │   ├── image_preprocessing.py      # Deskewing, bilateral filtering, orientation fix
│   │   └── table_extraction.py         # OpenCV morphology table & cell extraction
│   └── utils/
│       ├── __init__.py
│       ├── network.py                  # Dynamic LAN IPv4 detection and startup banner
│       └── security.py                 # File validation, magic bytes, and API key auth
│
├── frontend/                            # Standalone Web Testing Application
│   ├── tamil-ocr-tester.html            # Complete single-page test dashboard
│   └── README.md                        # Frontend setup and usage guide
│
├── client_examples/                     # Ready-to-use cross-platform client code
│   ├── react_native_expo.js            # React Native / Expo with ImagePicker
│   ├── web_fetch.html                  # Browser JavaScript client
│   ├── desktop_python.py               # Python desktop client with requests
│   ├── node_client.js                  # Node.js 18+ client using native fetch
│   └── curl_examples.sh                # Ready-to-run cURL command scripts
│
├── tests/                               # Comprehensive Automated Test Suite
│   ├── conftest.py                     # Pytest fixtures and model initialization
│   ├── test_health.py                  # Health check (200 OK and 503 error tests)
│   ├── test_info.py                    # Capabilities and network metadata tests
│   ├── test_ocr.py                     # Real Tamil image OCR tests
│   ├── test_tables.py                  # Table extraction schema tests
│   ├── test_batch.py                   # Multi-image batch and limit tests
│   ├── test_validation.py              # Corrupted, non-image, and size limit tests
│   ├── test_security.py                # Optional API-key authentication tests
│   ├── test_cors.py                    # CORS preflight and allowed origins tests
│   └── test_network.py                 # LAN IPv4 detection unit tests
│
├── sample_images/                       # Real Tamil and table test samples
│   ├── 0.jpg                           # "கொடைக்கானல் Kodaikanal"
│   ├── 1.jpg                           # "நெடுஞ்சாலைத்"
│   ├── 4.jpg                           # "வாழ்கவளமுடன்"
│   └── table_test.png                  # Multi-row, multi-column sample table
│
├── .env.example                        # Environment variables template
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Verified Python dependencies
├── README.md                           # Documentation and deployment guide
└── run.py                              # Server launcher with CLI arguments
```

---

## ⚡ Key Features

1. **Exact Library Integration:** Built with `ocr-tamil==0.1.4`, using verified class signatures and model loaders.
2. **Dynamic LAN IP Detection:** Automatically inspects the machine's active network interfaces and prints the actual LAN IPv4 address on launch.
3. **Structured Table Detection:** Detects cell boundaries with OpenCV morphological kernels and clusters them into rows and columns. Returns both 2D row arrays and key-value JSON objects.
4. **Line-Preserving OCR:** Respects natural reading order and inserts line breaks (`\n`) across text lines.
5. **Safe Preprocessing:** EXIF orientation transpose, deskewing (-25° to +25°), bilateral noise reduction, and CLAHE contrast enhancement without damaging Tamil vowel signs or pulli dots.
6. **Built-in Security:** Optional API-key authentication (`X-API-Key` or `Bearer`), payload size limits (15 MB), MIME magic-byte verification, and strict DOM `textContent` injection.
7. **Frontend Web Tester:** Standalone, responsive HTML dashboard with drag-and-drop upload, table rendering, CSV/TSV export, and raw JSON inspection.

---

## 🚀 Quick Start Guide

### 1. Backend Setup & Run

#### A. Clone & Create Virtual Environment
```bash
git clone https://github.com/ASWINKUMAR-AE/ocr_Tamil.git
cd ocr_Tamil

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

#### B. Install Verified Dependencies
```bash
pip install -r requirements.txt
```

#### C. Download Model Checkpoints
The OCR engine uses three model checkpoints (`parseq_tamil.pt`, `craft_mlt_25k.pth`, and `parseq-bb5792a6.pt`).
Run this one-liner to download and cache them locally in `~/.model_weights`:

```bash
python -c "
import urllib.request, os, shutil
from pathlib import Path
mw = os.path.join(Path.home(), '.model_weights')
os.makedirs(mw, exist_ok=True)
base = 'https://huggingface.co/GnanaPrasath/ocr_tamil/resolve/main/'
for f in ['parseq_tamil.pt', 'craft_mlt_25k.pth']:
    dest = os.path.join(mw, f)
    if not os.path.exists(dest):
        print('Downloading', f)
        urllib.request.urlretrieve(base + f, dest)
shutil.copyfile(os.path.join(mw, 'parseq_tamil.pt'), os.path.join(mw, 'parseq_tamil_v6.ckpt'))
print('Model weights downloaded successfully.')
"
```

#### D. Start the Backend API Server
```bash
# Using launcher script
python run.py --host 0.0.0.0 --port 8000

# Or with live code reload (development)# Project: Improve Tamil + English Handwritten Table OCR Accuracy

Act as a senior OCR engineer, computer vision developer, and full-stack engineer.

## Objective

Improve my existing OCR application so that it accurately extracts handwritten and printed Tamil, English, numbers, and mixed-language tabular data from uploaded images.

The main test case is a handwritten farmer register containing:

1. S.No
2. Farmer Name
3. Phone No.
4. Pashu Aadhaar (12 Digit)

The application currently misses rows, merges cells, misrecognizes Tamil and English text, and places extracted values in incorrect columns.

**Do not create a new project or replace the existing application unnecessarily. First inspect the repository and improve the current implementation.**

## Phase 1: Audit the existing code

1. Inspect the frontend, backend, OCR engine, model configuration, dependencies, upload flow, API endpoints, and HTML table rendering.
2. Identify the actual causes of missing rows, incorrect character recognition, and cell misalignment.
3. Check whether OCR coordinates and confidence scores are available.
4. Identify existing Tamil OCR support and determine whether the selected engine can recognize handwriting adequately.
5. Preserve existing working APIs, authentication, database integration, UI, and deployment configuration.
6. Produce a short implementation plan based on the actual repository before modifying files.

Do not assume a particular framework or OCR engine without inspecting the project.

## Phase 2: Image preprocessing

Implement a preprocessing pipeline that preserves the original uploaded image.

- Correct image orientation and EXIF rotation.
- Detect and correct skew and perspective distortion.
- Correct uneven lighting and shadows.
- Apply suitable denoising, contrast enhancement, and adaptive thresholding.
- Upscale small text when appropriate.
- Generate alternative preprocessing variants when the original image is difficult to read.
- Avoid preprocessing that removes thin handwriting strokes or Tamil vowel marks.
- Detect image blur and poor resolution and provide useful warnings.
- Keep the original image available for visual verification.

Use OpenCV or the existing compatible image-processing libraries.

## Phase 3: Table structure detection

Do not depend on plain OCR text output to reconstruct the table.

- Detect the table's outer boundary.
- Detect horizontal row boundaries and vertical column boundaries.
- Support ruled tables, partially ruled tables, uneven handwritten tables, and photographs of notebook pages.
- Use line detection and text-layout analysis together.
- Detect merged cells and multi-line cell content.
- Correctly identify headers separately from data rows.
- Assign recognized words to cells using bounding-box coordinates and overlap.
- Sort rows from top to bottom and cells from left to right.
- Do not discard a row just because one cell is empty or unreadable.
- Do not shift later rows when an earlier row is incomplete.
- Preserve page, row, and column coordinates for debugging.

When table boundaries are uncertain, return a review warning instead of silently inventing a table structure.

## Phase 4: Tamil and English OCR

Support Tamil, English, numerals, and mixed-language documents.

- Inspect the current OCR engine and its supported languages.
- Select suitable OCR models based on actual Tamil handwriting support, English handwriting support, deployment resources, and licensing.
- Run OCR on detected cells or regions, not only on the entire page.
- Use separate recognition strategies for text fields and numeric fields when beneficial.
- Preserve Tamil Unicode characters and combining marks.
- Handle English initials, punctuation, abbreviations, and names.
- Compare multiple OCR outputs only when the improvement justifies the added latency.
- Use confidence scores and disagreement detection to flag uncertain text.
- Do not claim perfect handwriting recognition.
- Do not translate names or silently normalize a person's name into a different name.

If the current engine cannot reliably recognize handwritten Tamil, document this limitation and provide an optional pluggable OCR engine or model adapter.

## Phase 5: Field-level validation

Support configurable column schemas.

For the farmer-register test case:

- S.No: positive integer, checked for missing or duplicate sequence numbers.
- Farmer Name: Tamil or English text, including initials, punctuation, and spaces.
- Phone No.: string containing exactly 10 digits after permitted formatting is removed. For this dataset, verify that it is a plausible Indian mobile number.
- Pashu Aadhaar: string containing exactly 12 digits.

Important:
- Store phone numbers and Pashu Aadhaar identifiers as strings, never as floating-point numbers.
- Preserve leading zeros.
- Do not automatically invent missing digits.
- Do not silently replace uncertain characters with assumed digits.
- Validate formatting separately from identity verification.
- A 12-digit value must not be treated as an officially verified identity merely because its length is correct.
- Flag duplicates for human review rather than automatically deleting records.

Use configurable schemas rather than hardcoding this farmer register into the entire OCR system.

## Phase 6: OCR output and HTML table

Return structured results containing:

- Document ID and page number.
- Table headers and detected column boundaries.
- Each row and its individual cells.
- Original OCR text.
- Bounding boxes and confidence scores when available.
- Validation status and warnings.
- Row-level and cell-level review status.

Generate a semantic HTML table using proper thead, tbody, tr, th, and td elements.

Requirements:
- Preserve row-to-column alignment.
- Escape extracted content to prevent HTML injection.
- Do not render OCR output as executable HTML.
- Support Tamil Unicode correctly.
- Keep cells editable.
- Allow correction of individual cells.
- Show uncertain cells with accessible warning indicators.
- Keep original OCR output available for comparison.
- Support CSV and Excel export using appropriate text types for identifiers.

## Phase 7: Review and correction UI

Create a responsive interface with:

1. The original image.
2. An optional processed-image preview.
3. The extracted table.
4. A row and cell confidence display.
5. Highlighting for invalid or uncertain values.
6. A way to select a table cell and inspect its corresponding image region.
7. Manual correction, save, and export controls.
8. Clear distinction between extracted, manually corrected, and verified data.

Never automatically mark records as verified just because OCR has completed.

## Phase 8: Testing and acceptance criteria

Create automated tests using synthetic images and manually verified real-world samples.

Include:
- English handwritten tables.
- Tamil handwritten tables.
- Mixed Tamil and English tables.
- Numeric identifiers.
- Missing cells and empty rows.
- Curved notebook pages.
- Uneven lighting and shadows.
- Skewed and perspective-distorted photographs.
- Overlapping handwriting and table lines.
- Multiple tables and multi-page documents.
- Duplicate identifiers and malformed phone numbers.

Measure:
- Character error rate.
- Word error rate.
- Exact cell accuracy.
- Exact row accuracy.
- Column assignment accuracy.
- Field-specific exact match accuracy.
- Processing time and memory consumption.

Do not claim a target accuracy unless tests demonstrate it. Keep a manually verified ground-truth dataset and compare results before and after each change.

## Phase 9: Deployment and delivery

- Preserve existing API contracts wherever possible.
- Add regression tests for existing endpoints.
- Handle model initialization, dependency failures, timeouts, and memory limits.
- Reuse existing deployment settings and environment variables.
- Avoid exposing API keys in frontend code or logs.
- Add structured logging without unnecessarily logging personal data.
- Document required packages, model downloads, configuration, endpoints, and deployment steps.

## Final instructions

Work incrementally.

First report the repository audit and proposed implementation plan. Then implement the changes in small, testable stages. Run the available tests, fix regressions, and report which tests passed and failed.

Do not fabricate missing OCR values, do not silently discard rows, and do not claim the problem is fixed without evidence from the test dataset.

The goal is reliable, reviewable extraction—not merely producing a visually attractive HTML table.
python run.py --host 0.0.0.0 --port 8000 --reload
```

---

### Run with Docker

Use Docker with Linux containers. The image uses Python 3.10 and CPU-only
PyTorch, serves both the API and web tester on port 8000, and runs as a
non-root user.

```bash
docker build -t tamil-ocr .
docker run -d --name tamil-ocr -p 8000:8000 --mount type=volume,source=tamil-ocr-cache,target=/home/ocr/.cache tamil-ocr
docker logs -f tamil-ocr
```

Open the web tester at `http://localhost:8000`, API documentation at
`http://localhost:8000/docs`, or check readiness at `http://localhost:8000/health`.
The image includes a health check that succeeds once the OCR engine is ready.

The build downloads the Tamil and CRAFT checkpoints. On first startup, the OCR
library also downloads the English PARSeq model and its Torch Hub source, so
internet access is required and startup can take several minutes. The named
`tamil-ocr-cache` volume retains these downloads across container replacements.

To supply custom settings, copy `.env.example` to `.env`, edit it, and add
`--env-file .env` before the image name in `docker run`. Keep `HOST=0.0.0.0`
and `PORT=8000` for the port mapping above. To use a different host port, change
the mapping, for example `-p 8080:8000`. This image is CPU-only; keep
`ENABLE_CUDA=false`. Local `.env` files, virtual environments, model files,
and runtime caches are excluded from the build context.

### 2. Frontend Web Tester Run

Open a second terminal window to serve the standalone web tester:

```bash
cd frontend
python -m http.server 5500
```

Open in your browser:
👉 **`http://127.0.0.1:5500/tamil-ocr-tester.html`**

---

## 🌐 Dynamic LAN IP & Network Accessibility

When the server starts, it binds to `0.0.0.0` and displays your machine's actual LAN IP:

```text
================================================================================
           TAMIL OCR REST API SERVER - PRODUCTION READY
================================================================================
Engine:        ocr-tamil v0.1.4 (PARSeq Tamil + English, CRAFT Detector)
Server Host:   0.0.0.0
Server Port:   8000
--------------------------------------------------------------------------------
ACCESSIBILITY URLS:
  * Localhost:         http://127.0.0.1:8000
  * LAN / Wi-Fi URL:   http://<YOUR-LAN-IP>:8000
  * API Documentation: http://<YOUR-LAN-IP>:8000/docs
  * Health Endpoint:   http://<YOUR-LAN-IP>:8000/health
================================================================================
```

### Allowing Port 8000 Through Firewall

- **Windows (Run in Administrator PowerShell):**
  ```powershell
  New-NetFirewallRule -DisplayName "Tamil-OCR-Port-8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
  ```
- **Linux (UFW):**
  ```bash
  sudo ufw allow 8000/tcp
  sudo ufw reload
  ```

---

## 📖 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check verifying server and `surya-ocr` / ParSeq model initialization. |
| `GET` | `/api/v1/info` | Capabilities, supported MIME formats, limits, engine versions, and dynamic network metadata. |
| `POST`| `/api/v1/ocr` | Full OCR: extracts Tamil & English text and detects tabular structures. |
| `POST`| `/api/v1/ocr/structured`| **Aavin Register Intelligence (Surya 0.1.2)**: Extracts tabular structures and maps candidate records (Farmers, 12-digit Pashu Aadhaar IDs, milk quantities) in a review-first schema. |
| `POST`| `/api/v1/ocr/table`| Dedicated table extraction (Backward-compatible schema: `table_id`, `headers`, `rows`, `columns`, `confidence`). |
| `POST`| `/api/v1/ocr/text` | Fast plain-text extraction skipping table analysis. |
| `POST`| `/api/v1/ocr/batch`| Batch processing of up to 10 images with concurrency controls. |

### Example cURL Request

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/ocr" \
  -F "file=@sample_images/0.jpg" \
  -F "extract_tables=true" \
  -F "preprocess=true"
```

### Example JSON Response

```json
{
  "success": true,
  "request_id": "f5831969-980b-4835-bd91-ce730cb826ef",
  "filename": "0.jpg",
  "content_type": "image/jpeg",
  "extracted_text": "கொடைக்கானல்\nKodaikanal",
  "tables": [],
  "metadata": {
    "ocr_engine": "ocr-tamil",
    "preprocessing_applied": true,
    "processing_time_ms": 780.25,
    "image_dimensions": {
      "width": 600,
      "height": 200
    }
  },
  "warnings": []
}
```

---

## 🖥️ HTML Testing Web Application

Located at [`frontend/tamil-ocr-tester.html`](frontend/tamil-ocr-tester.html), this self-contained interface provides:

- **Connection Indicator:** Live connection test to `/health` with latency metrics.
- **Drag & Drop Upload Zone:** Supports JPEG, PNG, and WebP with preview, dimensions, and file size.
- **Unicode Tamil Display:** Renders Tamil characters accurately with word count, character count, and single-click **Copy Text**.
- **Native HTML Table Rendering:** Converts detected tables into real browser `<table>` elements with:
  - **Download CSV** (with UTF-8 BOM for Microsoft Excel compatibility).
  - **Copy TSV** (for instant pasting into Google Sheets / Excel).
  - **Download JSON**.
- **Collapsible Raw JSON Viewer:** Formatted code block showing exact backend responses.
- **Security:** Renders recognized text strictly via `textContent` to prevent HTML injection attacks.

---

## 📱 Client Integration Examples

Working, tested client snippets are provided in [`client_examples/`](client_examples/):

1. **React Native / Expo ([`client_examples/react_native_expo.js`](client_examples/react_native_expo.js)):**
   Complete component with Expo `ImagePicker` and native `fetch` with `FormData`.
2. **Browser JavaScript ([`client_examples/web_fetch.html`](client_examples/web_fetch.html)):**
   Single HTML file demonstrating asynchronous file uploads and table rendering.
3. **Desktop Python ([`client_examples/desktop_python.py`](client_examples/desktop_python.py)):**
   Command-line utility using the `requests` library.
4. **Node.js ([`client_examples/node_client.js`](client_examples/node_client.js)):**
   Node.js 18+ client using native `fetch` and `Blob`.
5. **cURL Scripts ([`client_examples/curl_examples.sh`](client_examples/curl_examples.sh)):**
   Ready-to-run terminal commands for all endpoints.

---

## 🧪 Automated Testing

The backend includes a comprehensive pytest suite covering all endpoints, schemas, error paths, and edge cases:

```bash
pytest tests/ -v
```

### Verified Test Results (19/19 Passing):
- `tests/test_health.py` — Health check (200 OK & 503 degraded states).
- `tests/test_info.py` — Server capabilities and network metadata.
- `tests/test_ocr.py` — Real Tamil text recognition and line preservation.
- `tests/test_tables.py` — Table extraction schema and non-table warnings.
- `tests/test_batch.py` — Multi-image processing and batch limit enforcement.
- `tests/test_validation.py` — Non-image rejection, corrupted payload, and 15MB size limit.
- `tests/test_security.py` — Optional API key authentication (`X-API-Key` & `Bearer`).
- `tests/test_cors.py` — CORS preflight and allowed origin headers.
- `tests/test_network.py` — LAN IPv4 classification and interface discovery.

---

## ❓ Troubleshooting & FAQ

#### 1. "Failed to fetch" or Connection Refused
- Ensure the backend server is running (`python run.py`).
- If calling from a mobile device or another computer, ensure both devices are on the same Wi-Fi network and use the server's LAN IP (e.g. `http://10.110.44.105:8000`).
- Ensure Windows Firewall or Linux UFW allows TCP port 8000.

#### 2. NumPy 2.x Compatibility
- `torch 2.1.2` requires `numpy<2.0.0`. The project pins `numpy>=1.24.0,<2.0.0` in `requirements.txt`.

#### 3. Table Borders Not Detected
- For optimal table extraction, ensure table grid lines have sufficient contrast. If borders are absent, the engine falls back to spatial text alignment and includes a warning field.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
