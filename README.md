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

# Or with live code reload (development)
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
| `GET` | `/health` | Health check verifying server and `ocr-tamil` model initialization. |
| `GET` | `/api/v1/info` | Capabilities, supported MIME formats, limits, and dynamic network metadata. |
| `POST`| `/api/v1/ocr` | Full OCR: extracts Tamil & English text and detects tabular structures. |
| `POST`| `/api/v1/ocr/table`| Dedicated table extraction returning structured rows, columns, and key-value data. |
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
