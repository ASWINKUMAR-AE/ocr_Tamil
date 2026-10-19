"""
Automated API Verification & Regression Test Suite for Tamil OCR REST API.
Tests all primary endpoints (Health, Docs, UI, Text OCR, Table Extraction)
and validates multi-format output (JSON, HTML, TEXT).
"""

import argparse
import os
import sys
import time
import requests

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def log_pass(name: str, detail: str = ""):
    print(f"[{GREEN}PASS{RESET}] {name} {detail}")


def log_fail(name: str, detail: str = ""):
    print(f"[{RED}FAIL{RESET}] {name} {detail}")


def log_info(msg: str):
    print(f"[{BLUE}INFO{RESET}] {msg}")


def run_tests(base_url: str):
    base_url = base_url.rstrip("/")
    log_info(f"Starting API Verification Suite against {base_url}")
    failures = 0
    start_total = time.time()

    # 1. Health check
    try:
        t0 = time.time()
        res = requests.get(f"{base_url}/health", timeout=10)
        dur = (time.time() - t0) * 1000
        if res.status_code == 200 and res.json().get("status") == "ok":
            log_pass("GET /health", f"({dur:.1f}ms) -> {res.json()}")
        else:
            log_fail("GET /health", f"Status: {res.status_code}, Body: {res.text}")
            failures += 1
    except Exception as e:
        log_fail("GET /health", f"Connection error: {e}")
        failures += 1

    # 2. Interactive Docs
    try:
        t0 = time.time()
        res = requests.get(f"{base_url}/docs", timeout=10)
        dur = (time.time() - t0) * 1000
        if res.status_code == 200:
            log_pass("GET /docs", f"({dur:.1f}ms) -> Swagger UI OK")
        else:
            log_fail("GET /docs", f"Status: {res.status_code}")
            failures += 1
    except Exception as e:
        log_fail("GET /docs", f"Error: {e}")
        failures += 1

    # 3. Web Dashboard /
    try:
        t0 = time.time()
        res = requests.get(f"{base_url}/", timeout=10)
        dur = (time.time() - t0) * 1000
        if res.status_code == 200 and ("Tamil OCR" in res.text or "<html" in res.text.lower()):
            log_pass("GET /", f"({dur:.1f}ms) -> Web Dashboard Loaded")
        else:
            log_fail("GET /", f"Status: {res.status_code}")
            failures += 1
    except Exception as e:
        log_fail("GET /", f"Error: {e}")
        failures += 1

    # 4. Text Extraction with Multi-Format validation (/api/v1/ocr/text)
    sample_img = os.path.join(os.path.dirname(__file__), "sample_images", "0.jpg")
    if os.path.exists(sample_img):
        try:
            t0 = time.time()
            with open(sample_img, "rb") as f:
                res = requests.post(
                    f"{base_url}/api/v1/ocr/text",
                    files={"file": ("0.jpg", f, "image/jpeg")},
                    data={"preprocess": "true", "format": "json"},
                    timeout=30,
                )
            dur = (time.time() - t0) * 1000
            if res.status_code == 200:
                data = res.json()
                text = data.get("text", "").replace("\n", " ")
                has_json_fmt = isinstance(data.get("json"), dict) and "lines" in data.get("json", {})
                has_html_fmt = "<div" in str(data.get("html"))
                has_text_fmt = len(data.get("text", "")) > 0
                has_formats_obj = "formats" in data and "json" in data["formats"]

                if has_json_fmt and has_html_fmt and has_text_fmt and has_formats_obj:
                    log_pass("POST /api/v1/ocr/text (JSON + HTML + TEXT fields)", f"({dur:.1f}ms) -> Text: '{text}' [All formats verified]")
                else:
                    log_fail("POST /api/v1/ocr/text (Format Check)", f"Missing format fields in response: {list(data.keys())}")
                    failures += 1
            else:
                log_fail("POST /api/v1/ocr/text", f"Status: {res.status_code}, Body: {res.text}")
                failures += 1
        except Exception as e:
            log_fail("POST /api/v1/ocr/text", f"Error: {e}")
            failures += 1
    else:
        log_info(f"Skipping text extraction test, {sample_img} not found")

    # 5. Direct HTML Format test
    if os.path.exists(sample_img):
        try:
            t0 = time.time()
            with open(sample_img, "rb") as f:
                res = requests.post(
                    f"{base_url}/api/v1/ocr/text",
                    files={"file": ("0.jpg", f, "image/jpeg")},
                    data={"format": "html"},
                    timeout=30,
                )
            dur = (time.time() - t0) * 1000
            content_type = res.headers.get("content-type", "")
            if res.status_code == 200 and "text/html" in content_type and "<div" in res.text:
                log_pass("POST /api/v1/ocr/text (format=html direct)", f"({dur:.1f}ms) -> Content-Type: {content_type}")
            else:
                log_fail("POST /api/v1/ocr/text (format=html direct)", f"Status: {res.status_code}, Content-Type: {content_type}")
                failures += 1
        except Exception as e:
            log_fail("POST /api/v1/ocr/text (format=html direct)", f"Error: {e}")
            failures += 1

    # 6. Direct Plain Text Format test
    if os.path.exists(sample_img):
        try:
            t0 = time.time()
            with open(sample_img, "rb") as f:
                res = requests.post(
                    f"{base_url}/api/v1/ocr/text",
                    files={"file": ("0.jpg", f, "image/jpeg")},
                    data={"format": "text"},
                    timeout=30,
                )
            dur = (time.time() - t0) * 1000
            content_type = res.headers.get("content-type", "")
            if res.status_code == 200 and "text/plain" in content_type and "கொடைக்கானல்" in res.text:
                log_pass("POST /api/v1/ocr/text (format=text direct)", f"({dur:.1f}ms) -> Content-Type: {content_type}")
            else:
                log_fail("POST /api/v1/ocr/text (format=text direct)", f"Status: {res.status_code}, Content-Type: {content_type}")
                failures += 1
        except Exception as e:
            log_fail("POST /api/v1/ocr/text (format=text direct)", f"Error: {e}")
            failures += 1

    # 7. Table Extraction (/api/v1/ocr)
    table_img = os.path.join(os.path.dirname(__file__), "sample_images", "table_test.png")
    if os.path.exists(table_img):
        try:
            t0 = time.time()
            with open(table_img, "rb") as f:
                res = requests.post(
                    f"{base_url}/api/v1/ocr",
                    files={"file": ("table_test.png", f, "image/png")},
                    data={"extract_tables": "true", "preprocess": "true", "format": "json"},
                    timeout=30,
                )
            dur = (time.time() - t0) * 1000
            if res.status_code == 200:
                data = res.json()
                tables = data.get("tables", [])
                has_html = "<table" in str(data.get("html"))
                has_json = "tables" in data.get("json", {})
                if has_html and has_json:
                    log_pass("POST /api/v1/ocr (Table Multi-Format)", f"({dur:.1f}ms) -> Detected {len(tables)} table(s) [HTML table + JSON grid verified]")
                else:
                    log_fail("POST /api/v1/ocr (Table Multi-Format)", f"Missing format structures in table response")
                    failures += 1
            else:
                log_fail("POST /api/v1/ocr (Table)", f"Status: {res.status_code}, Body: {res.text}")
                failures += 1
        except Exception as e:
            log_fail("POST /api/v1/ocr (Table)", f"Error: {e}")
            failures += 1

    # 8. Backward-Compatible Table Extraction (/api/v1/ocr/table)
    if os.path.exists(table_img):
        try:
            t0 = time.time()
            with open(table_img, "rb") as f:
                res = requests.post(
                    f"{base_url}/api/v1/ocr/table",
                    files={"file": ("table_test.png", f, "image/png")},
                    timeout=30,
                )
            dur = (time.time() - t0) * 1000
            if res.status_code == 200:
                data = res.json()
                if "table_id" in data and "headers" in data and "rows" in data and "columns" in data:
                    log_pass("POST /api/v1/ocr/table (Legacy Contract)", f"({dur:.1f}ms) -> Backward compatibility verified (table_id, headers, rows, columns)")
                else:
                    log_fail("POST /api/v1/ocr/table", f"Missing legacy keys in response: {list(data.keys())}")
                    failures += 1
            else:
                log_fail("POST /api/v1/ocr/table", f"Status: {res.status_code}, Body: {res.text}")
                failures += 1
        except Exception as e:
            log_fail("POST /api/v1/ocr/table", f"Error: {e}")
            failures += 1

    # 9. Structured OCR & Aavin Register Intelligence (/api/v1/ocr/structured)
    if os.path.exists(table_img):
        try:
            t0 = time.time()
            with open(table_img, "rb") as f:
                res = requests.post(
                    f"{base_url}/api/v1/ocr/structured",
                    files={"file": ("table_test.png", f, "image/png")},
                    data={"document_type": "aavin_register", "language_hint": "ta,en"},
                    timeout=30,
                )
            dur = (time.time() - t0) * 1000
            if res.status_code == 200:
                data = res.json()
                engine_name = data.get("engine", {}).get("name")
                engine_ver = data.get("engine", {}).get("version")
                if data.get("success") and engine_name == "surya-ocr" and engine_ver == "0.1.2":
                    log_pass("POST /api/v1/ocr/structured (Surya 0.1.2 + Aavin)", f"({dur:.1f}ms) -> Engine: {engine_name}=={engine_ver}, Tables: {len(data.get('tables', []))}, Records: {len(data.get('records', []))}")
                else:
                    log_fail("POST /api/v1/ocr/structured", f"Unexpected response schema: {data}")
                    failures += 1
            else:
                log_fail("POST /api/v1/ocr/structured", f"Status: {res.status_code}, Body: {res.text}")
                failures += 1
        except Exception as e:
            log_fail("POST /api/v1/ocr/structured", f"Error: {e}")
            failures += 1

    total_dur = (time.time() - start_total)
    print("=" * 60)
    if failures == 0:
        print(f"{GREEN}ALL TESTS PASSED SUCCESSFULLY in {total_dur:.2f}s!{RESET}")
        return 0
    else:
        print(f"{RED}{failures} TEST(S) FAILED out of suite.{RESET}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Tamil OCR API")
    parser.add_argument("--url", default="http://127.0.0.1", help="Base URL of Tamil OCR server")
    args = parser.parse_args()
    sys.exit(run_tests(args.url))
