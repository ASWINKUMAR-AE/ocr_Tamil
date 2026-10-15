"""
Desktop Python Client Integration Example for Tamil OCR REST API.
Demonstrates uploading an image, extracting Tamil & mixed text, parsing structured tables,
and handling network / server errors.
"""

import sys
import os
import argparse
import json
import requests

# Default LAN URL (adjust according to the server startup banner)
DEFAULT_SERVER_URL = "http://10.110.44.105:8000"


def perform_ocr(
    image_path: str,
    server_url: str = DEFAULT_SERVER_URL,
    api_key: str = None,
    extract_tables: bool = True,
    preprocess: bool = True,
    timeout: int = 30
):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return False

    endpoint = f"{server_url.rstrip('/')}/api/v1/ocr"
    print(f"Connecting to Tamil OCR API at: {endpoint}")
    print(f"Uploading image: {image_path}...")

    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/png")}
            data = {
                "extract_tables": "true" if extract_tables else "false",
                "preprocess": "true" if preprocess else "false",
            }

            response = requests.post(
                endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=timeout
            )

        if response.status_code == 200:
            result = response.json()
            print("\n" + "=" * 60)
            print("OCR PROCESSING SUCCESSFUL")
            print("=" * 60)
            print("Request ID:       ", result.get("request_id"))
            print("File:             ", result.get("filename"))
            print("Processing Time:  ", result["metadata"]["processing_time_ms"], "ms")
            print("-" * 60)
            print("EXTRACTED TAMIL TEXT:")
            print("-" * 60)
            print(result.get("extracted_text") or "(No text recognized)")
            print("-" * 60)

            tables = result.get("tables", [])
            if tables:
                print(f"DETECTED TABLES ({len(tables)}):")
                for tbl in tables:
                    print(f"\nTable #{tbl['table_index'] + 1} ({tbl['row_count']} rows x {tbl['column_count']} cols)")
                    print("Headers: ", " | ".join(tbl["headers"]))
                    print("-" * 40)
                    for row in tbl["rows"]:
                        print("  " + " | ".join(row))
                    print("\nMachine-readable JSON Data:")
                    print(json.dumps(tbl.get("data", []), indent=2, ensure_ascii=False))
            else:
                print("No tables detected.")

            if result.get("warnings"):
                print("\nWarnings:", result["warnings"])

            return True

        else:
            print(f"Error: Server responded with HTTP {response.status_code}")
            try:
                err_json = response.json()
                print("Details:", json.dumps(err_json, indent=2, ensure_ascii=False))
            except Exception:
                print("Response:", response.text)
            return False

    except requests.exceptions.ConnectionError:
        print(f"Connection Error: Could not connect to {server_url}.")
        print("Please verify:")
        print(" 1. The OCR server is running.")
        print(" 2. Both devices are on the same Wi-Fi / Local Area Network.")
        print(" 3. The server firewall allows TCP traffic on the configured port.")
        return False
    except requests.exceptions.Timeout:
        print(f"Timeout: Server took longer than {timeout} seconds to respond.")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Tamil OCR Python Desktop Client")
    parser.add_argument("image_path", help="Path to image file (JPEG, PNG, WebP)")
    parser.add_argument("--url", default=DEFAULT_SERVER_URL, help=f"Server URL (default: {DEFAULT_SERVER_URL})")
    parser.add_argument("--api-key", default=None, help="Optional API key")
    parser.add_argument("--no-tables", action="store_true", help="Disable table detection")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable image preprocessing")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")

    args = parser.parse_args()
    success = perform_ocr(
        image_path=args.image_path,
        server_url=args.url,
        api_key=args.api_key,
        extract_tables=not args.no_tables,
        preprocess=not args.no_preprocess,
        timeout=args.timeout
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
