# Tamil OCR API Testing Web Page

A standalone, responsive developer testing dashboard for the Tamil OCR REST API backend built with FastAPI and `ocr-tamil==0.1.4`.

File Location:
```text
frontend/tamil-ocr-tester.html
```

---

## 1. Quick Start Guide

### Step 1: Ensure the Backend API Server is Running

In your backend terminal:

```powershell
# In project root: d:\ocr
.\.venv\Scripts\Activate.ps1
python run.py --host 0.0.0.0 --port 8000
```

Verify that the backend is running by checking:
- Local URL: `http://127.0.0.1:8000/health`
- LAN URL: `http://<your-server-ip>:8000/health`

### Step 2: Serve the Frontend HTML Page

In a separate terminal, serve the `frontend/` directory using Python's built-in HTTP server:

```powershell
cd d:\ocr\frontend
python -m http.server 5500
```

Then open your browser and navigate to:

```text
http://127.0.0.1:5500/tamil-ocr-tester.html
```

*(Note: You can also open the file directly via `file:///d:/ocr/frontend/tamil-ocr-tester.html`, but running an HTTP server is recommended to prevent local browser-origin restrictions).*

---

## 2. API Configuration & Switching Hosts

The tester interface features an editable **Server Base URL** input field with quick-preset buttons:

- **Local Testing:** `http://127.0.0.1:8000` (when running the browser on the same computer as the server).
- **LAN / Mobile Testing:** `http://<your-server-ip>:8000` (e.g., `http://10.110.44.105:8000` when testing from another laptop, tablet, or phone connected to the same Wi-Fi network).
- **HTTPS Deployment:** `https://your-domain.com` (when deployed behind a reverse proxy like Nginx or Caddy with SSL).

### Optional API Key Support

If the backend has `API_KEY` enabled in `.env`:
- Enter the secret token in the **API Key (Optional)** field.
- The frontend will automatically attach it via the `X-API-Key` HTTP header.

---

## 3. Key Features Verified

| Feature | Description | Status |
|---|---|---|
| **API Connection Test** | Pings `/health` with an 8-second timeout, measuring latency and verifying engine status (`ocr-tamil v0.1.4`). | Verified |
| **Drag & Drop Upload** | Supports JPEG, PNG, and WebP up to 15 MB with client-side format & size validation. | Verified |
| **Live Image Preview** | Shows the image preview, pixel dimensions (`W × H`), and file size before uploading. | Verified |
| **Tamil Text Rendering** | Displays recognized Tamil and mixed-language text with Unicode support and line preservation. | Verified |
| **HTML Table Rendering** | Renders structured tables using real HTML `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`. | Verified |
| **DOM Security** | Strictly uses `textContent` (never `innerHTML`) to prevent XSS from untrusted OCR text. | Verified |
| **CSV & TSV Export** | Download table as `.csv` (with UTF-8 BOM for Excel Tamil support) or Copy as TSV for Google Sheets. | Verified |
| **Raw JSON Viewer** | Dark-themed collapsible code block showing full API response, HTTP status, and Request ID. | Verified |

---

## 4. Troubleshooting & Network Access

### 1. Connection Refused / "Failed to fetch"
- Check that the backend server is running (`python run.py`).
- Check that the Server Base URL in the UI matches the actual host and port.
- If accessing from another device on the LAN, verify that the Windows Firewall allows incoming TCP traffic on port 8000:
  ```powershell
  New-NetFirewallRule -DisplayName "Tamil-OCR-Port-8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
  ```

### 2. CORS Issues
- The backend has CORS enabled with `allow_origins=["*"]` by default.
- If testing from an explicit origin or domain, configure `CORS_ORIGINS` in your backend `.env` file.

### 3. Mixed Content Warning
- If hosting this frontend over HTTPS, modern browsers will block requests to an insecure HTTP backend (`http://...`). For HTTPS frontends, ensure the backend is also secured with HTTPS.
