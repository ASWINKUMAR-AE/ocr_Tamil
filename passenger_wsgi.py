"""
Phusion Passenger WSGI entry point for cPanel hosting.
Bridges the FastAPI ASGI application to WSGI using a2wsgi.
"""

import os
import sys

# 1. Ensure project root directory is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# 2. Set environment defaults if not already present
os.environ.setdefault("ENVIRONMENT", "production")

# 3. Import ASGI to WSGI adapter and FastAPI app
try:
    from a2wsgi import ASGIMiddleware
    from app.main import app as fastapi_app

    # Phusion Passenger looks for the variable named 'application'
    application = ASGIMiddleware(fastapi_app)
except Exception as e:
    # Helpful fallback in case dependencies are missing
    def application(environ, start_response):
        status = "500 Internal Server Error"
        output = f"Tamil OCR cPanel Startup Error: {str(e)}\nPlease ensure dependencies (fastapi, a2wsgi, torch, ocr-tamil) are installed in your cPanel virtual environment.".encode("utf-8")
        response_headers = [
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", str(len(output)))
        ]
        start_response(status, response_headers)
        return [output]
