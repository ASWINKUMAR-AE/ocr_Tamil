"""
Entry point script for launching the Tamil OCR REST API server.
Reads host and port from environment or command-line arguments.
"""

import argparse
import uvicorn
from app.config import settings


def main():
    parser = argparse.ArgumentParser(description="Tamil OCR REST API Server")
    parser.add_argument(
        "--host",
        type=str,
        default=settings.HOST,
        help=f"Host network interface to bind to (default: {settings.HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.PORT,
        help=f"Port number to listen on (default: {settings.PORT})"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development only)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )

    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
