"""
Application Configuration for Tamil OCR REST API.
Loads settings from environment variables and .env file with robust validation.
"""

from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server & Network
    HOST: str = Field(default="0.0.0.0", description="Host interface to bind to")
    PORT: int = Field(default=8000, description="Port number to listen on")
    AUTO_DETECT_LAN_IP: bool = Field(default=True, description="Dynamically detect system LAN IP")

    # OCR Engine
    ENABLE_CUDA: bool = Field(default=False, description="Enable GPU CUDA if available")
    MAX_CONCURRENT_OCR: int = Field(default=2, ge=1, le=16, description="Max concurrent OCR workers")
    OCR_ENGINE_NAME: str = Field(default="surya-ocr", description="Primary OCR & layout engine name")
    SURYA_VERSION: str = Field(default="0.1.2", description="Pinned Surya OCR version")
    SURYA_DETECTOR_CHECKPOINT: str = Field(default="vikp/line_detector", description="HuggingFace model checkpoint for Surya line detector")
    SURYA_DETECTION_BATCH_SIZE: int = Field(default=2, ge=1, le=16, description="Batch size for Surya inference")
    SURYA_DETECTION_THRESHOLD: float = Field(default=0.6, ge=0.1, le=1.0, description="Text detection confidence threshold")
    AAVIN_PROFILE_ENABLED: bool = Field(default=True, description="Enable Aavin Smart Cattle Passport extraction profile")
    
    # Security & Limits
    API_KEY: Optional[str] = Field(default=None, description="Optional API key for securing endpoints")
    MAX_UPLOAD_SIZE_MB: int = Field(default=15, ge=1, le=100, description="Maximum image upload size in MB")
    MAX_IMAGE_DIMENSION: int = Field(default=4096, ge=512, le=10000, description="Max width/height in pixels")
    MAX_BATCH_SIZE: int = Field(default=10, ge=1, le=50, description="Max images per batch request")

    # CORS & Environment
    CORS_ORIGINS: str = Field(default="*", description="Allowed CORS origins (comma-separated or *)")
    ENVIRONMENT: str = Field(default="development", description="Runtime environment")
    LOG_LEVEL: str = Field(default="INFO", description="Logging verbosity")

    # App Metadata
    APP_NAME: str = "Tamil OCR API"
    APP_VERSION: str = "0.2.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        if not self.CORS_ORIGINS or self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def is_auth_enabled(self) -> bool:
        return bool(self.API_KEY and self.API_KEY.strip())


settings = Settings()
