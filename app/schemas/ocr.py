"""
Pydantic Schemas for Tamil OCR API Requests and Responses.
Provides strong typing, validation, and serialization matching the API specification.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Health & Service Information Schemas
# ============================================================================

class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    service: str = Field(..., json_schema_extra={"example": "Tamil OCR API"})
    ocr_engine: str = Field(..., json_schema_extra={"example": "ocr-tamil"})
    version: str = Field(..., json_schema_extra={"example": "0.1.4"})


class NetworkInterfaceInfo(BaseModel):
    interface: str
    ip: str
    netmask: str
    is_primary: bool
    is_lan: bool
    status: str


class NetworkSummary(BaseModel):
    configured_host: str
    configured_port: int
    primary_lan_ip: str
    local_url: str
    lan_url: str
    docs_url: str
    health_url: str
    interfaces: List[NetworkInterfaceInfo]


class ApiInfoResponse(BaseModel):
    service: str
    version: str
    ocr_engine: str
    supported_image_formats: List[str]
    max_upload_size_mb: int
    max_image_dimension: int
    max_batch_size: int
    auth_enabled: bool
    capabilities: List[str]
    network: NetworkSummary


# ============================================================================
# Table & Extraction Schemas
# ============================================================================

class TableColumn(BaseModel):
    key: str = Field(..., description="Stable programmatic key for the column")
    label: str = Field(..., description="Original Tamil/English text header label")


class TableResult(BaseModel):
    table_index: int = Field(default=0, description="0-indexed position of table in image")
    headers: List[str] = Field(default_factory=list, description="Column header strings")
    rows: List[List[str]] = Field(default_factory=list, description="Row values as 2D string array")
    row_count: int = Field(default=0, description="Total number of rows")
    column_count: int = Field(default=0, description="Total number of columns")
    columns: List[TableColumn] = Field(default_factory=list, description="Structured column definitions")
    data: List[Dict[str, str]] = Field(default_factory=list, description="Key-value row objects")
    confidence: Optional[float] = Field(default=None, description="Detection confidence score")


class ProcessingMetadata(BaseModel):
    ocr_engine: str = Field(default="ocr-tamil")
    preprocessing_applied: bool = Field(default=True)
    processing_time_ms: float = Field(..., description="Execution time in milliseconds")
    image_dimensions: Optional[Dict[str, int]] = Field(default=None, description="Width and height")


# ============================================================================
# Main OCR Response Schemas
# ============================================================================

class OCRResponse(BaseModel):
    success: bool = True
    request_id: str = Field(..., description="Unique UUID for request tracing")
    filename: str = Field(..., description="Name of uploaded image file")
    content_type: str = Field(..., description="MIME content type")
    extracted_text: str = Field(..., description="Full recognized Tamil & mixed-language text")
    tables: List[TableResult] = Field(default_factory=list, description="Detected tabular structures")
    metadata: ProcessingMetadata
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")


class TextOnlyResponse(BaseModel):
    success: bool = True
    request_id: str
    filename: str
    content_type: str
    extracted_text: str
    metadata: ProcessingMetadata
    warnings: List[str] = Field(default_factory=list)


class TableOnlyResponse(BaseModel):
    success: bool = True
    request_id: str
    filename: str
    content_type: str
    tables: List[TableResult]
    extracted_text: str
    metadata: ProcessingMetadata
    warnings: List[str] = Field(default_factory=list)


# ============================================================================
# Batch OCR Schemas
# ============================================================================

class BatchItemResult(BaseModel):
    filename: str
    success: bool
    extracted_text: Optional[str] = None
    tables: List[TableResult] = Field(default_factory=list)
    metadata: Optional[ProcessingMetadata] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class BatchOCRResponse(BaseModel):
    success: bool = True
    request_id: str
    total_images: int
    successful_images: int
    failed_images: int
    results: List[BatchItemResult]
    total_processing_time_ms: float


# ============================================================================
# Error Response Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    success: bool = False
    request_id: str
    error: str
    error_code: str
    details: Optional[Any] = None
