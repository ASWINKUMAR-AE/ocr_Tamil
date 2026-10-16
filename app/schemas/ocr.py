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

class ColumnHeaderDetail(BaseModel):
    original: str = Field(default="", description="Detected header text from document")
    display: str = Field(default="", description="Display header text")
    original_header: Optional[str] = Field(default=None, description="Original raw detected header text")
    display_header: Optional[str] = Field(default=None, description="Clean display header text")
    semantic_type: str = Field(default="unknown", description="Inferred semantic type: identifier, name, date, gender, phone, address, quantity, amount, serial_number, unknown")
    confidence: float = Field(default=0.9, description="Confidence score")
    inference_method: str = Field(default="header", description="Inference method: explicit_header, pattern_and_context, or position_fallback")
    classification_warnings: List[str] = Field(default_factory=list, description="Column-level warnings, e.g. conflicting pattern vs header")


class TableColumn(BaseModel):
    key: str = Field(..., description="Stable programmatic key for the column")
    label: str = Field(..., description="Original Tamil/English text header label")
    original_header: Optional[str] = Field(default=None, description="Original raw detected header text")
    display_header: Optional[str] = Field(default=None, description="Clean display header text")
    semantic_type: Optional[str] = Field(default="unknown", description="Inferred semantic type")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    inference_method: Optional[str] = Field(default="header", description="explicit_header, pattern_and_context, or position_fallback")
    classification_warnings: List[str] = Field(default_factory=list, description="Column-level warnings")


class TableResult(BaseModel):
    table_index: int = Field(default=0, description="0-indexed position of table in image")
    table_id: Optional[int] = Field(default=None, description="1-indexed table identifier")
    headers: List[str] = Field(default_factory=list, description="Column header strings")
    headers_detail: List[ColumnHeaderDetail] = Field(default_factory=list, description="Detailed column header objects")
    rows: List[List[str]] = Field(default_factory=list, description="Row values as 2D string array")
    row_count: int = Field(default=0, description="Total number of rows")
    column_count: int = Field(default=0, description="Total number of columns")
    columns: List[TableColumn] = Field(default_factory=list, description="Structured column definitions")
    data: List[Dict[str, str]] = Field(default_factory=list, description="Key-value row objects")
    confidence: Optional[float] = Field(default=None, description="Detection confidence score")
    detection_strategy: Optional[str] = Field(default=None, description="Strategy used: bbox_clustering, morphology_grid, or text_alignment")
    is_uncertain: bool = Field(default=False, description="Flag indicating if the table structure requires user review")
    warnings: List[str] = Field(default_factory=list, description="Table-level warnings or notes")
    html: Optional[str] = Field(default=None, description="Semantic HTML table markup")
    detected_languages: List[str] = Field(default_factory=lambda: ["ta"], description="Languages detected in table")
    unassigned_text: Optional[str] = Field(default=None, description="Text tokens outside table bounding boundaries")


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
    detected_languages: List[str] = Field(default_factory=lambda: ["ta"], description="Languages detected across image: ta, en, or mixed")
    unassigned_text: Optional[str] = Field(default=None, description="Text tokens located outside recognized table boundaries")


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
