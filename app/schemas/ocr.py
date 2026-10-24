"""
Pydantic Schemas for Tamil OCR API Requests and Responses.
Provides strong typing, validation, and serialization matching the API specification.
Includes schemas for Aavin Abstract/Dispatch Reports (shift-wise production & dispatch tables).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


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


class OCRFormats(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    json: Dict[str, Any] = Field(..., description="Structured JSON representation of extracted data")
    html: str = Field(..., description="Semantic HTML representation of extracted data")
    text: str = Field(..., description="Plain-text representation with preserved line breaks")


# ============================================================================
# Engine & Aavin Domain Candidate Record Schemas
# ============================================================================

class EngineInfo(BaseModel):
    name: str = Field(default="surya-ocr", description="OCR & layout engine identifier")
    version: str = Field(default="0.1.2", description="Pinned engine version")


class CattleCandidateRecord(BaseModel):
    cow_index: Optional[int] = Field(default=None, description="1-indexed sequence number of cow (e.g. Cow 1 of N)")
    tag_id: str = Field(..., description="12-digit string Pashu Aadhaar / Tag ID, leading-zero safe")
    raw_tag_id: Optional[str] = Field(default=None, description="Original raw recognized string")
    breed: Optional[str] = Field(default=None, description="Breed name in Tamil or English")
    age: Optional[str] = Field(default=None, description="Age or lactation info")
    status: str = Field(default="candidate", description="Status of candidate record")
    review_required: bool = Field(default=True, description="Review flag for human confirmation")
    confidence: float = Field(default=0.90, description="Detection confidence score")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings or notes")


class MilkCandidateRecord(BaseModel):
    date: Optional[str] = Field(default=None, description="Collection date string")
    session: Optional[str] = Field(default=None, description="Morning or Evening session")
    litres: Optional[float] = Field(default=None, description="Supplied litres as decimal, not zero when absent")
    booth_id: Optional[str] = Field(default=None, description="Society or booth code")
    confidence: float = Field(default=0.90, description="Extraction confidence")
    review_required: bool = Field(default=True, description="Human review required flag")
    warnings: List[str] = Field(default_factory=list, description="Warnings")


class FarmerCandidateRecord(BaseModel):
    record_id: int = Field(default=1, description="Sequential candidate record ID")
    farmer_name: Optional[str] = Field(default=None, description="Normalized farmer name preserving Tamil Unicode")
    raw_farmer_name: Optional[str] = Field(default=None, description="Raw OCR recognized text before normalization")
    phone: Optional[str] = Field(default=None, description="Normalized 10-digit Indian mobile number string")
    raw_phone: Optional[str] = Field(default=None, description="Raw phone text from cell")
    village: Optional[str] = Field(default=None, description="Village or locality name")
    cow_count: Optional[int] = Field(default=None, description="Declared cattle count")
    member_id: Optional[str] = Field(default=None, description="Society member or farmer identifier")
    cattle_ids: List[str] = Field(default_factory=list, description="12-digit Pashu Aadhaar tag IDs belonging to this farmer")
    cattle: List[CattleCandidateRecord] = Field(default_factory=list, description="Individual cattle candidate records")
    milk_records: List[MilkCandidateRecord] = Field(default_factory=list, description="Milk collection entries")
    source_row_indices: List[int] = Field(default_factory=list, description="Table row indices contributing to this record")
    confidence: float = Field(default=0.90, description="Overall record confidence")
    review_required: bool = Field(default=True, description="Review required before saving")
    validation_status: str = Field(default="valid", description="Status: valid, review_required, or invalid")
    validation_warnings: List[str] = Field(default_factory=list, description="Field validation messages")


# ============================================================================
# Aavin Abstract / Dispatch Report Schemas
# ============================================================================

class AbstractProductRecord(BaseModel):
    """
    One product row extracted from an Aavin shift-wise dispatch abstract.

    Example document: GA1301 Madurai District Co-operative Milk Producers' Union
    Abstract table columns: No | Type of Product | QTY (LTR) | QTY (PKT) | TRAY
    """
    row_no: Optional[str] = Field(default=None, description="Serial number / row number (e.g. '1', '6')")
    product_name: Optional[str] = Field(default=None, description="Product name as printed (e.g. 'COW MILK 500 ML')")
    qty_ltr: Optional[float] = Field(default=None, description="Quantity in litres (decimal)")
    qty_pkt: Optional[int] = Field(default=None, description="Quantity in packets (integer count)")
    tray: Optional[float] = Field(default=None, description="Tray count (may be fractional, e.g. 2148.01)")
    is_subtotal: bool = Field(default=False, description="True if this row is a subtotal line (e.g. MILK TOTAL)")
    is_grand_total: bool = Field(default=False, description="True if this row is the GRAND TOTAL line")
    subtotal_label: Optional[str] = Field(default=None, description="Label text for subtotal/total rows")
    confidence: float = Field(default=0.90, description="Extraction confidence score")
    review_required: bool = Field(default=True, description="Human review required flag")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")


class AbstractReportMeta(BaseModel):
    """Header metadata extracted from an Aavin abstract/dispatch report."""
    unit_code: Optional[str] = Field(default=None, description="Plant/unit code (e.g. 'GA1301')")
    unit_name: Optional[str] = Field(default=None, description="Union/plant name")
    address: Optional[str] = Field(default=None, description="Address line from report header")
    report_date: Optional[str] = Field(default=None, description="Report date string (e.g. '21.10.2026')")
    shift: Optional[str] = Field(default=None, description="Shift: MORNING or EVENING")
    report_title: Optional[str] = Field(default=None, description="Report title (e.g. 'ABSTRACT')")
    confidence: float = Field(default=0.85, description="Header extraction confidence")


class AbstractReportSummary(BaseModel):
    """
    Complete structured output for an Aavin Abstract / Dispatch Report image.
    Captures header metadata, all product rows, subtotals, and the grand total.
    """
    meta: AbstractReportMeta = Field(default_factory=AbstractReportMeta, description="Report header metadata")
    products: List[AbstractProductRecord] = Field(default_factory=list, description="Individual product data rows")
    milk_total: Optional[AbstractProductRecord] = Field(default=None, description="Milk category subtotal row")
    curd_butter_total: Optional[AbstractProductRecord] = Field(default=None, description="Curd & Buttermilk subtotal row")
    grand_total: Optional[AbstractProductRecord] = Field(default=None, description="Grand total row")
    total_qty_ltr: Optional[float] = Field(default=None, description="Grand total QTY in Litres")
    total_qty_pkt: Optional[int] = Field(default=None, description="Grand total QTY in Packets")
    total_tray: Optional[float] = Field(default=None, description="Grand total Trays")
    confidence: float = Field(default=0.90, description="Overall report confidence")
    review_required: bool = Field(default=True, description="Review required before saving")
    warnings: List[str] = Field(default_factory=list, description="Report-level warnings")


# ============================================================================
# Main OCR Response Schemas
# ============================================================================

class OCRResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    success: bool = True
    request_id: str = Field(..., description="Unique UUID for request tracing")
    api_version: str = Field(default="v1", description="REST API version")
    engine: EngineInfo = Field(default_factory=EngineInfo, description="OCR engine metadata")
    language_hint: Optional[str] = Field(default=None, description="Language hint used during recognition")
    filename: str = Field(..., description="Name of uploaded image file")
    content_type: str = Field(..., description="MIME content type")
    extracted_text: str = Field(..., description="Full recognized Tamil & mixed-language text")
    text: str = Field(default="", description="Clean plain text format with reading order preserved")
    html: str = Field(default="", description="Safe semantic HTML representation (paragraphs and rendered tables)")
    json: Dict[str, Any] = Field(default_factory=dict, description="Rich structured JSON format with line, word, and table metadata")
    formats: Optional[OCRFormats] = Field(default=None, description="Bundled json, html, and text format outputs")
    tables: List[TableResult] = Field(default_factory=list, description="Detected tabular structures")
    records: List[FarmerCandidateRecord] = Field(default_factory=list, description="Mapped Aavin farmer/cattle candidate records")
    abstract_records: List[AbstractReportSummary] = Field(default_factory=list, description="Structured Aavin abstract/dispatch report records (shift-wise production tables)")
    metadata: ProcessingMetadata
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    errors: List[str] = Field(default_factory=list, description="Non-fatal processing errors or notices")
    detected_languages: List[str] = Field(default_factory=lambda: ["ta"], description="Languages detected across image: ta, en, or mixed")
    unassigned_text: Optional[str] = Field(default=None, description="Text tokens located outside recognized table boundaries")


class TextOnlyResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    success: bool = True
    request_id: str
    api_version: str = Field(default="v1", description="REST API version")
    engine: EngineInfo = Field(default_factory=EngineInfo, description="OCR engine metadata")
    filename: str
    content_type: str
    extracted_text: str
    text: str = Field(default="", description="Clean plain text format")
    html: str = Field(default="", description="Semantic HTML format")
    json: Dict[str, Any] = Field(default_factory=dict, description="Structured JSON format")
    formats: Optional[OCRFormats] = Field(default=None, description="Bundled json, html, and text format outputs")
    metadata: ProcessingMetadata
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class TableOnlyResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    success: bool = True
    request_id: str
    api_version: str = Field(default="v1", description="REST API version")
    engine: EngineInfo = Field(default_factory=EngineInfo, description="OCR engine metadata")
    language_hint: Optional[str] = Field(default=None, description="Language hint used during recognition")
    filename: str
    content_type: str
    tables: List[TableResult]
    # Root level legacy compatibility fields matching TableResult
    table_id: Optional[int] = Field(default=None, description="1-indexed table identifier (legacy root field)")
    headers: List[str] = Field(default_factory=list, description="Column header strings (legacy root field)")
    headers_detail: List[ColumnHeaderDetail] = Field(default_factory=list, description="Detailed column header objects (legacy root field)")
    rows: List[List[str]] = Field(default_factory=list, description="Row values as 2D string array (legacy root field)")
    columns: List[TableColumn] = Field(default_factory=list, description="Structured column definitions (legacy root field)")
    confidence: Optional[float] = Field(default=None, description="Detection confidence score (legacy root field)")
    detection_strategy: Optional[str] = Field(default=None, description="Strategy used: surya_layout, bbox_clustering, morphology_grid (legacy root field)")
    extracted_text: str
    text: str = Field(default="", description="Clean plain text format")
    html: str = Field(default="", description="Semantic HTML format")
    json: Dict[str, Any] = Field(default_factory=dict, description="Structured JSON format")
    formats: Optional[OCRFormats] = Field(default=None, description="Bundled json, html, and text format outputs")
    records: List[FarmerCandidateRecord] = Field(default_factory=list, description="Extracted Aavin domain candidate records")
    abstract_records: List[AbstractReportSummary] = Field(default_factory=list, description="Structured Aavin abstract/dispatch report records")
    metadata: ProcessingMetadata
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class StructuredOCRResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    success: bool = True
    request_id: str
    api_version: str = "v1"
    engine: EngineInfo = Field(default_factory=EngineInfo)
    filename: str
    content_type: str
    language_hint: Optional[str] = None
    document_type: str = Field(default="aavin_register", description="Register profile used for semantic extraction: aavin_register | aavin_abstract")
    tables: List[TableResult] = Field(default_factory=list, description="Detected and reconstructed tables")
    records: List[FarmerCandidateRecord] = Field(default_factory=list, description="Extracted Aavin farmer & cattle candidate records")
    candidates: List[FarmerCandidateRecord] = Field(default_factory=list, description="Alias for records")
    abstract_records: List[AbstractReportSummary] = Field(default_factory=list, description="Structured Aavin abstract/dispatch report records (shift-wise production tables)")
    review_required: bool = Field(default=True, description="Strict review flag requiring explicit operator confirmation before saving to database")
    unassigned_cattle_ids: List[str] = Field(default_factory=list, description="Pashu Aadhaar tag IDs unassociated with a specific farmer")
    metadata: ProcessingMetadata
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# ============================================================================
# Batch OCR Schemas
# ============================================================================

class BatchItemResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    filename: str
    success: bool
    extracted_text: Optional[str] = None
    text: Optional[str] = None
    html: Optional[str] = None
    json: Optional[Dict[str, Any]] = None
    formats: Optional[OCRFormats] = None
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
