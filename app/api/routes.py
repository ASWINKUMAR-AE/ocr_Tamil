"""
FastAPI Router for Tamil OCR Endpoints.
Implements /health, /api/v1/info, /api/v1/ocr, /api/v1/ocr/table, /api/v1/ocr/text, and /api/v1/ocr/batch.
"""

import time
import logging
from typing import List, Optional
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status
)
from fastapi.responses import HTMLResponse, PlainTextResponse

from app.config import settings
from app.schemas.ocr import (
    ApiInfoResponse,
    BatchItemResult,
    BatchOCRResponse,
    EngineInfo,
    HealthResponse,
    OCRFormats,
    OCRResponse,
    ProcessingMetadata,
    StructuredOCRResponse,
    TableOnlyResponse,
    TextOnlyResponse,
)
from app.services.aavin_mapping import aavin_mapper
from app.services.format_builder import build_output_formats
from app.services.image_preprocessing import image_preprocessor
from app.services.ocr_service import ocr_service
from app.services.table_extraction import table_extractor
from app.utils.network import get_network_summary
from app.utils.security import generate_request_id, validate_image_bytes, verify_api_key

logger = logging.getLogger("tamil_ocr.routes")
router = APIRouter()


# ============================================================================
# Health Check Endpoint
# ============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["Diagnostics"]
)
async def health_check():
    """
    Check server and OCR engine status.
    Returns 200 OK when the engine is loaded and functioning, or 503 if degraded.
    """
    if not ocr_service.is_healthy():
        # Attempt lazy initialization if not already done
        ocr_service.initialize()

    if not ocr_service.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "service": settings.APP_NAME,
                "ocr_engine": settings.OCR_ENGINE_NAME,
                "version": settings.APP_VERSION,
                "error": ocr_service.initialization_error or "OCR engine failed to initialize",
            }
        )

    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        ocr_engine=settings.OCR_ENGINE_NAME,
        version=settings.APP_VERSION,
    )


# ============================================================================
# API Information Endpoint
# ============================================================================

@router.get(
    "/api/v1/info",
    response_model=ApiInfoResponse,
    summary="API Information & Capabilities",
    tags=["Diagnostics"],
    dependencies=[Depends(verify_api_key)]
)
async def get_api_info():
    """
    Return server metadata, enabled capabilities, limits, and dynamic network URLs.
    """
    net_summary = get_network_summary(settings.HOST, settings.PORT)

    return ApiInfoResponse(
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        ocr_engine=settings.OCR_ENGINE_NAME,
        supported_image_formats=["image/jpeg", "image/png", "image/webp"],
        max_upload_size_mb=settings.MAX_UPLOAD_SIZE_MB,
        max_image_dimension=settings.MAX_IMAGE_DIMENSION,
        max_batch_size=settings.MAX_BATCH_SIZE,
        auth_enabled=settings.is_auth_enabled,
        capabilities=[
            "tamil_text_recognition",
            "english_text_recognition",
            "surya_line_detection",
            "surya_affinity_layout",
            "table_structure_detection",
            "cell_level_extraction",
            "aavin_cattle_passport_extraction",
            "pashu_aadhaar_validation",
            "review_first_workflow",
            "orientation_correction",
            "deskewing",
            "contrast_enhancement",
            "json_format",
            "html_format",
            "text_format"
        ],
        network=net_summary,
    )


# ============================================================================
# Main OCR Endpoint (Text + Tables)
# ============================================================================

@router.post(
    "/api/v1/ocr",
    response_model=OCRResponse,
    summary="Full OCR and Table Extraction",
    tags=["OCR"],
    dependencies=[Depends(verify_api_key)]
)
async def process_ocr(
    response: Response,
    file: UploadFile = File(..., description="Uploaded image file (JPEG, PNG, WebP)"),
    extract_tables: bool = Form(default=True, description="Attempt table structure detection"),
    preprocess: bool = Form(default=True, description="Apply image preprocessing"),
    language_hint: Optional[str] = Form(default=None, description="Optional language hint (ta/en)"),
    format: str = Form(default="json", description="Output format: 'json' (default), 'html', 'text', or 'all'")
):
    """
    Process an uploaded image:
    1. Validates format and byte safety.
    2. Corrects EXIF orientation and deskews.
    3. Runs Tamil/English OCR with reading order and line breaks preserved.
    4. Detects tables and returns structured JSON rows and columns.
    5. Formats output into JSON, HTML, and TEXT formats.
    """
    req_id = generate_request_id()
    response.headers["X-Request-ID"] = req_id
    start_time = time.perf_counter()

    # Read and validate bytes
    content = await file.read()
    is_valid, err_msg = validate_image_bytes(content, file.filename or "upload", file.content_type)
    if not is_valid:
        status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if "exceeds" in err_msg else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail={"request_id": req_id, "error": err_msg})

    try:
        # Preprocess image
        processed_bgr, original_bgr, prep_meta = image_preprocessor.process(
            content, enable_preprocess=preprocess
        )
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"request_id": req_id, "error": f"Failed to decode or process image: {str(e)}"}
        )

    # Concurrency control around heavy PyTorch OCR
    async with ocr_service.semaphore:
        try:
            # Run text extraction with spatial word details
            extracted_text, words = ocr_service.predict_with_details(
                processed_bgr, detect=True, preserve_lines=True, language_hint=language_hint
            )
        except Exception as e:
            logger.error(f"OCR inference failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"request_id": req_id, "error": f"OCR processing failed: {str(e)}"}
            )

        # Run multi-strategy table extraction if enabled
        tables = []
        warnings = []
        if extract_tables:
            try:
                tables, table_warnings = table_extractor.extract_tables(
                    processed_bgr,
                    words=words,
                    extracted_text=extracted_text,
                    extract_text=True
                )
                warnings.extend(table_warnings)
            except Exception as e:
                logger.warning(f"Table extraction error: {e}")
                warnings.append(f"Table extraction warning: {str(e)}")

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    doc_langs = ocr_service.detect_document_languages(words)
    unassigned = None
    if tables and hasattr(tables[0], "unassigned_text") and tables[0].unassigned_text:
        unassigned = tables[0].unassigned_text

    meta_dict = {
        "ocr_engine": settings.OCR_ENGINE_NAME,
        "preprocessing_applied": prep_meta.get("preprocessing_applied", False),
        "processing_time_ms": elapsed_ms,
        "image_dimensions": {
            "width": prep_meta.get("original_width", 0),
            "height": prep_meta.get("original_height", 0)
        }
    }
    text_out, html_out, json_out, formats_bundle = build_output_formats(
        extracted_text=extracted_text,
        tables=tables,
        filename=file.filename or "image",
        metadata=meta_dict
    )

    fmt = (format or "json").strip().lower()
    if fmt in ("html", "text/html"):
        return HTMLResponse(content=html_out, media_type="text/html; charset=utf-8")
    if fmt in ("text", "txt", "text/plain"):
        return PlainTextResponse(content=text_out, media_type="text/plain; charset=utf-8")

    first_tbl = tables[0] if tables else None
    cand_records, _ = aavin_mapper.extract_candidate_records(first_tbl) if first_tbl else ([], [])

    return OCRResponse(
        success=True,
        request_id=req_id,
        api_version="v1",
        engine=EngineInfo(name=settings.OCR_ENGINE_NAME, version=settings.SURYA_VERSION),
        language_hint=language_hint,
        filename=file.filename or "image",
        content_type=file.content_type or "image/png",
        extracted_text=extracted_text,
        text=text_out,
        html=html_out,
        json=json_out,
        formats=OCRFormats(**formats_bundle),
        tables=tables,
        records=cand_records,
        metadata=ProcessingMetadata(**meta_dict),
        warnings=warnings,
        errors=[],
        detected_languages=doc_langs,
        unassigned_text=unassigned
    )


# ============================================================================
# Table Extraction Endpoint
# ============================================================================

@router.post(
    "/api/v1/ocr/table",
    response_model=TableOnlyResponse,
    summary="Dedicated Table Extraction",
    tags=["OCR"],
    dependencies=[Depends(verify_api_key)]
)
async def extract_table_only(
    response: Response,
    file: UploadFile = File(..., description="Uploaded image containing table(s)"),
    preprocess: bool = Form(default=True, description="Apply image preprocessing"),
    language_hint: Optional[str] = Form(default=None, description="Optional language hint (ta/en)"),
    format: str = Form(default="json", description="Output format: 'json' (default), 'html', 'text', or 'all'")
):
    """
    Dedicated endpoint for table recognition.
    Returns structured rows, column definitions, key-value data mappings, and full text in JSON, HTML, and TEXT.
    """
    req_id = generate_request_id()
    response.headers["X-Request-ID"] = req_id
    start_time = time.perf_counter()

    content = await file.read()
    is_valid, err_msg = validate_image_bytes(content, file.filename or "upload", file.content_type)
    if not is_valid:
        status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if "exceeds" in err_msg else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail={"request_id": req_id, "error": err_msg})

    try:
        processed_bgr, _, prep_meta = image_preprocessor.process(content, enable_preprocess=preprocess)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"request_id": req_id, "error": f"Failed to decode image: {str(e)}"}
        )

    async with ocr_service.semaphore:
        extracted_text, words = ocr_service.predict_with_details(
            processed_bgr, detect=True, preserve_lines=True, language_hint=language_hint
        )
        tables, warnings = table_extractor.extract_tables(
            processed_bgr,
            words=words,
            extracted_text=extracted_text,
            extract_text=True
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    meta_dict = {
        "ocr_engine": settings.OCR_ENGINE_NAME,
        "preprocessing_applied": prep_meta.get("preprocessing_applied", False),
        "processing_time_ms": elapsed_ms,
        "image_dimensions": {
            "width": prep_meta.get("original_width", 0),
            "height": prep_meta.get("original_height", 0)
        }
    }
    text_out, html_out, json_out, formats_bundle = build_output_formats(
        extracted_text=extracted_text,
        tables=tables,
        filename=file.filename or "image",
        metadata=meta_dict
    )

    fmt = (format or "json").strip().lower()
    if fmt in ("html", "text/html"):
        return HTMLResponse(content=html_out, media_type="text/html; charset=utf-8")
    if fmt in ("text", "txt", "text/plain"):
        return PlainTextResponse(content=text_out, media_type="text/plain; charset=utf-8")

    first_tbl = tables[0] if tables else None
    cand_records, _ = aavin_mapper.extract_candidate_records(first_tbl) if first_tbl else ([], [])

    return TableOnlyResponse(
        success=True,
        request_id=req_id,
        api_version="v1",
        engine=EngineInfo(name=settings.OCR_ENGINE_NAME, version=settings.SURYA_VERSION),
        language_hint=language_hint,
        filename=file.filename or "image",
        content_type=file.content_type or "image/png",
        tables=tables,
        table_id=first_tbl.table_id if first_tbl else None,
        headers=first_tbl.headers if first_tbl else [],
        headers_detail=first_tbl.headers_detail if first_tbl else [],
        rows=first_tbl.rows if first_tbl else [],
        columns=first_tbl.columns if first_tbl else [],
        confidence=first_tbl.confidence if first_tbl else None,
        detection_strategy=first_tbl.detection_strategy if first_tbl else None,
        records=cand_records,
        extracted_text=extracted_text,
        text=text_out,
        html=html_out,
        json=json_out,
        formats=OCRFormats(**formats_bundle),
        metadata=ProcessingMetadata(**meta_dict),
        warnings=warnings,
        errors=[]
    )


# ============================================================================
# Structured Aavin Extraction Endpoint
# ============================================================================

@router.post(
    "/api/v1/ocr/structured",
    response_model=StructuredOCRResponse,
    summary="Aavin Register Table Intelligence & Candidate Record Extraction",
    tags=["OCR"],
    dependencies=[Depends(verify_api_key)]
)
async def extract_structured_aavin(
    response: Response,
    file: UploadFile = File(..., description="Uploaded register image file (JPEG, PNG, WebP)"),
    document_type: str = Form(default="aavin_register", description="Register profile type"),
    preprocess: bool = Form(default=True, description="Apply image preprocessing"),
    language_hint: Optional[str] = Form(default="ta,en", description="Language hint: ta, en, or ta,en"),
    schema_hint: Optional[str] = Form(default=None, description="Optional JSON schema hint e.g. {'0': 'member_id', '1': 'farmer_name'}")
):
    """
    Structured extraction endpoint for Aavin register tables:
    1. Validates and preprocesses uploaded register image.
    2. Runs Surya 0.1.2 layout & affinity line detection + Tamil/English character recognition.
    3. Reconstructs table geometry (rows, columns, cells) with continuation row handling.
    4. Maps Tamil and English headers to Aavin Cattle Passport domain entities.
    5. Returns review-first candidate records for farmers, 12-digit cattle IDs, and milk quantities.
    """
    req_id = generate_request_id()
    response.headers["X-Request-ID"] = req_id
    start_time = time.perf_counter()

    content = await file.read()
    is_valid, err_msg = validate_image_bytes(content, file.filename or "upload", file.content_type)
    if not is_valid:
        status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if "exceeds" in err_msg else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail={"request_id": req_id, "error": err_msg})

    try:
        processed_bgr, _, prep_meta = image_preprocessor.process(content, enable_preprocess=preprocess)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"request_id": req_id, "error": f"Failed to decode image: {str(e)}"}
        )

    parsed_schema_hint = None
    if schema_hint:
        try:
            import json
            parsed_schema_hint = json.loads(schema_hint)
        except Exception:
            logger.debug(f"Failed to parse schema_hint as JSON: {schema_hint}")

    async with ocr_service.semaphore:
        extracted_text, words = ocr_service.predict_with_details(
            processed_bgr, detect=True, preserve_lines=True, language_hint=language_hint
        )
        tables, warnings = table_extractor.extract_tables(
            processed_bgr,
            words=words,
            extracted_text=extracted_text,
            extract_text=True,
            schema_hint=parsed_schema_hint
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    meta_dict = {
        "ocr_engine": settings.OCR_ENGINE_NAME,
        "preprocessing_applied": prep_meta.get("preprocessing_applied", False),
        "processing_time_ms": elapsed_ms,
        "image_dimensions": {
            "width": prep_meta.get("original_width", 0),
            "height": prep_meta.get("original_height", 0)
        }
    }

    all_records = []
    all_unassigned = []
    for tbl in tables:
        tbl_records, tbl_unassigned = aavin_mapper.extract_candidate_records(tbl, schema_hint=parsed_schema_hint)
        all_records.extend(tbl_records)
        all_unassigned.extend(tbl_unassigned)

    return StructuredOCRResponse(
        success=True,
        request_id=req_id,
        api_version="v1",
        engine=EngineInfo(name=settings.OCR_ENGINE_NAME, version=settings.SURYA_VERSION),
        filename=file.filename or "image",
        content_type=file.content_type or "image/png",
        language_hint=language_hint,
        document_type=document_type,
        tables=tables,
        records=all_records,
        candidates=all_records,
        review_required=True,
        unassigned_cattle_ids=all_unassigned,
        metadata=ProcessingMetadata(**meta_dict),
        warnings=warnings,
        errors=[]
    )


# ============================================================================
# Plain Text Endpoint
# ============================================================================

@router.post(
    "/api/v1/ocr/text",
    response_model=TextOnlyResponse,
    summary="Plain Text Extraction Only",
    tags=["OCR"],
    dependencies=[Depends(verify_api_key)]
)
async def extract_text_only(
    response: Response,
    file: UploadFile = File(..., description="Uploaded image file"),
    preprocess: bool = Form(default=True, description="Apply image preprocessing"),
    format: str = Form(default="json", description="Output format: 'json' (default), 'html', 'text', or 'all'")
):
    """
    Extract only plain text and basic execution metadata.
    Provides fast JSON, HTML, and TEXT representations.
    """
    req_id = generate_request_id()
    response.headers["X-Request-ID"] = req_id
    start_time = time.perf_counter()

    content = await file.read()
    is_valid, err_msg = validate_image_bytes(content, file.filename or "upload", file.content_type)
    if not is_valid:
        status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if "exceeds" in err_msg else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail={"request_id": req_id, "error": err_msg})

    try:
        processed_bgr, _, prep_meta = image_preprocessor.process(content, enable_preprocess=preprocess)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"request_id": req_id, "error": f"Failed to decode image: {str(e)}"}
        )

    async with ocr_service.semaphore:
        extracted_text = ocr_service.predict_text(processed_bgr, detect=True, preserve_lines=True)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    meta_dict = {
        "ocr_engine": settings.OCR_ENGINE_NAME,
        "preprocessing_applied": prep_meta.get("preprocessing_applied", False),
        "processing_time_ms": elapsed_ms,
        "image_dimensions": {
            "width": prep_meta.get("original_width", 0),
            "height": prep_meta.get("original_height", 0)
        }
    }
    text_out, html_out, json_out, formats_bundle = build_output_formats(
        extracted_text=extracted_text,
        tables=[],
        filename=file.filename or "image",
        metadata=meta_dict
    )

    fmt = (format or "json").strip().lower()
    if fmt in ("html", "text/html"):
        return HTMLResponse(content=html_out, media_type="text/html; charset=utf-8")
    if fmt in ("text", "txt", "text/plain"):
        return PlainTextResponse(content=text_out, media_type="text/plain; charset=utf-8")

    return TextOnlyResponse(
        success=True,
        request_id=req_id,
        filename=file.filename or "image",
        content_type=file.content_type or "image/png",
        extracted_text=extracted_text,
        text=text_out,
        html=html_out,
        json=json_out,
        formats=OCRFormats(**formats_bundle),
        metadata=ProcessingMetadata(**meta_dict),
        warnings=[]
    )


# ============================================================================
# Batch OCR Endpoint
# ============================================================================

@router.post(
    "/api/v1/ocr/batch",
    response_model=BatchOCRResponse,
    summary="Batch OCR on Multiple Images",
    tags=["OCR"],
    dependencies=[Depends(verify_api_key)]
)
async def process_batch(
    response: Response,
    files: List[UploadFile] = File(..., description="Array of image files"),
    extract_tables: bool = Form(default=False, description="Extract tables from each image"),
    preprocess: bool = Form(default=True, description="Apply image preprocessing")
):
    """
    Process multiple images in a single batch request.
    Provides isolated error handling per image and memory-safe concurrency throttling.
    """
    req_id = generate_request_id()
    response.headers["X-Request-ID"] = req_id
    total_start = time.perf_counter()

    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"request_id": req_id, "error": "No files provided in batch upload."}
        )

    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "request_id": req_id,
                "error": f"Batch size {len(files)} exceeds maximum allowed limit of {settings.MAX_BATCH_SIZE}."
            }
        )

    results: List[BatchItemResult] = []
    success_count = 0
    failure_count = 0

    for upload_file in files:
        fname = upload_file.filename or "unnamed_image"
        try:
            content = await upload_file.read()
            is_valid, err_msg = validate_image_bytes(content, fname, upload_file.content_type)
            if not is_valid:
                results.append(BatchItemResult(
                    filename=fname,
                    success=False,
                    error=err_msg
                ))
                failure_count += 1
                continue

            item_start = time.perf_counter()
            processed_bgr, _, prep_meta = image_preprocessor.process(
                content, enable_preprocess=preprocess
            )

            async with ocr_service.semaphore:
                text, words = ocr_service.predict_with_details(
                    processed_bgr, detect=True, preserve_lines=True
                )
                tables = []
                warnings = []
                if extract_tables:
                    tables, twarns = table_extractor.extract_tables(
                        processed_bgr,
                        words=words,
                        extracted_text=text,
                        extract_text=True
                    )
                    warnings.extend(twarns)

            item_ms = round((time.perf_counter() - item_start) * 1000, 2)
            item_meta = {
                "ocr_engine": settings.OCR_ENGINE_NAME,
                "preprocessing_applied": prep_meta.get("preprocessing_applied", False),
                "processing_time_ms": item_ms,
                "image_dimensions": {
                    "width": prep_meta.get("original_width", 0),
                    "height": prep_meta.get("original_height", 0)
                }
            }
            text_out, html_out, json_out, formats_bundle = build_output_formats(
                extracted_text=text,
                tables=tables,
                filename=fname,
                metadata=item_meta
            )
            results.append(BatchItemResult(
                filename=fname,
                success=True,
                extracted_text=text,
                text=text_out,
                html=html_out,
                json=json_out,
                formats=OCRFormats(**formats_bundle),
                tables=tables,
                metadata=ProcessingMetadata(**item_meta),
                warnings=warnings
            ))
            success_count += 1

        except Exception as e:
            logger.error(f"Error processing batch item {fname}: {e}")
            results.append(BatchItemResult(
                filename=fname,
                success=False,
                error=f"Processing failed: {str(e)}"
            ))
            failure_count += 1

    total_ms = round((time.perf_counter() - total_start) * 1000, 2)

    return BatchOCRResponse(
        success=True,
        request_id=req_id,
        total_images=len(files),
        successful_images=success_count,
        failed_images=failure_count,
        results=results,
        total_processing_time_ms=total_ms
    )
