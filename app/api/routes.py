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

from app.config import settings
from app.schemas.ocr import (
    ApiInfoResponse,
    BatchItemResult,
    BatchOCRResponse,
    HealthResponse,
    OCRResponse,
    ProcessingMetadata,
    TableOnlyResponse,
    TextOnlyResponse,
)
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
            "natural_scene_text_detection",
            "line_order_preservation",
            "table_structure_detection",
            "cell_level_extraction",
            "orientation_correction",
            "deskewing",
            "contrast_enhancement"
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
    language_hint: Optional[str] = Form(default=None, description="Optional language hint (ta/en)")
):
    """
    Process an uploaded image:
    1. Validates format and byte safety.
    2. Corrects EXIF orientation and deskews.
    3. Runs Tamil/English OCR with reading order and line breaks preserved.
    4. Detects tables and returns structured JSON rows and columns.
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

    return OCRResponse(
        success=True,
        request_id=req_id,
        filename=file.filename or "image",
        content_type=file.content_type or "image/png",
        extracted_text=extracted_text,
        tables=tables,
        metadata=ProcessingMetadata(
            ocr_engine=settings.OCR_ENGINE_NAME,
            preprocessing_applied=prep_meta.get("preprocessing_applied", False),
            processing_time_ms=elapsed_ms,
            image_dimensions={
                "width": prep_meta.get("original_width", 0),
                "height": prep_meta.get("original_height", 0)
            }
        ),
        warnings=warnings,
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
    language_hint: Optional[str] = Form(default=None, description="Optional language hint (ta/en)")
):
    """
    Dedicated endpoint for table recognition.
    Returns structured rows, column definitions, key-value data mappings, and full text.
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

    return TableOnlyResponse(
        success=True,
        request_id=req_id,
        filename=file.filename or "image",
        content_type=file.content_type or "image/png",
        tables=tables,
        extracted_text=extracted_text,
        metadata=ProcessingMetadata(
            ocr_engine=settings.OCR_ENGINE_NAME,
            preprocessing_applied=prep_meta.get("preprocessing_applied", False),
            processing_time_ms=elapsed_ms,
            image_dimensions={
                "width": prep_meta.get("original_width", 0),
                "height": prep_meta.get("original_height", 0)
            }
        ),
        warnings=warnings
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
    preprocess: bool = Form(default=True, description="Apply image preprocessing")
):
    """
    Extract only plain text and basic execution metadata without running table parsing.
    Ideal for mobile scanners and speed-critical document recognition.
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

    return TextOnlyResponse(
        success=True,
        request_id=req_id,
        filename=file.filename or "image",
        content_type=file.content_type or "image/png",
        extracted_text=extracted_text,
        metadata=ProcessingMetadata(
            ocr_engine=settings.OCR_ENGINE_NAME,
            preprocessing_applied=prep_meta.get("preprocessing_applied", False),
            processing_time_ms=elapsed_ms,
            image_dimensions={
                "width": prep_meta.get("original_width", 0),
                "height": prep_meta.get("original_height", 0)
            }
        ),
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
            results.append(BatchItemResult(
                filename=fname,
                success=True,
                extracted_text=text,
                tables=tables,
                metadata=ProcessingMetadata(
                    ocr_engine=settings.OCR_ENGINE_NAME,
                    preprocessing_applied=prep_meta.get("preprocessing_applied", False),
                    processing_time_ms=item_ms,
                    image_dimensions={
                        "width": prep_meta.get("original_width", 0),
                        "height": prep_meta.get("original_height", 0)
                    }
                ),
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
