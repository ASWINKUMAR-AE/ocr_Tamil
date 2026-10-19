"""
Surya OCR 0.1.2 Engine Service.
Provides line detection, document layout segmentation, and vertical/horizontal affinity line
extraction using the pinned surya-ocr==0.1.2 Segformer architecture (vikp/line_detector).
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch

from app.config import settings

logger = logging.getLogger("tamil_ocr.surya_engine")


class SuryaEngine:
    """
    Singleton wrapper for Surya OCR 0.1.2 Segformer line detector and affinity segmentation.
    Extracts text-line bounding boxes, vertical table column boundaries, and horizontal row boundaries.
    """

    @classmethod
    def get_instance(cls) -> "SuryaEngine":
        """Return the global SuryaEngine singleton instance."""
        return surya_engine

    def __init__(self):
        self._initialized: bool = False
        self._init_error: Optional[str] = None
        self._model = None
        self._processor = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.device = "cuda" if (settings.ENABLE_CUDA and torch.cuda.is_available()) else "cpu"

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazy creation of semaphore bound to current event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_OCR)
        return self._semaphore

    def initialize(self):
        """Initialize Surya 0.1.2 model and image processor synchronously."""
        if self._initialized:
            return

        try:
            logger.info("Initializing Surya OCR 0.1.2 Segformer line detector...")
            from surya.model.segformer import load_model, load_processor

            dtype = torch.float32 if self.device == "cpu" else torch.float16
            self._processor = load_processor(checkpoint=settings.SURYA_DETECTOR_CHECKPOINT)
            self._model = load_model(
                checkpoint=settings.SURYA_DETECTOR_CHECKPOINT,
                device=self.device,
                dtype=dtype
            )

            self._initialized = True
            self._init_error = None
            logger.info(f"Surya OCR 0.1.2 successfully loaded on device: {self.device}")

        except Exception as e:
            self._init_error = str(e)
            self._initialized = False
            logger.error(f"Failed to initialize Surya OCR 0.1.2: {e}", exc_info=True)

    def is_healthy(self) -> bool:
        """Verify that the Surya engine is loaded and ready."""
        return self._initialized and (self._model is not None) and (self._processor is not None)

    @property
    def initialization_error(self) -> Optional[str]:
        return self._init_error

    def detect_layout(
        self,
        image: Union[np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        Run Surya 0.1.2 batch_inference on a single image.
        
        Args:
            image: OpenCV BGR ndarray or PIL Image
            
        Returns:
            Dictionary containing:
                - bboxes: list of [x1, y1, x2, y2] line bounding boxes
                - vertical_lines: list of detected vertical line coordinates (column boundaries)
                - horizontal_lines: list of detected horizontal line coordinates (row boundaries)
                - heatmap: PIL Image of text heatmap
                - affinity_map: PIL Image of affinity map
        """
        if not self.is_healthy():
            self.initialize()
            if not self.is_healthy():
                raise RuntimeError(f"Surya engine is unavailable: {self._init_error}")

        # Convert OpenCV BGR to PIL Image (RGB)
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                pil_img = Image.fromarray(image).convert("RGB")
            elif image.shape[2] == 4:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                pil_img = Image.fromarray(rgb)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Check minimum dimensions
        w, h = pil_img.size
        if w < 16 or h < 16:
            return {
                "bboxes": [],
                "vertical_lines": [],
                "horizontal_lines": [],
                "heatmap": None,
                "affinity_map": None
            }

        try:
            from surya.detection import batch_inference

            results = batch_inference([pil_img], self._model, self._processor)
            if not results or len(results) == 0:
                return {
                    "bboxes": [],
                    "vertical_lines": [],
                    "horizontal_lines": [],
                    "heatmap": None,
                    "affinity_map": None
                }

            res = results[0]
            # Ensure coordinates are standard python ints
            clean_bboxes = []
            for b in res.get("bboxes", []):
                clean_bboxes.append([int(round(coord)) for coord in b])

            clean_vlines = []
            for vl in res.get("vertical_lines", []):
                box = vl.get("bbox") if isinstance(vl, dict) else vl
                if box:
                    clean_vlines.append([int(round(coord)) for coord in box])

            clean_hlines = []
            for hl in res.get("horizontal_lines", []):
                box = hl.get("bbox") if isinstance(hl, dict) else hl
                if box:
                    clean_hlines.append([int(round(coord)) for coord in box])

            return {
                "bboxes": clean_bboxes,
                "vertical_lines": clean_vlines,
                "horizontal_lines": clean_hlines,
                "heatmap": res.get("heatmap"),
                "affinity_map": res.get("affinity_map")
            }

        except Exception as e:
            logger.error(f"Error during Surya layout inference: {e}", exc_info=True)
            raise

    # Friendly alias for line detection
    detect_lines = detect_layout


# Global singleton instance
surya_engine = SuryaEngine()
