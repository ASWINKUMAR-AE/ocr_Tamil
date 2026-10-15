"""
OCR Service module wrapping ocr-tamil library.
Provides thread-safe initialization, concurrency throttling, line-preserving text extraction,
and cell-level recognition for tabular data.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from app.config import settings

logger = logging.getLogger("tamil_ocr.service")


class OCRService:
    """Singleton service wrapping the verified ocr-tamil engine."""

    def __init__(self):
        self._initialized = False
        self._init_error: Optional[str] = None
        self._ocr_detect = None
        self._ocr_recognize = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.device = "cuda" if (settings.ENABLE_CUDA and torch.cuda.is_available()) else "cpu"

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazy creation of semaphore bound to current event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_OCR)
        return self._semaphore

    def _resolve_model_paths(self) -> Tuple[str, str]:
        """Locate verified model weights from known system and package paths."""
        home_weights = os.path.join(Path.home(), ".model_weights")
        tamil_candidates = [
            os.path.join(home_weights, "parseq_tamil_v6.ckpt"),
            os.path.join(home_weights, "parseq_tamil.pt"),
            os.path.join(os.getcwd(), "model_weights", "parseq_tamil_v6.ckpt"),
            os.path.join(os.getcwd(), "model_weights", "parseq_tamil.pt"),
        ]
        detect_candidates = [
            os.path.join(home_weights, "craft_mlt_25k.pth"),
            os.path.join(os.getcwd(), "model_weights", "craft_mlt_25k.pth"),
        ]

        tamil_path = next((p for p in tamil_candidates if os.path.exists(p)), None)
        detect_path = next((p for p in detect_candidates if os.path.exists(p)), None)

        if not tamil_path:
            # Fallback to default in home weights if not pre-populated
            tamil_path = os.path.join(home_weights, "parseq_tamil_v6.ckpt")
        if not detect_path:
            detect_path = os.path.join(home_weights, "craft_mlt_25k.pth")

        return tamil_path, detect_path

    def initialize(self):
        """Initialize the OCR models synchronously."""
        if self._initialized:
            return

        try:
            logger.info("Initializing ocr-tamil engine...")
            from ocr_tamil.ocr import OCR

            tamil_ckpt, craft_pth = self._resolve_model_paths()
            enable_cuda = (self.device == "cuda")

            logger.info(f"Using device: {self.device}, CUDA: {enable_cuda}")
            logger.info(f"Tamil model: {tamil_ckpt}")
            logger.info(f"Detector model: {craft_pth}")

            # Instance for full image text detection + recognition
            self._ocr_detect = OCR(
                detect=True,
                tamil_model_path=tamil_ckpt,
                detect_model_path=craft_pth,
                enable_cuda=enable_cuda
            )

            # Instance for single word/cell crop recognition
            self._ocr_recognize = OCR(
                detect=False,
                tamil_model_path=tamil_ckpt,
                detect_model_path=craft_pth,
                enable_cuda=enable_cuda
            )

            self._initialized = True
            self._init_error = None
            logger.info("ocr-tamil engine successfully initialized.")

        except Exception as e:
            self._init_error = str(e)
            self._initialized = False
            logger.error(f"Failed to initialize ocr-tamil engine: {e}", exc_info=True)

    def is_healthy(self) -> bool:
        """Verify that the OCR engine is ready for inference."""
        return self._initialized and (self._ocr_detect is not None)

    @property
    def initialization_error(self) -> Optional[str]:
        return self._init_error

    def recognize_crop(self, crop_img: np.ndarray) -> str:
        """
        Recognize text from a single cropped word or table cell.
        Returns recognized string (Tamil/English/numeric).
        """
        if not self.is_healthy():
            raise RuntimeError(f"OCR engine is not initialized: {self._init_error}")

        if crop_img is None or crop_img.size == 0:
            return ""

        try:
            # If crop is too thin or tiny, skip
            if crop_img.shape[0] < 5 or crop_img.shape[1] < 5:
                return ""

            result = self._ocr_recognize.predict(crop_img)
            return (result or "").strip()
        except Exception as e:
            logger.warning(f"Error recognizing cell crop: {e}")
            return ""

    def predict_text(
        self,
        image: np.ndarray,
        detect: bool = True,
        preserve_lines: bool = True
    ) -> str:
        """
        Perform OCR on an image.
        If detect=True, runs CRAFT text detection to find regions, then recognizes each.
        If preserve_lines=True, groups bounding boxes into logical lines with newlines.
        """
        if not self.is_healthy():
            raise RuntimeError(f"OCR engine is not initialized: {self._init_error}")

        if image is None or image.size == 0:
            return ""

        if not detect:
            return self.recognize_crop(image)

        # Full image detection + recognition
        try:
            if not preserve_lines:
                raw_pred = self._ocr_detect.predict(image)
                return (raw_pred or "").strip()

            # Line-preserving extraction
            from ocr_tamil.craft_text_detector import get_prediction, export_detected_regions

            # Normalize size within safe bounds
            size = min(max(image.shape[0], image.shape[1], 1280), 2560)
            prediction_result = get_prediction(
                image=image,
                craft_net=self._ocr_detect.craft_net,
                text_threshold=0.7,
                link_threshold=0.25,
                low_text=0.40,
                cuda=(self.device == "cuda"),
                long_size=size,
                poly=False
            )

            if len(prediction_result) == 0:
                # No text detected by CRAFT
                return ""

            # Extract bounding boxes
            new_bbox = []
            for bb in prediction_result:
                xs, ys = bb[:, 0], bb[:, 1]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                new_bbox.append([min_x, min_y, max_x - min_x, max_y - min_y])

            # Line clustering
            c = np.array(new_bbox)
            max_height = np.median(c[:, 3]) * 0.5
            by_y = sorted(new_bbox, key=lambda x: x[1])

            line_y = by_y[0][1]
            line_idx = 1
            by_line = []
            for x, y, w, h in by_y:
                if y > line_y + max_height:
                    line_y = y
                    line_idx += 1
                by_line.append((line_idx, x, y, w, h))

            # Sort by line first, then left-to-right (x)
            sorted_line_items = sorted(by_line, key=lambda item: (item[0], item[1]))

            # Map sorted items back to prediction polygons
            updated_prediction_result = []
            line_numbers = []
            for line_no, x, y, w, h in sorted_line_items:
                orig_idx = new_bbox.index([x, y, w, h])
                updated_prediction_result.append(prediction_result[orig_idx])
                line_numbers.append(line_no)

            # Export region crops
            exported_regions = export_detected_regions(
                image=image,
                regions=updated_prediction_result,
                output_dir="temp_images",
                rectify=False
            )

            # Recognize crops and build lines
            lines: List[List[str]] = []
            current_line_no = None
            current_line_words: List[str] = []

            for line_no, crop in zip(line_numbers, exported_regions):
                word = self.recognize_crop(crop)
                if not word:
                    continue

                if current_line_no is None:
                    current_line_no = line_no

                if line_no != current_line_no:
                    if current_line_words:
                        lines.append(current_line_words)
                    current_line_words = [word]
                    current_line_no = line_no
                else:
                    current_line_words.append(word)

            if current_line_words:
                lines.append(current_line_words)

            formatted_text = "\n".join(" ".join(w for w in line) for line in lines)
            return formatted_text.strip()

        except Exception as e:
            logger.warning(f"Line preservation encountered issue, falling back to standard predict: {e}")
            try:
                raw_pred = self._ocr_detect.predict(image)
                return (raw_pred or "").strip()
            except Exception as e2:
                logger.error(f"Fallback predict failed: {e2}")
                return ""


ocr_service = OCRService()
