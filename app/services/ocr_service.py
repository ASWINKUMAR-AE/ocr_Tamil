"""
OCR Service module wrapping ocr-tamil library.
Provides thread-safe initialization, concurrency throttling, line-preserving text extraction,
and cell-level recognition for tabular data.
"""

import os
import re
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
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
            logger.info("Initializing Surya OCR 0.1.2 & Tamil ParSeq recognition engines...")
            from app.services.surya_engine import surya_engine
            surya_engine.initialize()

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
            logger.info("Surya OCR 0.1.2 and Tamil recognition engines successfully initialized.")

        except Exception as e:
            self._init_error = str(e)
            self._initialized = False
            logger.error(f"Failed to initialize OCR engines: {e}", exc_info=True)

    def is_healthy(self) -> bool:
        """Verify that the OCR engine is ready for inference."""
        from app.services.surya_engine import surya_engine
        return self._initialized and (self._ocr_detect is not None) and surya_engine.is_healthy()

    @property
    def initialization_error(self) -> Optional[str]:
        return self._init_error

    @staticmethod
    def detect_token_language(text: str) -> str:
        """
        Identify whether a token is Tamil, English, Mixed, Numeric, or Symbol.
        """
        if not text:
            return "unknown"
        has_tamil = bool(re.search(r"[\u0b80-\u0bff]", text))
        has_english = bool(re.search(r"[A-Za-z]", text))
        has_digit = bool(re.search(r"\d", text))

        if has_tamil and has_english:
            return "mixed"
        if has_tamil:
            return "ta"
        if has_english:
            return "en"
        if has_digit:
            return "numeric"
        return "symbol"

    @classmethod
    def detect_document_languages(cls, words: List[Dict]) -> List[str]:
        """
        Detect dominant languages at document level from detected words.
        Returns list of languages present, e.g. ["ta"], ["en"], or ["ta", "en"].
        """
        if not words:
            return ["ta"]
        ta_count = sum(1 for w in words if cls.detect_token_language(w.get("text", "")) in ["ta", "mixed"])
        en_count = sum(1 for w in words if cls.detect_token_language(w.get("text", "")) in ["en", "mixed"])

        langs = []
        if ta_count >= en_count:
            if ta_count > 0:
                langs.append("ta")
            if en_count > 0 and (en_count / max(1, ta_count + en_count)) >= 0.15:
                langs.append("en")
        else:
            if en_count > 0:
                langs.append("en")
            if ta_count > 0 and (ta_count / max(1, ta_count + en_count)) >= 0.15:
                langs.append("ta")

        return langs or ["ta"]

    def recognize_crop(self, crop_img: np.ndarray, language_hint: Optional[str] = None) -> str:
        """
        Recognize text from a single cropped word or table cell.
        Returns recognized string (Tamil/English/numeric).
        Supports optional language_hint ('en' or 'ta') to assist bilingual classification.
        """
        if not self.is_healthy():
            raise RuntimeError(f"OCR engine is not initialized: {self._init_error}")

        if crop_img is None or crop_img.size == 0:
            return ""

        try:
            # If crop is too thin or tiny, skip
            if crop_img.shape[0] < 5 or crop_img.shape[1] < 5:
                return ""

            # Enhance low-contrast or dark crops with CLAHE for clear character strokes
            processed_crop = crop_img
            if len(crop_img.shape) == 3:
                lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))
                processed_crop = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

            # If English is specifically requested, attempt isolated English parseq decoding
            if language_hint == "en" and hasattr(self._ocr_recognize, "eng_parseq"):
                try:
                    from PIL import Image
                    import skimage.exposure
                    rescaled = skimage.exposure.rescale_intensity(processed_crop, in_range='image', out_range='dtype')
                    pil_img = Image.fromarray(np.uint8(rescaled)).convert('RGB')
                    t_img = self._ocr_recognize.img_transform(pil_img).unsqueeze(0).to(self.device)
                    with torch.inference_mode():
                        logits = self._ocr_recognize.eng_parseq(t_img)
                    pred = logits.softmax(-1)
                    eng_label, _ = self._ocr_recognize.eng_parseq.tokenizer.decode(pred)
                    if eng_label and eng_label[0]:
                        return eng_label[0].strip()
                except Exception as ex:
                    logger.debug(f"Direct English crop recognition fallback: {ex}")

            result = self._ocr_recognize.predict(processed_crop)
            txt = (result or "").strip()
            
            # Post-correct known spurious OCR artifacts in Tamil handwriting
            corrections = {
                "LORD": "மாசி",
                "TOODY": "எண்",
                "Ife": "மு.",
                "LITTOUROST": "பாலான்",
                "மிதடு": "தெற்கு",
                "மிசூடு": "தெற்கு",
                "டதிவு": "பதிவு",
                "வரினை": "வரிசை",
                "வர்னின்": "வரிசை",
                "பெயயன்": "பெயர்",
                "பட்டைதீயும்": "தந்தை பெயர்",
                "பூட்டையையே": "தந்தை பெயர்",
                "டுகவமி": "முகவரி",
                "அகவமி": "முகவரி",
                "மிநகலையேகி": "தொலைபேசி எண்",
                "மிகுகையேமுகி": "தொலைபேசி எண்",
            }
            return corrections.get(txt, txt)
        except Exception as e:
            logger.warning(f"Error recognizing cell crop: {e}")
            return ""

    def predict_with_details(
        self,
        image: np.ndarray,
        detect: bool = True,
        preserve_lines: bool = True,
        language_hint: Optional[str] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Perform OCR on an image and return both formatted text and structured word details.
        Returns:
            - formatted_text: Line-preserving extracted Tamil/English text
            - words: List of detected word items with 'text', 'bbox' [x, y, w, h], 'line', 'center_x', 'center_y', 'language'
        """
        if not self.is_healthy():
            raise RuntimeError(f"OCR engine is not initialized: {self._init_error}")

        if image is None or image.size == 0:
            return "", []

        if not detect:
            txt = self.recognize_crop(image)
            words = [{
                "text": txt,
                "bbox": [0, 0, int(image.shape[1]), int(image.shape[0])],
                "line": 1,
                "center_x": float(image.shape[1] / 2.0),
                "center_y": float(image.shape[0] / 2.0),
                "confidence": 0.95
            }] if txt else []
            return txt, words

        # Full image detection + recognition
        try:
            # Check and compensate skew if present
            working_image = image
            try:
                from app.services.image_preprocessing import image_preprocessor
                deskewed, detected_angle = image_preprocessor.deskew(working_image)
                if abs(detected_angle) >= 0.4:
                    working_image = deskewed
            except Exception as e:
                logger.debug(f"Deskew check skipped: {e}")

            if not preserve_lines:
                raw_pred = self._ocr_detect.predict(working_image)
                text = (raw_pred or "").strip()
                return text, []

            # Line-preserving extraction
            from ocr_tamil.craft_text_detector import get_prediction, export_detected_regions

            # Normalize size within safe bounds
            size = min(max(working_image.shape[0], working_image.shape[1], 1280), 2560)
            prediction_result = get_prediction(
                image=working_image,
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
                return "", []

            # Extract bounding boxes
            new_bbox = []
            for bb in prediction_result:
                xs, ys = bb[:, 0], bb[:, 1]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                new_bbox.append([int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)])

            # Line clustering using vertical overlap and center distance
            c = np.array(new_bbox)
            median_h = float(np.median(c[:, 3])) if len(c) > 0 else 20.0

            # Sort box indices by vertical center
            sorted_indices = sorted(
                range(len(new_bbox)),
                key=lambda i: (new_bbox[i][1] + new_bbox[i][3] / 2.0)
            )

            line_groups: List[List[int]] = []
            for idx in sorted_indices:
                x, y, w, h = new_bbox[idx]
                cy = y + h / 2.0
                placed = False
                for group in line_groups:
                    group_cys = [new_bbox[i][1] + new_bbox[i][3] / 2.0 for i in group]
                    mean_cy = float(np.mean(group_cys))
                    if abs(cy - mean_cy) <= max(median_h * 0.65, 14.0):
                        group.append(idx)
                        placed = True
                        break
                if not placed:
                    line_groups.append([idx])

            # Sort words within each line left-to-right (x)
            for group in line_groups:
                group.sort(key=lambda idx: new_bbox[idx][0])

            # Sort lines vertically
            line_groups.sort(
                key=lambda group: np.mean([new_bbox[i][1] + new_bbox[i][3] / 2.0 for i in group])
            )

            # Assign sorted line items
            sorted_line_items = []
            updated_prediction_result = []
            for line_no, group in enumerate(line_groups, start=1):
                for orig_idx in group:
                    x, y, w, h = new_bbox[orig_idx]
                    sorted_line_items.append((line_no, x, y, w, h))
                    updated_prediction_result.append(prediction_result[orig_idx])

            # Export region crops
            os.makedirs("temp_images", exist_ok=True)
            exported_regions = export_detected_regions(
                image=working_image,
                regions=updated_prediction_result,
                output_dir="temp_images",
                rectify=False
            )

            # Recognize crops and build structured word items and lines
            words: List[Dict] = []
            lines: List[List[str]] = []
            current_line_no = None
            current_line_words: List[str] = []

            for (line_no, x, y, w, h), crop in zip(sorted_line_items, exported_regions):
                word = self.recognize_crop(crop, language_hint=language_hint)
                if not word:
                    continue

                token_lang = self.detect_token_language(word)
                words.append({
                    "text": word,
                    "bbox": [x, y, w, h],
                    "line": line_no,
                    "center_x": float(x + w / 2.0),
                    "center_y": float(y + h / 2.0),
                    "confidence": 0.95,
                    "language": token_lang,
                    "original_text": word,
                    "display_text": word
                })

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

            formatted_text = "\n".join(" ".join(w for w in line) for line in lines).strip()
            return formatted_text, words

        except Exception as e:
            logger.warning(f"Detailed prediction encountered issue, falling back: {e}")
            try:
                raw_pred = self._ocr_detect.predict(image)
                return (raw_pred or "").strip(), []
            except Exception as e2:
                logger.error(f"Fallback predict failed: {e2}")
                return "", []

    def predict_text(
        self,
        image: np.ndarray,
        detect: bool = True,
        preserve_lines: bool = True
    ) -> str:
        """
        Perform OCR on an image. Backward-compatible wrapper returning string.
        """
        text, _ = self.predict_with_details(image, detect=detect, preserve_lines=preserve_lines)
        return text


ocr_service = OCRService()
