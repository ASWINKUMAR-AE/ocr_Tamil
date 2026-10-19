"""
Image Preprocessing Service for Tamil OCR API.
Provides orientation correction, non-destructive contrast enhancement,
deskewing, noise reduction, and dimension normalization tailored for Tamil script.
"""

import io
import logging
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageOps

from app.config import settings

logger = logging.getLogger("tamil_ocr.preprocessing")


class ImagePreprocessor:
    """Preprocesses images for optimal Tamil text detection and recognition."""

    @staticmethod
    def decode_and_orient(image_bytes: bytes) -> np.ndarray:
        """
        Decode raw image bytes using Pillow to honor EXIF orientation tags,
        then convert into BGR NumPy array for OpenCV and CRAFT processing.
        """
        pil_img = Image.open(io.BytesIO(image_bytes))

        # Automatically transpose based on EXIF orientation if present
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception as e:
            logger.debug(f"EXIF transpose failed: {e}")

        # Convert RGBA/Palette/Grayscale to RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        rgb_arr = np.array(pil_img)
        # Convert RGB to BGR for OpenCV
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        return bgr_arr

    @staticmethod
    def safe_resize(img: np.ndarray, max_dim: int = 2560) -> Tuple[np.ndarray, bool]:
        """
        Resize image if dimensions exceed max_dim or are too small for recognition.
        Preserves aspect ratio strictly.
        """
        h, w = img.shape[:2]
        resized = False

        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resized = True
        elif min(h, w) < 32:
            scale = 32.0 / float(min(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            resized = True

        return img, resized

    @staticmethod
    def deskew(img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect text skew angle and rotate to straighten text lines.
        Uses Hough transform on edge/ruled lines with contour fallback.
        Only corrects slight skew (-25 to +25 degrees) to avoid false 90-degree flips.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        (h, w) = img.shape[:2]

        angle = 0.0

        # Method 1: Probabilistic Hough Lines on Canny edges (reliable for notebooks, forms, documents)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        min_line_len = max(50, int(w * 0.05))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=min_line_len, maxLineGap=12)
        if lines is not None and len(lines) >= 3:
            h_angles = []
            for l in lines:
                x1, y1, x2, y2 = l[0]
                deg = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if abs(deg) <= 25.0:
                    h_angles.append(deg)
            if len(h_angles) >= 3:
                angle = float(np.median(h_angles))

        # Method 2: Fallback to text contour minAreaRect if Hough lines didn't find clear angles
        if abs(angle) < 0.4:
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) >= 100:
                rect_angle = cv2.minAreaRect(coords)[-1]
                if rect_angle < -45:
                    rect_angle = -(90 + rect_angle)
                else:
                    rect_angle = -rect_angle
                if 0.5 <= abs(rect_angle) <= 25.0:
                    angle = float(rect_angle)

        # Only deskew if significant skew detected, within safe boundaries
        if 0.4 <= abs(angle) <= 25.0:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            return rotated, float(round(angle, 2))

        return img, 0.0

    @staticmethod
    def enhance_contrast(img: np.ndarray) -> np.ndarray:
        """
        Apply gentle, dimension-adaptive CLAHE to improve text contrast
        without over-amplifying noise or degrading fine Tamil pulli (dots) and curves.
        """
        h, w = img.shape[:2]
        grid_x = max(2, min(8, w // 100))
        grid_y = max(2, min(8, h // 100))
        clip_limit = 1.2 if min(h, w) < 150 else 1.8

        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_x, grid_y))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_x, grid_y))
            return clahe.apply(img)

    @staticmethod
    def reduce_noise_bilateral(img: np.ndarray) -> np.ndarray:
        """
        Bilateral filtering smooths noise while preserving crisp character edges,
        avoiding the blurring of small Tamil vowel symbols.
        """
        return cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)

    def process(
        self,
        image_bytes: bytes,
        enable_preprocess: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Main preprocessing pipeline.
        Returns:
            - processed_bgr: Preprocessed NumPy BGR image
            - original_bgr: Decoded and orientation-corrected NumPy BGR image
            - metadata: Dictionary detailing applied steps
        """
        # Step 1: Decode and fix orientation
        original_bgr = self.decode_and_orient(image_bytes)

        # Step 2: Safe resize to prevent out-of-memory or excessive sizes
        original_bgr, was_resized = self.safe_resize(
            original_bgr, max_dim=settings.MAX_IMAGE_DIMENSION
        )

        h, w = original_bgr.shape[:2]
        meta = {
            "preprocessing_applied": enable_preprocess,
            "original_width": w,
            "original_height": h,
            "resized": was_resized,
            "skew_angle": 0.0,
            "steps": ["orientation_correction"],
        }

        if was_resized:
            meta["steps"].append("safe_resize")

        if not enable_preprocess:
            return original_bgr, original_bgr.copy(), meta

        # Preprocessing sequence
        working_img = original_bgr.copy()

        # Step 3: Deskew
        working_img, angle = self.deskew(working_img)
        if abs(angle) >= 0.5:
            meta["skew_angle"] = round(angle, 2)
            meta["steps"].append("deskew")

        # Step 4: Bilateral noise reduction
        working_img = self.reduce_noise_bilateral(working_img)
        meta["steps"].append("noise_reduction")

        # Step 5: Contrast enhancement
        working_img = self.enhance_contrast(working_img)
        meta["steps"].append("contrast_enhancement")

        return working_img, original_bgr, meta


image_preprocessor = ImagePreprocessor()
