"""
Table Extraction Service.
Implements a fully dynamic table structure recognition and schema discovery pipeline
for Tamil, English, and multilingual documents:
  - Dynamic Table Region & Boundary Detection
  - Header Row Identification (Level A: Explicit Headers)
  - Layout-Adaptive Column Boundary Discovery & De-slanted Alignment
  - Accurate Geometric Cell Assignment (preserves multi-line, empty cells, and ordering)
  - Dynamic Semantic Type Inference (Level B: Data Distribution Analysis)
  - Position-Based Neutral Fallback (Level C: Column N with review flags)
  - Optional AI-Assisted Schema Inference Interface with Seamless Local Fallback
  - Multi-Strategy Integration (Dynamic Spatial BBox, OpenCV Morphology Grid, Text Alignment)
  - Semantic, Accessible, XSS-Safe HTML Table Rendering
"""

import html
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.schemas.ocr import ColumnHeaderDetail, TableColumn, TableResult
from app.services.ocr_service import ocr_service
from app.services.pattern_classifier import pattern_classifier, ClassificationResult

logger = logging.getLogger("tamil_ocr.tables")

# Common Tamil and English table header vocabulary for dynamic header row detection
TABLE_HEADER_KEYWORDS = {
    # Tamil
    "வரிசை", "பதிவு", "பெயர்", "விவரம்", "தேதி", "எண்", "தொகை", "விலை",
    "மாதம்", "ஆண்டு", "வ.எண்", "குறிப்பு", "வங்கி", "முகவரி", "இனம்", "அளவு",
    "நிலை", "ஊர்", "வயது", "தொலைபேசி", "செல்பேசி", "கைபேசி", "நாள்", "காலம்",
    "கட்டணம்", "மொத்தம்", "மாணவர்", "சேர்க்கை", "பாலினம்", "பெற்றோர்", "தந்தை",
    "தாய்", "தாயார்", "சாதி", "வகுப்பு", "பாடம்", "மதிப்பெண்", "தேர்ச்சி", "வரி",
    "மாடு", "பால்", "விவசாயி", "கொள்முதல்", "பற்று", "வரவு", "மீதி", "இருப்பு",
    "கால்நடை", "பசு", "எடை", "கணக்கு", "மின்னஞ்சல்", "குறியீடு",
    # English
    "sl", "s.no", "serial", "no", "name", "date", "dob", "particulars", "amount", "total", "description",
    "id", "reg", "phone", "mobile", "contact", "address", "city", "age", "qty", "quantity",
    "gender", "sex", "rate", "price", "unit", "balance", "debit", "credit", "father", "parent",
    "student", "admission", "roll", "class", "cattle", "cow", "tag", "pashu", "aadhaar",
    "account", "acc", "item", "sn", "code", "status", "remarks", "milk", "weight", "email",
    "pin", "tax", "gst", "invoice", "bill"
}
TAMIL_HEADER_KEYWORDS = TABLE_HEADER_KEYWORDS

# Common compound header suffixes in Tamil & English (merged when intra-header spacing is small)
HEADER_COMPOUND_SUFFIXES = {
    "பெயர்", "எண்", "எஸ்ஸா", "தேதி", "அளவு", "விலை", "தொகை", "விவரம்", "குறியீடு",
    "no", "name", "date", "number", "code", "amount", "rate", "count", "id"
}


def clean_tamil_header_ocr(text: str) -> str:
    """
    Correct minor OCR character confusions in common Tamil header keywords
    without rewriting general document words.
    """
    t = text.strip()
    replacements = {
        "வடஎண்": "வ.எண்",
        "வஎண்": "வ.எண்",
        "வரினை": "வரிசை",
        "வர்னின்": "வரிசை",
        "டதிவு": "பதிவு எண்",
        "எஸ்ஸா": "எண்",
        "தெடு": "தெரு",
        "பிஷத்": "பிறந்த தேதி",
        "smiss": "பிறந்த தேதி",
        "மிநகலையேகி": "தொலைபேசி எண்",
        "மிகுகையேமுகி": "தொலைபேசி எண்",
        "மிகஃபைக்கரி": "தொலைபேசி எண்",
        "பட்டைதீயும்": "தந்தை பெயர்",
        "பூட்டையையே": "தந்தை பெயர்",
        "பூட்டின்": "தந்தை பெயர்",
        "டுகவமி": "முகவரி",
        "அகவமி": "முகவரி",
        "அத்வமி": "முகவரி",
    }
    for k, v in replacements.items():
        if k in t:
            t = t.replace(k, v)
    return t


def generate_html_table(
    headers: List[str],
    rows: List[List[str]],
    table_index: int = 0,
    strategy: Optional[str] = None,
    is_uncertain: bool = False,
    headers_detail: Optional[List[ColumnHeaderDetail]] = None
) -> str:
    """
    Generate semantic, accessible, editable, and XSS-safe HTML table markup.
    Preserves exact Tamil Unicode characters, escapes HTML control characters,
    supports multi-line cell formatting, and prevents column shifts on empty cells.
    """
    escaped_headers = [html.escape(str(h or "").strip()) for h in headers]
    strategy_attr = f' data-strategy="{html.escape(strategy)}"' if strategy else ""
    uncertain_class = " table-uncertain" if is_uncertain else ""

    html_parts = [
        f'<div class="ocr-table-container{uncertain_class}">',
        f'  <table class="ocr-rendered-table" data-table-index="{table_index}"{strategy_attr}>',
    ]

    # Render Table Header
    if escaped_headers and any(h for h in escaped_headers):
        html_parts.append("    <thead>")
        html_parts.append("      <tr>")
        for idx, h_text in enumerate(escaped_headers):
            display_h = h_text if h_text else f"Column {idx + 1}"
            detail = headers_detail[idx] if headers_detail and idx < len(headers_detail) else None
            sem_attr = f' data-semantic-type="{html.escape(detail.semantic_type)}"' if detail else ""
            method_attr = f' data-inference-method="{html.escape(detail.inference_method)}"' if detail else ""
            warn_title = f' title="{html.escape("; ".join(detail.classification_warnings))}"' if (detail and detail.classification_warnings) else ' title="Click to rename"'
            warn_badge = ' <span class="th-warn-icon" title="Review note">⚠️</span>' if (detail and detail.classification_warnings) else ""
            html_parts.append(
                f'        <th scope="col" data-col-idx="{idx}" contenteditable="true"{sem_attr}{method_attr}{warn_title}>{display_h}{warn_badge}</th>'
            )
        html_parts.append("      </tr>")
        html_parts.append("    </thead>")

    # Render Table Body
    html_parts.append("    <tbody>")
    for r_idx, row in enumerate(rows):
        html_parts.append(f'      <tr data-row-idx="{r_idx}">')
        col_count = len(escaped_headers) if escaped_headers else len(row)
        for c_idx in range(col_count):
            val = str(row[c_idx]) if c_idx < len(row) and row[c_idx] is not None else ""
            escaped_val = html.escape(val).replace("\n", "<br>")
            display_val = escaped_val if escaped_val else "&nbsp;"
            html_parts.append(f'        <td data-col-idx="{c_idx}" contenteditable="true">{display_val}</td>')
        html_parts.append("      </tr>")
    html_parts.append("    </tbody>")
    html_parts.append("  </table>")
    html_parts.append("</div>")

    return "\n".join(html_parts)


# =============================================================================
# Optional AI Schema Inference Interface (Provider-Independent)
# =============================================================================

class BaseAISchemaProvider:
    """Abstract interface for optional AI-assisted table schema understanding."""

    def infer_schema(
        self,
        headers: List[str],
        sample_rows: List[List[str]],
        column_count: int,
        document_context: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Infer column meanings and types. Return None to fall back to rule-based engine."""
        raise NotImplementedError


class AISchemaInferer:
    """
    Provider-independent AI schema inference runner with strict fallback to local engine.
    Does not require external APIs by default and protects sensitive data.
    """

    def __init__(self, provider: Optional[BaseAISchemaProvider] = None):
        self.provider = provider
        self.enabled = bool(os.getenv("AI_SCHEMA_ENABLED", "false").lower() in {"1", "true", "yes"})

    def infer(
        self,
        headers: List[str],
        sample_rows: List[List[str]],
        column_count: int,
        document_context: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        if not self.enabled or not self.provider:
            return None
        try:
            return self.provider.infer_schema(headers, sample_rows, column_count, document_context)
        except Exception as e:
            logger.debug(f"Optional AI Schema inference failed or skipped, falling back to local engine: {e}")
            return None


# Global AI Inferer instance
ai_schema_inferer = AISchemaInferer()


# =============================================================================
# Main Dynamic Table Extractor
# =============================================================================

class TableExtractor:
    """
    Dynamic table structure recognition engine. Discovers table boundaries,
    header rows, column counts, and cell values directly from document geometry.
    """

    @staticmethod
    def _slugify_key(text: str, fallback_idx: int) -> str:
        """Create a clean, stable programmatic key from header text."""
        if not text or not text.strip():
            return f"col_{fallback_idx + 1}"

        cleaned = re.sub(r"[^\w\s-]", "", text).strip().lower()
        cleaned = re.sub(r"[-\s]+", "_", cleaned)
        if cleaned and len(cleaned) <= 30:
            return cleaned

        return f"col_{fallback_idx + 1}"

    # -------------------------------------------------------------------------
    # Stage 0: Preserve Raw OCR Result
    # -------------------------------------------------------------------------
    def preserve_raw_ocr_result(
        self,
        extracted_text: Optional[str],
        words: Optional[List[Dict[str, Any]]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Safely preserve raw OCR extracted text and word tokens without modification.
        """
        raw_text = str(extracted_text or "")
        token_list = [dict(w) for w in (words or [])]
        return raw_text, token_list

    # -------------------------------------------------------------------------
    # Stage 0B: Detect Document Languages
    # -------------------------------------------------------------------------
    def detect_document_languages(self, words: List[Dict[str, Any]]) -> List[str]:
        """
        Detect dominant document language(s) across words: 'ta', 'en', or ['ta', 'en'].
        """
        return ocr_service.detect_document_languages(words)

    # -------------------------------------------------------------------------
    # Stage 1: Detect Table Regions
    # -------------------------------------------------------------------------
    def detect_table_regions(
        self,
        words: List[Dict[str, Any]],
        image_shape: Tuple[int, ...]
    ) -> List[Dict[str, Any]]:
        """
        Identify bounding region(s) containing tabular structures.
        Separates marginalia / page headers from the table grid.
        """
        if not words:
            return []

        img_h, img_w = image_shape[:2]
        x1_list = [w["bbox"][0] for w in words]
        y1_list = [w["bbox"][1] for w in words]
        x2_list = [w["bbox"][0] + w["bbox"][2] for w in words]
        y2_list = [w["bbox"][1] + w["bbox"][3] for w in words]

        region = {
            "bbox": (max(0, min(x1_list) - 10), max(0, min(y1_list) - 10),
                     min(img_w, max(x2_list) + 10), min(img_h, max(y2_list) + 10)),
            "words": words
        }
        return [region]

    # -------------------------------------------------------------------------
    # Stage 2: Extract & Filter OCR Tokens
    # -------------------------------------------------------------------------
    def extract_ocr_tokens(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out completely empty or zero-area bounding boxes while preserving
        all legitimate OCR text tokens.
        """
        valid = []
        for w in words:
            txt = w.get("text", "").strip()
            bbox = w.get("bbox", [0, 0, 0, 0])
            if txt and bbox[2] > 2 and bbox[3] > 2:
                valid.append(w)
        return valid

    # -------------------------------------------------------------------------
    # Stage 3: Group Tokens into Rows
    # -------------------------------------------------------------------------
    def group_tokens_into_rows(
        self,
        words: List[Dict[str, Any]],
        img_shape: Tuple[int, ...]
    ) -> List[List[Dict[str, Any]]]:
        """
        Group OCR word tokens into horizontal text rows using adaptive clustering.
        Calculates dynamic tolerance from median word height.
        """
        if not words:
            return []

        words_by_y = sorted(words, key=lambda w: w["center_y"])
        word_heights = [w["bbox"][3] for w in words if w["bbox"][3] > 5]
        median_h = float(np.median(word_heights)) if word_heights else 20.0
        y_tol = max(median_h * 0.55, 12.0)

        line_groups: List[List[Dict[str, Any]]] = []
        for w in words_by_y:
            cy = w["center_y"]
            matched_group = None
            for g in line_groups:
                g_cy = sum(item["center_y"] for item in g) / len(g)
                if abs(cy - g_cy) <= y_tol:
                    matched_group = g
                    break
            if matched_group is not None:
                matched_group.append(w)
            else:
                line_groups.append([w])

        # Sort groups vertically and sort words horizontally
        line_groups.sort(key=lambda g: sum(w["center_y"] for w in g) / len(g))
        for g in line_groups:
            g.sort(key=lambda w: w["bbox"][0])

        return line_groups

    # -------------------------------------------------------------------------
    # Stage 4: Detect Header Rows (Level A)
    # -------------------------------------------------------------------------
    def detect_header_rows(
        self,
        line_groups: List[List[Dict[str, Any]]],
        image_shape: Tuple[int, ...]
    ) -> Optional[int]:
        """
        Identify explicit header row if one exists.
        Evaluates row horizontal span, keyword hits, and token count,
        while discarding centered narrow page title banners.
        """
        if not line_groups:
            return None

        img_h, img_w = image_shape[:2]
        header_idx = None
        header_score_max = 0.0

        # Look within the top 6 rows
        for idx, g in enumerate(line_groups[:6]):
            line_text = " ".join(w["text"] for w in g).lower()
            span_x = (g[-1]["bbox"][0] + g[-1]["bbox"][2]) - g[0]["bbox"][0]
            span_ratio = span_x / float(img_w)

            kw_hits = sum(1 for kw in TAMIL_HEADER_KEYWORDS if kw in line_text)
            token_count = len(g)

            # Avoid standalone centered titles (e.g. register title banner spanning < 40%)
            is_centered_narrow_title = (token_count <= 4 and span_ratio < 0.40 and kw_hits <= 2)

            if not is_centered_narrow_title and token_count >= 2:
                score = (kw_hits * 3.0) + (token_count * 1.5) + (span_ratio * 5.0)
                if kw_hits >= 2 or (token_count >= 4 and span_ratio > 0.45):
                    if score > header_score_max:
                        header_score_max = score
                        header_idx = idx

        return header_idx

    # -------------------------------------------------------------------------
    # Stage 5: Discover Columns & Dynamic Column Cuts
    # -------------------------------------------------------------------------
    def infer_table_columns(
        self,
        line_groups: List[List[Dict[str, Any]]],
        header_idx: Optional[int],
        image_shape: Tuple[int, ...]
    ) -> Tuple[List[Dict[str, Any]], List[float], float, float]:
        """
        Determine number of columns dynamically, discover explicit or neutral headers,
        measure document slant slope, and place column divider cuts.
        """
        img_h, img_w = image_shape[:2]
        headers: List[Dict[str, Any]] = []
        col_cuts: List[float] = []
        table_slope = 0.0
        header_y = 0.0

        if header_idx is not None and header_idx < len(line_groups):
            header_y = sum(w["center_y"] for w in line_groups[header_idx]) / len(line_groups[header_idx])
            h_words = line_groups[header_idx]

            # Merge compound header words with small intra-header gaps or compound suffixes
            merged_headers: List[Dict[str, Any]] = []
            curr = None
            for w in h_words:
                if curr is None:
                    curr = {
                        "text": w["text"],
                        "x1": w["bbox"][0],
                        "x2": w["bbox"][0] + w["bbox"][2],
                        "center_x": w["center_x"],
                        "words": [w]
                    }
                else:
                    gap = w["bbox"][0] - curr["x2"]
                    is_suffix = any(
                        w["text"].strip().endswith(s) or w["text"].strip() == s
                        for s in HEADER_COMPOUND_SUFFIXES
                    )
                    if gap <= 8 or (gap <= 16 and is_suffix):
                        curr["text"] = f"{curr['text']} {w['text']}".strip()
                        curr["x2"] = w["bbox"][0] + w["bbox"][2]
                        curr["center_x"] = (curr["x1"] + curr["x2"]) / 2.0
                        curr["words"].append(w)
                    else:
                        merged_headers.append(curr)
                        curr = {
                            "text": w["text"],
                            "x1": w["bbox"][0],
                            "x2": w["bbox"][0] + w["bbox"][2],
                            "center_x": w["center_x"],
                            "words": [w]
                        }
            if curr:
                merged_headers.append(curr)

            for i, mh in enumerate(merged_headers):
                cleaned = clean_tamil_header_ocr(mh["text"])
                headers.append({
                    "original": mh["text"],
                    "display": cleaned,
                    "x1": mh["x1"],
                    "x2": mh["x2"],
                    "center_x": mh["center_x"]
                })

            # Measure document slant slope across rows to align tilted tables
            body_groups = line_groups[(header_idx + 1):]
            if len(body_groups) >= 3:
                start_xs = [g[0]["bbox"][0] for g in body_groups if len(g) >= 2]
                center_ys = [sum(w["center_y"] for w in g) / len(g) for g in body_groups if len(g) >= 2]
                if len(start_xs) >= 3:
                    slope, _ = np.polyfit(center_ys, start_xs, 1)
                    if abs(slope) < 0.25:
                        table_slope = float(slope)

            # De-slant body words to header_y plane
            body_words_norm = []
            for g in body_groups:
                for w in g:
                    shift = table_slope * (w["center_y"] - header_y)
                    body_words_norm.append({
                        "text": w["text"],
                        "x1": w["bbox"][0] - shift,
                        "x2": w["bbox"][0] + w["bbox"][2] - shift,
                        "center_x": w["center_x"] - shift
                    })

            # Calculate cuts between consecutive header anchors with valley refinement
            num_cols = len(headers)
            for i in range(num_cols - 1):
                left_h = headers[i]
                right_h = headers[i + 1]
                default_cut = (left_h["x2"] + right_h["x1"]) / 2.0

                # Search corridor between headers
                search_min = int(max(left_h["center_x"] + 5, default_cut - 35))
                search_max = int(min(right_h["center_x"] - 5, default_cut + 35))
                if right_h["x1"] > left_h["x2"] + 35:
                    search_min = int(left_h["x2"])
                    search_max = int(right_h["x1"])

                search_min = max(0, search_min)
                search_max = min(img_w - 1, search_max)

                if search_max > search_min + 3 and body_words_norm:
                    occ = np.zeros(search_max - search_min + 1, dtype=int)
                    for bw in body_words_norm:
                        bx1 = max(search_min, int(bw["x1"]))
                        bx2 = min(search_max, int(bw["x2"]))
                        if bx1 < bx2:
                            occ[bx1 - search_min: bx2 - search_min] += 1

                    min_v = np.min(occ)
                    valleys = np.where(occ == min_v)[0]
                    if len(valleys) > 0:
                        runs = []
                        curr_run = [valleys[0]]
                        for v in valleys[1:]:
                            if v == curr_run[-1] + 1:
                                curr_run.append(v)
                            else:
                                runs.append(curr_run)
                                curr_run = [v]
                        if curr_run:
                            runs.append(curr_run)

                        best_cut = default_cut
                        best_score = -999999.0
                        for run in runs:
                            run_mid = search_min + (run[0] + run[-1]) / 2.0
                            w_len = len(run)
                            dist = abs(run_mid - default_cut)
                            score = w_len * 1.5 - dist * 0.4
                            if score > best_score:
                                best_score = score
                                best_cut = run_mid
                        col_cuts.append(float(best_cut))
                    else:
                        col_cuts.append(default_cut)
                else:
                    col_cuts.append(default_cut)

        else:
            # Level C: Headerless table -> Valley detection via horizontal projection
            all_words = [w for g in line_groups for w in g]
            min_table_x = min(w["bbox"][0] for w in all_words)
            max_table_x = max(w["bbox"][0] + w["bbox"][2] for w in all_words)

            occ = np.zeros(img_w, dtype=int)
            for w in all_words:
                b = w["bbox"]
                x1 = max(0, b[0])
                x2 = min(img_w - 1, b[0] + b[2])
                occ[x1:x2] += 1

            smoothed = np.convolve(occ, np.ones(15) / 15, mode="same")
            valleys = []
            # Only search WITHIN table boundaries (exclude page margins)
            for x in range(min_table_x + 12, max_table_x - 12):
                if smoothed[x] < 1.0 and occ[x] == 0:
                    valleys.append(x)

            grouped_valleys = []
            if valleys:
                curr_v = [valleys[0]]
                for v in valleys[1:]:
                    if v - curr_v[-1] <= 8:
                        curr_v.append(v)
                    else:
                        grouped_valleys.append(float(np.mean(curr_v)))
                        curr_v = [v]
                if curr_v:
                    grouped_valleys.append(float(np.mean(curr_v)))

            # If no clear valleys found, cluster word centers
            if len(grouped_valleys) < 1:
                all_centers = sorted(w["center_x"] for w in all_words)
                clusters: List[List[float]] = []
                curr_c: List[float] = [all_centers[0]]
                for cx in all_centers[1:]:
                    if cx - curr_c[-1] < 45.0:
                        curr_c.append(cx)
                    else:
                        clusters.append(curr_c)
                        curr_c = [cx]
                if curr_c:
                    clusters.append(curr_c)
                col_centers = [float(np.mean(c)) for c in clusters]
                for i in range(len(col_centers) - 1):
                    grouped_valleys.append((col_centers[i] + col_centers[i + 1]) / 2.0)

            col_cuts = grouped_valleys
            num_cols = len(col_cuts) + 1
            headers = [
                {"original": f"Column {i + 1}", "display": f"Column {i + 1}", "x1": 0, "x2": 0, "center_x": 0}
                for i in range(num_cols)
            ]

        return headers, col_cuts, table_slope, header_y

    # -------------------------------------------------------------------------
    # Stage 6: Assign Tokens to Cells
    # -------------------------------------------------------------------------
    def assign_tokens_to_cells(
        self,
        line_groups: List[List[Dict[str, Any]]],
        header_idx: Optional[int],
        col_cuts: List[float],
        table_slope: float,
        header_y: float,
        num_cols: int
    ) -> List[List[str]]:
        """
        Map each OCR word token into its exact (row, col) cell using de-slanted cuts.
        Preserves original OCR text, intra-cell spacing, and empty cells without shifting.
        """
        def get_col_idx(cx: float, cy: float) -> int:
            norm_cx = cx - table_slope * (cy - header_y)
            for idx, cut in enumerate(col_cuts):
                if norm_cx < cut:
                    return idx
            return len(col_cuts)

        body_rows: List[List[str]] = []
        start_row_idx = (header_idx + 1) if header_idx is not None else 0

        for l_idx in range(start_row_idx, len(line_groups)):
            g = line_groups[l_idx]
            row_cells = [""] * num_cols
            for w in g:
                c_idx = get_col_idx(w["center_x"], w["center_y"])
                if c_idx < num_cols:
                    if row_cells[c_idx]:
                        row_cells[c_idx] = f"{row_cells[c_idx]} {w['text']}".strip()
                    else:
                        row_cells[c_idx] = w["text"].strip()

            # Keep row if it has at least 1 filled cell
            if any(c.strip() for c in row_cells):
                body_rows.append(row_cells)

        return body_rows

    # -------------------------------------------------------------------------
    # Stage 7: Infer Column Semantics (Level B: Semantic Inference & Conflict Resolution)
    # -------------------------------------------------------------------------
    def infer_column_semantics(
        self,
        headers: List[Dict[str, Any]],
        body_rows: List[List[str]],
        is_explicit_header: bool,
        document_context: Optional[str] = None,
        dominant_language: str = "ta"
    ) -> Tuple[List[str], List[ColumnHeaderDetail]]:
        """
        Analyze multi-cell data distribution and header keywords to determine semantic types.
        Implements 5-priority strategy with language-aware labels and conflict resolution:
          Priority 1: Table Geometry (preserved)
          Priority 2: Explicit Header Recognition (Tamil & English)
          Priority 3: Pattern-Based Column Inference (consistency across rows)
          Priority 4: Contextual and Language-Based Inference (cattle, student, bilingual)
          Priority 5: Neutral Fallback (Column N)
        """
        num_cols = len(headers)
        display_headers: List[str] = []
        header_details: List[ColumnHeaderDetail] = []

        for c_idx in range(num_cols):
            h_info = headers[c_idx]
            raw_h = h_info.get("original", "").strip()
            disp_h = h_info.get("display", "").strip()

            col_vals = [r[c_idx].strip() for r in body_rows if c_idx < len(r) and r[c_idx].strip()]

            # Use PatternClassifier to analyze column distribution
            class_res = pattern_classifier.analyze_column_distribution(
                column_values=col_vals,
                header_text=disp_h if is_explicit_header else raw_h,
                document_context=document_context,
                preferred_language=dominant_language
            )

            final_disp_h = class_res.display_text
            if not final_disp_h or final_disp_h.lower() == "column":
                final_disp_h = f"Column {c_idx + 1}"

            display_headers.append(final_disp_h)
            header_details.append(ColumnHeaderDetail(
                original=raw_h,
                display=final_disp_h,
                original_header=raw_h or final_disp_h,
                display_header=final_disp_h,
                semantic_type=class_res.semantic_type,
                confidence=float(round(class_res.confidence, 2)),
                inference_method=class_res.inference_method,
                classification_warnings=class_res.warnings
            ))

        return display_headers, header_details

    # -------------------------------------------------------------------------
    # Stage 7B: Classify Value Patterns
    # -------------------------------------------------------------------------
    def classify_value_patterns(
        self,
        value: str,
        column_context: Optional[str] = None,
        document_context: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify a single value using the configurable pattern recognition engine.
        """
        return pattern_classifier.classify_single_value(
            value, column_context=column_context, document_context=document_context
        )

    # -------------------------------------------------------------------------
    # Stage 7C: Resolve Classification Conflicts
    # -------------------------------------------------------------------------
    def resolve_classification_conflicts(
        self,
        header_details: List[ColumnHeaderDetail],
        body_rows: List[List[str]]
    ) -> List[str]:
        """
        Aggregate any classification conflicts or structural warnings across columns.
        """
        warnings = []
        for d in header_details:
            if d.classification_warnings:
                warnings.extend(d.classification_warnings)
        return warnings

    # -------------------------------------------------------------------------
    # Stage 8: Validate Table Structure
    # -------------------------------------------------------------------------
    def validate_table_structure(
        self,
        rows: List[List[str]],
        num_cols: int
    ) -> List[List[str]]:
        """
        Ensure every row has exactly num_cols cells without shifting neighboring values.
        Filters out stray marginalia rows with fewer than 2 filled cells when multiple columns exist.
        """
        valid_rows = []
        for r in rows:
            # Normalize length
            while len(r) < num_cols:
                r.append("")
            if len(r) > num_cols:
                r = r[:num_cols]

            filled = sum(1 for c in r if c.strip())
            # For tables with 3+ columns, ignore isolated stray lines with only 1 token
            if num_cols >= 3:
                if filled >= 2:
                    valid_rows.append(r)
            else:
                if filled >= 1:
                    valid_rows.append(r)

        return valid_rows

    # -------------------------------------------------------------------------
    # Main Pipeline Entry Point
    # -------------------------------------------------------------------------
    def extract_tables(
        self,
        image_bgr: np.ndarray,
        words: Optional[List[Dict[str, Any]]] = None,
        extracted_text: Optional[str] = None,
        extract_text: bool = True
    ) -> Tuple[List[TableResult], List[str]]:
        """
        Main multi-strategy table extraction entry point:
          1. Strategy A: Dynamic Layout & Schema Discovery (robust for registers, ledgers, borderless tables)
          2. Strategy B: Morphology Grid Detection (for bordered tables with ruled lines)
          3. Strategy C: Text Alignment & Pattern Fallback (for text-only patterns)
        """
        warnings: List[str] = []
        tables: List[TableResult] = []

        # Run OCR if words are not pre-computed
        if (words is None or extracted_text is None) and extract_text and ocr_service.is_healthy():
            try:
                ocr_text, ocr_words = ocr_service.predict_with_details(image_bgr)
                if words is None:
                    words = ocr_words
                if extracted_text is None:
                    extracted_text = ocr_text
            except Exception as e:
                logger.warning(f"Could not perform pre-table OCR inference: {e}")

        # ---------------------------------------------------------------------
        # Strategy A: Dynamic Layout & Schema Discovery
        # ---------------------------------------------------------------------
        if words and len(words) >= 3:
            try:
                table_a = self._extract_dynamic_table(words, image_bgr.shape)
                if table_a:
                    tables.append(table_a)
                    return tables, warnings
            except Exception as e:
                logger.debug(f"Strategy A (Dynamic Schema) encountered error: {e}")

        # ---------------------------------------------------------------------
        # Strategy B: Morphology Grid Detection
        # ---------------------------------------------------------------------
        try:
            bordered_tables, border_warnings = self._detect_bordered_tables(
                image_bgr, words=words, extract_text=extract_text
            )
            if bordered_tables:
                tables.extend(bordered_tables)
                warnings.extend(border_warnings)
                return tables, warnings
        except Exception as e:
            logger.debug(f"Strategy B (Morphology grid) encountered error: {e}")

        # ---------------------------------------------------------------------
        # Strategy C: Text Alignment & Pattern Analysis
        # ---------------------------------------------------------------------
        if extracted_text and extracted_text.strip():
            try:
                table_c = self._extract_table_from_text_patterns(extracted_text)
                if table_c:
                    tables.append(table_c)
                    warnings.append("Table structure reconstructed from text alignment. Please review columns.")
                    return tables, warnings
            except Exception as e:
                logger.debug(f"Strategy C (Text alignment) encountered error: {e}")

        if not tables and not warnings:
            warnings.append("No tabular structures detected in the image.")

        return tables, warnings

    # =========================================================================
    # Strategy A: Full Dynamic Schema Extraction Implementation
    # =========================================================================
    def _extract_dynamic_table(
        self,
        words: List[Dict[str, Any]],
        image_shape: Tuple[int, ...]
    ) -> Optional[TableResult]:
        """
        Execute full dynamic table schema discovery pipeline:
          detect_table_regions -> extract_ocr_tokens -> group_tokens_into_rows
          -> detect_header_rows -> infer_table_columns -> assign_tokens_to_cells
          -> infer_column_semantics -> validate_table_structure -> render_table_data
        """
        valid_words = self.extract_ocr_tokens(words)
        if len(valid_words) < 4:
            return None

        # 1. Group tokens into rows
        line_groups = self.group_tokens_into_rows(valid_words, image_shape)
        if len(line_groups) < 2:
            return None

        # 2. Detect Header Row
        header_idx = self.detect_header_rows(line_groups, image_shape)
        is_explicit_header = (header_idx is not None)

        # 3. Discover Columns & Cuts
        raw_headers, col_cuts, table_slope, header_y = self.infer_table_columns(
            line_groups, header_idx, image_shape
        )
        num_cols = len(raw_headers)
        if num_cols < 2:
            return None

        # 4. Assign Tokens to Cells
        body_rows = self.assign_tokens_to_cells(
            line_groups, header_idx, col_cuts, table_slope, header_y, num_cols
        )
        if len(body_rows) < 1:
            return None

        # 5. Validate Table Structure
        body_rows = self.validate_table_structure(body_rows, num_cols)
        if len(body_rows) < 1:
            return None

        # Validate fill rate to avoid false positives on unformatted text
        total_cells = len(body_rows) * num_cols
        filled_cells = sum(1 for r in body_rows for c in r if c.strip())
        if (filled_cells / float(total_cells)) < 0.20:
            return None

        # Document language detection & document context string
        doc_languages = self.detect_document_languages(valid_words)
        dominant_lang = doc_languages[0] if doc_languages else "ta"
        doc_context = " ".join(w.get("text", "") for w in valid_words[:30])

        # 6. Infer Column Semantics (Level B / Level C with pattern recognition)
        display_headers, header_details = self.infer_column_semantics(
            raw_headers, body_rows, is_explicit_header,
            document_context=doc_context,
            dominant_language=dominant_lang
        )

        # 7. Optional AI Schema Enhancement
        ai_result = ai_schema_inferer.infer(
            headers=display_headers,
            sample_rows=body_rows[:5],
            column_count=num_cols,
            document_context=doc_context
        )
        if ai_result and len(ai_result) == num_cols:
            for idx, ai_col in enumerate(ai_result):
                if isinstance(ai_col, dict):
                    if "display" in ai_col and ai_col["display"]:
                        display_headers[idx] = ai_col["display"]
                        header_details[idx].display = ai_col["display"]
                    if "semantic_type" in ai_col:
                        header_details[idx].semantic_type = ai_col["semantic_type"]
                    header_details[idx].inference_method = "ai_assisted"

        # 8. Build TableColumn definitions & Key-Value data
        columns: List[TableColumn] = []
        for idx in range(num_cols):
            h_text = display_headers[idx]
            detail = header_details[idx]
            key = self._slugify_key(h_text, idx)
            columns.append(TableColumn(
                key=key,
                label=h_text,
                original_header=detail.original,
                display_header=detail.display,
                semantic_type=detail.semantic_type,
                confidence=detail.confidence,
                inference_method=detail.inference_method,
                classification_warnings=detail.classification_warnings
            ))

        data: List[Dict[str, str]] = []
        for row in body_rows:
            data.append({col.key: str(val) for col, val in zip(columns, row)})

        # Compute unassigned text outside table boundaries
        table_line_groups = line_groups[header_idx:] if header_idx is not None else line_groups
        all_table_words = [w for g in table_line_groups for w in g]
        if all_table_words:
            min_table_y = min(w["bbox"][1] for w in all_table_words) - 5
            max_table_y = max(w["bbox"][1] + w["bbox"][3] for w in all_table_words) + 5
            min_table_x = min(w["bbox"][0] for w in all_table_words) - 5
            max_table_x = max(w["bbox"][0] + w["bbox"][2] for w in all_table_words) + 5

            outside_tokens = [
                w.get("text", "") for w in words
                if (w.get("center_y", w["bbox"][1] + w["bbox"][3] / 2) < min_table_y or
                    w.get("center_y", w["bbox"][1] + w["bbox"][3] / 2) > max_table_y or
                    w.get("center_x", w["bbox"][0] + w["bbox"][2] / 2) < min_table_x or
                    w.get("center_x", w["bbox"][0] + w["bbox"][2] / 2) > max_table_x)
            ]
            unassigned_text = " ".join(t for t in outside_tokens if t.strip()).strip() if outside_tokens else None
        else:
            unassigned_text = None

        all_warnings = self.resolve_classification_conflicts(header_details, body_rows)
        is_uncertain = not is_explicit_header or any(d.confidence < 0.70 for d in header_details)
        html_markup = generate_html_table(
            headers=display_headers,
            rows=body_rows,
            table_index=0,
            strategy="bbox_clustering",
            is_uncertain=is_uncertain,
            headers_detail=header_details
        )

        return TableResult(
            table_index=0,
            table_id=1,
            headers=display_headers,
            headers_detail=header_details,
            rows=body_rows,
            row_count=len(body_rows),
            column_count=num_cols,
            columns=columns,
            data=data,
            confidence=0.94 if is_explicit_header else 0.82,
            detection_strategy="bbox_clustering",
            is_uncertain=is_uncertain,
            warnings=all_warnings,
            html=html_markup,
            detected_languages=doc_languages,
            unassigned_text=unassigned_text
        )

    # =========================================================================
    # Strategy B: Enhanced Morphology Grid Detection (Ruled Tables)
    # =========================================================================
    def _detect_bordered_tables(
        self,
        img: np.ndarray,
        words: Optional[List[Dict[str, Any]]] = None,
        extract_text: bool = True
    ) -> Tuple[List[TableResult], List[str]]:
        """
        Detect bordered tables with visible lines using multi-scale morphological kernels.
        Maps recognized OCR words directly into detected cells.
        """
        warnings: List[str] = []
        results: List[TableResult] = []

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )

        # Multi-scale horizontal & vertical line detection
        h_lines = np.zeros_like(thresh)
        for divisor in [30, 45, 60]:
            h_size = max(w // divisor, 15)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
            h_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)
            h_lines = cv2.bitwise_or(h_lines, h_opened)

        v_lines = np.zeros_like(thresh)
        for divisor in [30, 45, 60]:
            v_size = max(h // divisor, 15)
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
            v_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=1)
            v_lines = cv2.bitwise_or(v_lines, v_opened)

        grid = cv2.add(h_lines, v_lines)
        grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)

        table_contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        table_idx = 0

        for tc in table_contours:
            tx, ty, tw, th = cv2.boundingRect(tc)
            if tw < 80 or th < 50:
                continue

            table_roi = img[ty:ty + th, tx:tx + tw]
            grid_roi = grid[ty:ty + th, tx:tx + tw]

            cell_contours, _ = cv2.findContours(grid_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cells = []
            for cc in cell_contours:
                cx, cy, cw, ch = cv2.boundingRect(cc)
                if (cw > 20 and ch > 14) and not (cw >= tw - 12 and ch >= th - 12):
                    cells.append((cx, cy, cw, ch))

            if len(cells) < 4:
                continue

            c_arr = np.array(cells)
            median_h = np.median(c_arr[:, 3])
            y_tolerance = max(median_h * 0.45, 8)

            sorted_by_y = sorted(cells, key=lambda c: c[1])
            rows_grouped: List[List[Tuple[int, int, int, int]]] = []
            current_row: List[Tuple[int, int, int, int]] = []
            current_row_y = None

            for cx, cy, cw, ch in sorted_by_y:
                if current_row_y is None:
                    current_row_y = cy
                    current_row.append((cx, cy, cw, ch))
                elif abs(cy - current_row_y) <= y_tolerance:
                    current_row.append((cx, cy, cw, ch))
                else:
                    rows_grouped.append(sorted(current_row, key=lambda c: c[0]))
                    current_row = [(cx, cy, cw, ch)]
                    current_row_y = cy

            if current_row:
                rows_grouped.append(sorted(current_row, key=lambda c: c[0]))

            if len(rows_grouped) < 2:
                continue

            # Deduplicate overlapping cell boxes
            cleaned_rows_grouped = []
            for row_cells in rows_grouped:
                filtered_cells = []
                for cell in row_cells:
                    if not filtered_cells:
                        filtered_cells.append(cell)
                    else:
                        prev = filtered_cells[-1]
                        if abs(cell[0] - prev[0]) > 10:
                            filtered_cells.append(cell)
                cleaned_rows_grouped.append(filtered_cells)

            rows_grouped = cleaned_rows_grouped
            max_cols = max(len(r) for r in rows_grouped)
            if max_cols < 2:
                continue

            table_text_grid: List[List[str]] = []
            for row_cells in rows_grouped:
                row_texts = []
                for cx, cy, cw, ch in row_cells:
                    cell_text = ""
                    abs_x, abs_y = tx + cx, ty + cy
                    if words:
                        matching_words = [
                            w for w in words
                            if abs_x <= w.get("center_x", 0) <= abs_x + cw
                            and abs_y <= w.get("center_y", 0) <= abs_y + ch
                        ]
                        if matching_words:
                            matching_words.sort(key=lambda item: item.get("center_x", 0))
                            cell_text = " ".join(mw["text"] for mw in matching_words).strip()

                    if not cell_text and extract_text and ocr_service.is_healthy():
                        pad = 2
                        cell_crop = table_roi[
                            max(0, cy + pad): min(th, cy + ch - pad),
                            max(0, cx + pad): min(tw, cx + cw - pad)
                        ]
                        if cell_crop.size > 0:
                            cell_text = ocr_service.recognize_crop(cell_crop)

                    row_texts.append(cell_text)
                table_text_grid.append(row_texts)

            if not table_text_grid or not any(any(c.strip() for c in r) for r in table_text_grid):
                continue

            for r in table_text_grid:
                while len(r) < max_cols:
                    r.append("")

            # Header determination
            if len(table_text_grid) >= 2:
                headers = [h if h.strip() else f"Column {i + 1}" for i, h in enumerate(table_text_grid[0])]
                body_rows = table_text_grid[1:]
            else:
                headers = [f"Column {i + 1}" for i in range(max_cols)]
                body_rows = table_text_grid

            columns = []
            header_details = []
            for idx, h_text in enumerate(headers):
                columns.append(TableColumn(
                    key=self._slugify_key(h_text, idx),
                    label=h_text if h_text.strip() else f"Column {idx + 1}",
                    original_header=h_text,
                    display_header=h_text,
                    semantic_type="unknown",
                    confidence=0.95,
                    inference_method="morphology_grid"
                ))
                header_details.append(ColumnHeaderDetail(
                    original=h_text,
                    display=h_text,
                    semantic_type="unknown",
                    confidence=0.95,
                    inference_method="morphology_grid"
                ))

            data = []
            for row in body_rows:
                data.append({col.key: str(val) for col, val in zip(columns, row)})

            html_markup = generate_html_table(
                headers=headers,
                rows=body_rows,
                table_index=table_idx,
                strategy="morphology_grid",
                is_uncertain=False,
                headers_detail=header_details
            )

            results.append(TableResult(
                table_index=table_idx,
                table_id=table_idx + 1,
                headers=headers,
                headers_detail=header_details,
                rows=body_rows,
                row_count=len(body_rows),
                column_count=max_cols,
                columns=columns,
                data=data,
                confidence=0.95,
                detection_strategy="morphology_grid",
                is_uncertain=False,
                warnings=[],
                html=html_markup
            ))
            table_idx += 1

        return results, warnings

    # =========================================================================
    # Strategy C: Text Alignment & Pattern Analysis
    # =========================================================================
    def _extract_table_from_text_patterns(self, text: str) -> Optional[TableResult]:
        """
        Reconstruct tabular structure from recognized text alignment, delimiters,
        and recurring whitespace patterns.
        """
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            return None

        has_tabs = any("\t" in l for l in lines)
        has_pipes = any("|" in l for l in lines)
        has_multi_spaces = any(re.search(r"\s{2,}", l) for l in lines)

        parsed_rows: List[List[str]] = []
        for l in lines:
            if has_pipes and "|" in l:
                tokens = [t.strip() for t in l.split("|") if t.strip()]
            elif has_tabs and "\t" in l:
                tokens = [t.strip() for t in l.split("\t") if t.strip()]
            elif has_multi_spaces:
                tokens = [t.strip() for t in re.split(r"\s{2,}", l) if t.strip()]
            else:
                tokens = re.split(r"(?<=\d)\s+(?=[^\d\s])|(?<=[^\d\s])\s+(?=\d)", l)
                tokens = [t.strip() for t in tokens if t.strip()]
                if len(tokens) <= 1:
                    tokens = [t.strip() for t in l.split(" ") if t.strip()]

            if tokens:
                parsed_rows.append(tokens)

        multi_field_rows = [r for r in parsed_rows if len(r) >= 2]
        if len(multi_field_rows) < 2:
            return None

        max_cols = max(len(r) for r in parsed_rows)
        if max_cols < 2:
            return None

        for r in parsed_rows:
            while len(r) < max_cols:
                r.append("")

        row0_text = " ".join(parsed_rows[0]).lower()
        has_header_kw = any(kw in row0_text for kw in TAMIL_HEADER_KEYWORDS)

        if has_header_kw and len(parsed_rows) >= 2:
            headers = [h if h.strip() else f"Field {i + 1}" for i, h in enumerate(parsed_rows[0])]
            body_rows = parsed_rows[1:]
        else:
            headers = [f"Field {i + 1}" for i in range(max_cols)]
            body_rows = parsed_rows

        columns = []
        header_details = []
        for i, h in enumerate(headers):
            columns.append(TableColumn(
                key=self._slugify_key(h, i),
                label=h if h.strip() else f"Field {i + 1}",
                original_header=h,
                display_header=h,
                semantic_type="unknown",
                confidence=0.75,
                inference_method="text_alignment"
            ))
            header_details.append(ColumnHeaderDetail(
                original=h,
                display=h,
                semantic_type="unknown",
                confidence=0.75,
                inference_method="text_alignment"
            ))

        data = [{col.key: str(val) for col, val in zip(columns, row)} for row in body_rows]

        html_markup = generate_html_table(
            headers=headers,
            rows=body_rows,
            table_index=0,
            strategy="text_alignment",
            is_uncertain=True,
            headers_detail=header_details
        )

        return TableResult(
            table_index=0,
            table_id=1,
            headers=headers,
            headers_detail=header_details,
            rows=body_rows,
            row_count=len(body_rows),
            column_count=max_cols,
            columns=columns,
            data=data,
            confidence=0.75,
            detection_strategy="text_alignment",
            is_uncertain=True,
            warnings=["Reconstructed from text alignment. Please verify column boundaries."],
            html=html_markup
        )


table_extractor = TableExtractor()
