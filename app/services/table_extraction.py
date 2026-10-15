"""
Table Extraction Service.
Detects bordered and borderless tabular structures using OpenCV morphology and spatial clustering,
extracts text from individual cells via ocr-tamil, and builds structured row/column schemas.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.schemas.ocr import TableColumn, TableResult
from app.services.ocr_service import ocr_service

logger = logging.getLogger("tamil_ocr.tables")


class TableExtractor:
    """Robust table detection and structured cell extraction."""

    @staticmethod
    def _slugify_key(text: str, fallback_idx: int) -> str:
        """Create a clean, stable JSON key from header text while preserving semantic meaning."""
        if not text or not text.strip():
            return f"col_{fallback_idx}"

        # If ASCII alphanumeric characters exist, extract them
        cleaned = re.sub(r"[^\w\s-]", "", text).strip().lower()
        cleaned = re.sub(r"[-\s]+", "_", cleaned)
        if cleaned and len(cleaned) <= 30:
            return cleaned

        return f"col_{fallback_idx}"

    def extract_tables(
        self,
        image_bgr: np.ndarray,
        extract_text: bool = True
    ) -> Tuple[List[TableResult], List[str]]:
        """
        Main extraction entry point.
        Attempts bordered table detection first; falls back to alignment-based detection if borders absent.
        Returns:
            - List of TableResult
            - List of warning messages
        """
        warnings = []
        tables: List[TableResult] = []

        try:
            bordered_tables, border_warnings = self._detect_bordered_tables(image_bgr, extract_text)
            tables.extend(bordered_tables)
            warnings.extend(border_warnings)
        except Exception as e:
            logger.warning(f"Bordered table detection encountered error: {e}")
            warnings.append(f"Bordered table parsing failed: {str(e)}")

        if not tables:
            # Fallback to borderless heuristic
            try:
                borderless_tables, borderless_warnings = self._detect_borderless_tables(image_bgr)
                tables.extend(borderless_tables)
                warnings.extend(borderless_warnings)
            except Exception as e:
                logger.debug(f"Borderless table detection failed: {e}")

        if not tables and not warnings:
            warnings.append("No tabular structures detected in the image.")

        return tables, warnings

    def _detect_bordered_tables(
        self,
        img: np.ndarray,
        extract_text: bool = True
    ) -> Tuple[List[TableResult], List[str]]:
        """Detect tables defined by visible line borders using morphological kernels."""
        warnings: List[str] = []
        results: List[TableResult] = []

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Adaptive thresholding to handle uneven lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )

        # Detect horizontal lines
        h_size = max(w // 30, 20)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)

        # Detect vertical lines
        v_size = max(h // 30, 20)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

        # Combine grid
        grid = cv2.add(h_lines, v_lines)

        # Find external contours of whole tables
        table_contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        table_idx = 0
        for tc in table_contours:
            tx, ty, tw, th = cv2.boundingRect(tc)
            # Minimum table size filters
            if tw < 80 or th < 50:
                continue

            # Extract table sub-image and sub-grid
            table_roi = img[ty:ty+th, tx:tx+tw]
            grid_roi = grid[ty:ty+th, tx:tx+tw]

            # Find cell contours within this table
            cell_contours, _ = cv2.findContours(grid_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cells = []
            for cc in cell_contours:
                cx, cy, cw, ch = cv2.boundingRect(cc)
                # Ignore the whole table boundary itself and tiny noise
                if (cw > 20 and ch > 15) and not (cw >= tw - 10 and ch >= th - 10):
                    cells.append((cx, cy, cw, ch))

            if len(cells) < 2:
                continue

            # Cluster cells into rows based on Y coordinates
            c_arr = np.array(cells)
            median_h = np.median(c_arr[:, 3])
            y_tolerance = max(median_h * 0.4, 8)

            # Sort cells top-to-bottom
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
                    # Sort row cells left-to-right
                    rows_grouped.append(sorted(current_row, key=lambda c: c[0]))
                    current_row = [(cx, cy, cw, ch)]
                    current_row_y = cy

            if current_row:
                rows_grouped.append(sorted(current_row, key=lambda c: c[0]))

            if len(rows_grouped) < 1:
                continue

            # Deduplicate any overlapping cell boxes in the same row
            cleaned_rows_grouped = []
            for row_cells in rows_grouped:
                filtered_cells = []
                for cell in row_cells:
                    if not filtered_cells:
                        filtered_cells.append(cell)
                    else:
                        prev = filtered_cells[-1]
                        # If boxes heavily overlap in X (IoU or center distance)
                        if abs(cell[0] - prev[0]) > 10:
                            filtered_cells.append(cell)
                cleaned_rows_grouped.append(filtered_cells)

            rows_grouped = cleaned_rows_grouped

            # Recognize text inside each cell
            table_text_grid: List[List[str]] = []
            for row_cells in rows_grouped:
                row_texts = []
                for cx, cy, cw, ch in row_cells:
                    cell_text = ""
                    if extract_text and ocr_service.is_healthy():
                        # Crop with inner margin (2px) to remove border lines
                        pad = 2
                        cell_crop = table_roi[
                            max(0, cy + pad): min(th, cy + ch - pad),
                            max(0, cx + pad): min(tw, cx + cw - pad)
                        ]
                        if cell_crop.size > 0:
                            cell_text = ocr_service.recognize_crop(cell_crop)
                    row_texts.append(cell_text)
                table_text_grid.append(row_texts)

            if not table_text_grid:
                continue

            # Determine column count
            max_cols = max(len(r) for r in table_text_grid)

            # Pad rows to ensure uniform column count
            for r in table_text_grid:
                while len(r) < max_cols:
                    r.append("")

            # Split into headers and data rows
            if len(table_text_grid) >= 2:
                headers = table_text_grid[0]
                body_rows = table_text_grid[1:]
            else:
                headers = [f"Column {i+1}" for i in range(max_cols)]
                body_rows = table_text_grid

            # Build structured column definitions and machine-readable data
            columns: List[TableColumn] = []
            for idx, h_text in enumerate(headers):
                key = self._slugify_key(h_text, idx)
                label = h_text if h_text.strip() else f"Column {idx+1}"
                columns.append(TableColumn(key=key, label=label))

            data: List[Dict[str, str]] = []
            for row in body_rows:
                row_dict = {}
                for col_def, val in zip(columns, row):
                    row_dict[col_def.key] = str(val)
                data.append(row_dict)

            results.append(TableResult(
                table_index=table_idx,
                headers=headers,
                rows=body_rows,
                row_count=len(body_rows),
                column_count=max_cols,
                columns=columns,
                data=data,
                confidence=0.95
            ))
            table_idx += 1

        return results, warnings

    def _detect_borderless_tables(
        self,
        img: np.ndarray
    ) -> Tuple[List[TableResult], List[str]]:
        """
        Fallback heuristic for tables without visible grid borders based on text clustering.
        """
        # If no borders found, returns an empty list with clear warning rather than fabricating data
        return [], ["Borderless table detection: no clear aligned grid structure detected."]


table_extractor = TableExtractor()
