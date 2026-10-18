"""
Format Builder Utility for Tamil OCR API.
Transforms extracted OCR text and tables into three standardized formats:
1. JSON: Rich structured data (lines, word counts, table grids, key-values).
2. HTML: Clean, semantic HTML document fragment (safe against XSS).
3. TEXT: Plain text with reading order and preserved line breaks.
"""

import html
from typing import Any, Dict, List, Optional, Tuple


def _serialize_table(t: Any) -> Dict[str, Any]:
    """Helper to safely convert a TableResult model or dict into a JSON-serializable dict."""
    if hasattr(t, "model_dump"):
        d = t.model_dump()
    elif hasattr(t, "dict"):
        d = t.dict()
    elif isinstance(t, dict):
        d = t
    else:
        d = {
            "headers": getattr(t, "headers", []),
            "rows": getattr(t, "rows", []),
            "data": getattr(t, "data", []),
            "html": getattr(t, "html", ""),
            "confidence": getattr(t, "confidence", None),
        }
    return {
        "table_index": d.get("table_index", 0),
        "headers": d.get("headers", []),
        "rows": d.get("rows", []),
        "data": d.get("data", []),
        "row_count": d.get("row_count", len(d.get("rows", []))),
        "column_count": d.get("column_count", len(d.get("headers", []))),
        "confidence": d.get("confidence"),
        "detection_strategy": d.get("detection_strategy"),
        "detected_languages": d.get("detected_languages", ["ta"]),
    }


def build_output_formats(
    extracted_text: str,
    tables: Optional[List[Any]] = None,
    filename: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    """
    Builds the standardized (text, html, json, formats_bundle) representations.

    Returns:
        (text_str, html_str, json_dict, formats_bundle)
    """
    tables = tables or []
    clean_text = (extracted_text or "").strip()

    # -------------------------------------------------------------------------
    # 1. TEXT FORMAT
    # -------------------------------------------------------------------------
    text_parts: List[str] = []
    if clean_text:
        text_parts.append(clean_text)

    # Append tabular text if tables exist and are not already in clean_text
    if tables and not clean_text:
        for idx, t in enumerate(tables):
            serialized = _serialize_table(t)
            headers = serialized.get("headers", [])
            rows = serialized.get("rows", [])
            if headers:
                text_parts.append(f"[Table {idx + 1}]")
                text_parts.append("\t".join(str(h) for h in headers))
            for row in rows:
                text_parts.append("\t".join(str(c) for c in row))

    text_format = "\n".join(text_parts).strip()

    # -------------------------------------------------------------------------
    # 2. HTML FORMAT
    # -------------------------------------------------------------------------
    html_lines = ['<div class="ocr-document">']

    # Text section
    if clean_text:
        html_lines.append('  <div class="ocr-text-section">')
        for raw_line in clean_text.splitlines():
            stripped = raw_line.strip()
            if stripped:
                escaped = html.escape(stripped)
                html_lines.append(f"    <p>{escaped}</p>")
        html_lines.append("  </div>")

    # Tables section
    if tables:
        html_lines.append('  <div class="ocr-tables-section">')
        for idx, t in enumerate(tables):
            table_html = getattr(t, "html", None)
            if isinstance(t, dict):
                table_html = t.get("html")

            if table_html and table_html.strip():
                html_lines.append(f"    {table_html.strip()}")
            else:
                # Fallback generate table HTML if not pre-rendered
                serialized = _serialize_table(t)
                headers = serialized.get("headers", [])
                rows = serialized.get("rows", [])
                html_lines.append('    <div class="ocr-table-container">')
                html_lines.append(f'      <table class="ocr-rendered-table" data-table-index="{idx}">')
                if headers:
                    html_lines.append("        <thead><tr>")
                    for h in headers:
                        html_lines.append(f"          <th>{html.escape(str(h))}</th>")
                    html_lines.append("        </tr></thead>")
                if rows:
                    html_lines.append("        <tbody>")
                    for r in rows:
                        html_lines.append("          <tr>")
                        for cell in r:
                            html_lines.append(f"            <td>{html.escape(str(cell))}</td>")
                        html_lines.append("          </tr>")
                    html_lines.append("        </tbody>")
                html_lines.append("      </table>")
                html_lines.append("    </div>")
        html_lines.append("  </div>")

    html_lines.append("</div>")
    html_format = "\n".join(html_lines)

    # -------------------------------------------------------------------------
    # 3. JSON FORMAT
    # -------------------------------------------------------------------------
    non_empty_lines = [l.strip() for l in clean_text.splitlines() if l.strip()]
    serialized_tables = [_serialize_table(t) for t in tables]

    json_format: Dict[str, Any] = {
        "filename": filename,
        "text": text_format,
        "lines": non_empty_lines,
        "line_count": len(non_empty_lines),
        "word_count": len(text_format.split()) if text_format else 0,
        "character_count": len(text_format),
        "has_tables": len(serialized_tables) > 0,
        "table_count": len(serialized_tables),
        "tables": serialized_tables,
    }
    if metadata:
        json_format["metadata"] = metadata

    # -------------------------------------------------------------------------
    # 4. FORMATS BUNDLE
    # -------------------------------------------------------------------------
    formats_bundle: Dict[str, Any] = {
        "text": text_format,
        "html": html_format,
        "json": json_format,
    }

    return text_format, html_format, json_format, formats_bundle
