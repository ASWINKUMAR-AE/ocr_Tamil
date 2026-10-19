"""
Aavin Smart Cattle Passport & Milk Safety Twin Semantic Mapping Profile.
Provides configurable semantic column mapping, data validation, multi-cattle grouping,
and review-first candidate record generation from reconstructed table structures.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.schemas.ocr import (
    CattleCandidateRecord,
    ColumnHeaderDetail,
    FarmerCandidateRecord,
    MilkCandidateRecord,
    TableResult,
)

logger = logging.getLogger("tamil_ocr.aavin_mapping")

# Semantic Column Synonyms (Tamil and English)
HEADER_SYNONYMS: Dict[str, Set[str]] = {
    "farmer_name": {
        # Tamil
        "விவசாயி பெயர்", "விவசாயியின் பெயர்", "பெயர்", "உறுப்பினர் பெயர்",
        "உரிமையாளர் பெயர்", "விவசாயி", "உறுப்பினர்", "பாலாளர் பெயர்",
        # English
        "farmer name", "farmer", "name", "owner name", "member name",
        "farmer_name", "supplier name", "producer name"
    },
    "phone": {
        # Tamil
        "கைபேசி", "தொலைபேசி", "கைபேசி எண்", "தொலைபேசி எண்", "செல் எண்",
        "அலைபேசி", "தொடர்பு எண்", "மொபைல்",
        # English
        "phone", "mobile", "contact", "phone no", "mobile no", "cell",
        "telephone", "contact no", "mobile number"
    },
    "village": {
        # Tamil
        "கிராமம்", "ஊர்", "முகவரி", "இருப்பிடம்", "கிளை", "பகுதி", "வசிப்பிடம்",
        # English
        "village", "address", "place", "town", "locality", "habitation", "city"
    },
    "cattle_id": {
        # Tamil
        "மாடு எண்", "கால்நடை எண்", "மாட்டின் எண்", "காது எண்", "பசு எண்",
        "ஆதார் எண்", "கால்நடை அடையாள அட்டை", "குறியீட்டு எண்", "டேக் எண்",
        "பசு ஆதார்", "பசு ஆதார் எண்", "ஆதார்",
        # English
        "cattle id", "tag id", "tag no", "tag number", "pashu aadhaar",
        "pashu aadhar", "animal id", "ear tag", "cattle tag", "tag",
        "pashu_aadhaar", "inaph id", "12 digit", "12 digit number",
        "12-digit", "aadhar", "aadhaar", "aadhar id", "aadhaar id"
    },
    "cow_count": {
        # Tamil
        "மாடுகள் எண்ணிக்கை", "மாடு எண்ணிக்கை", "மொத்த மாடுகள்", "எண்ணிக்கை",
        "பசுக்கள் எண்ணிக்கை", "கால்நடைகள்",
        # English
        "cow count", "cattle count", "number of cattle", "cows", "total cows",
        "animal count", "no of cows", "qty of cattle"
    },
    "breed": {
        # Tamil
        "இனம்", "மாட்டின் இனம்", "கால்நடை இனம்", "வகையினம்",
        # English
        "breed", "cattle breed", "variety", "species"
    },
    "milk_quantity": {
        # Tamil
        "பால் அளவு", "லிட்டர்", "பால் (லிட்டர்)", "கொள்முதல்", "அளவு",
        "பால் லிட்டர்", "மொத்த பால்",
        # English
        "milk", "litres", "milk quantity", "qty", "quantity", "milk (l)",
        "liters", "volume", "milk supply"
    },
    "date": {
        # Tamil
        "தேதி", "நாள்", "கொள்முதல் நாள்", "பதிவு தேதி",
        # English
        "date", "collection date", "day", "date of collection"
    },
    "session": {
        # Tamil
        "வேளை", "நேரம்", "காலை", "மாலை", "காலை/மாலை",
        # English
        "session", "shift", "morning", "evening", "time"
    },
    "member_id": {
        # Tamil
        "உறுப்பினர் எண்", "விவசாயி எண்", "பதிவு எண்", "வ.எண்", "வரிசை எண்",
        # English
        "member id", "farmer id", "reg no", "membership no", "id", "s.no", "sl no"
    },
    "society": {
        # Tamil
        "சங்கம்", "பால் சங்கம்", "கிளை சங்கம்", "மையம்", "கூட்டுறவு சங்கம்",
        # English
        "society", "booth", "center", "dairy society", "cooperative"
    }
}


# =============================================================================
# Normalization and Validation Utilities
# =============================================================================

def clean_digits(text: str) -> str:
    """Extract only ASCII digits from text."""
    return re.sub(r"\D", "", text or "")


def is_valid_12digit_cattle_id(candidate: str) -> Tuple[bool, str]:
    """
    Validate 12-digit Pashu Aadhaar / INAPH cattle tag ID.
    Must be treated strictly as a string with leading zeros preserved.
    Returns (is_valid, 12_digit_string).
    """
    digits = clean_digits(candidate)
    if len(digits) == 12:
        return True, digits
    return False, ""


def extract_cattle_ids_with_raw_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Extract one or more 12-digit cattle IDs with their corresponding raw substrings.
    Handles cells with N comma-separated IDs (e.g. 'ID1, ID2, ID3'),
    semicolons, newlines, slashes, or space-grouped 12-digit sequences.
    Strictly preserves 12-digit strings with leading zeros.
    Returns list of (clean_12_digit_id, raw_substring) tuples in order of appearance.
    """
    if not text:
        return []

    results: List[Tuple[str, str]] = []
    seen = set()

    # 1. Delimiter-based splitting (comma, semicolon, newline, slash, pipe)
    # Directly addresses cells with N comma-separated IDs: "ID1, ID2, ID3"
    chunks = re.split(r"[,;\n\r|/]+", text)
    for chunk in chunks:
        c_raw = chunk.strip()
        if not c_raw:
            continue

        chunk_added = False
        # A. Check isolated 12-digit sequence in chunk
        for m in re.finditer(r"\b(\d{12})\b", c_raw):
            tag = m.group(1)
            if tag not in seen:
                seen.add(tag)
                results.append((tag, m.group(0)))
                chunk_added = True

        # B. Check 4-4-4 formatted pattern (e.g. 4006 7854 1362 or 4006-7854-1362)
        if not chunk_added:
            for m in re.finditer(r"\b(\d{4}[\s\-_]+\d{4}[\s\-_]+\d{4})\b", c_raw):
                tag = clean_digits(m.group(1))
                if len(tag) == 12 and tag not in seen:
                    seen.add(tag)
                    results.append((tag, m.group(1).strip()))
                    chunk_added = True

        # C. Check if clean digits in chunk form 12 digits (or multiples of 12)
        if not chunk_added:
            d = clean_digits(c_raw)
            if len(d) == 12:
                if d not in seen:
                    seen.add(d)
                    results.append((d, c_raw))
                    chunk_added = True
            elif len(d) > 12 and len(d) % 12 == 0:
                for k in range(0, len(d), 12):
                    tag = d[k:k+12]
                    if tag not in seen:
                        seen.add(tag)
                        results.append((tag, tag))
                chunk_added = True
            elif len(d) > 12:
                for m in re.finditer(r"(\d{12})", d):
                    tag = m.group(1)
                    if tag not in seen:
                        seen.add(tag)
                        results.append((tag, tag))
                chunk_added = True

    # 2. Text-wide fallback if delimiter splitting produced no results
    if not results:
        for m in re.finditer(r"\b(\d{12})\b", text):
            tag = m.group(1)
            if tag not in seen:
                seen.add(tag)
                results.append((tag, m.group(0)))
        for m in re.finditer(r"\b(\d{4}[\s\-_]+\d{4}[\s\-_]+\d{4})\b", text):
            tag = clean_digits(m.group(1))
            if len(tag) == 12 and tag not in seen:
                seen.add(tag)
                results.append((tag, m.group(1).strip()))
        if not results:
            d = clean_digits(text)
            for m in re.finditer(r"(\d{12})", d):
                tag = m.group(1)
                if tag not in seen:
                    seen.add(tag)
                    results.append((tag, tag))

    return results


def extract_cattle_ids_from_text(text: str) -> List[str]:
    """
    Extract one or more 12-digit cattle IDs from cell text.
    Handles space, comma, newline, or slash separated tag IDs.
    Returns list of clean 12-digit string IDs.
    """
    return [t[0] for t in extract_cattle_ids_with_raw_from_text(text)]


def normalize_indian_phone(candidate: str) -> Tuple[bool, Optional[str]]:
    """
    Validate 10-digit Indian phone number with optional +91 / 0 prefix.
    Returns (is_valid, normalized_10_digit_string).
    """
    digits = clean_digits(candidate)
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return True, digits[2:]
    if len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
        return True, digits[1:]
    if len(digits) == 10 and digits[0] in "6789":
        return True, digits
    return False, None


def parse_milk_quantity(text: str) -> Optional[float]:
    """
    Parse milk quantity in litres.
    Preserves decimals. Empty text returns None (never default to 0.0).
    """
    if not text:
        return None
    cleaned = text.strip().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if m:
        try:
            val = float(m.group(1))
            if 0.1 <= val <= 200.0:  # Reasonable single session/daily supply bounds
                return round(val, 2)
        except ValueError:
            pass
    return None


def parse_cow_count(text: str) -> Optional[int]:
    """Parse declared cow count from integer text."""
    if not text:
        return None
    digits = clean_digits(text)
    if digits and len(digits) <= 3:
        try:
            cnt = int(digits)
            if 1 <= cnt <= 500:
                return cnt
        except ValueError:
            pass
    return None


# =============================================================================
# Semantic Column Classifier
# =============================================================================

class AavinTableMapper:
    """
    Maps table columns and cell data to Aavin Smart Cattle Passport entities:
    Farmers, Cattle Pashu Aadhaar Tags, and Milk Collection records.
    """

    @classmethod
    def match_header_semantic(cls, header_text: str) -> Optional[str]:
        """Match a column header against known Aavin semantic synonyms."""
        if not header_text:
            return None
        norm = header_text.strip().lower()
        # Remove punctuation
        norm_clean = re.sub(r"[\.:,;\-_/]", " ", norm).strip()

        for field_name, synonyms in HEADER_SYNONYMS.items():
            for syn in synonyms:
                if syn in norm or syn in norm_clean:
                    return field_name
        return None

    @classmethod
    def infer_column_mapping(
        cls,
        headers: List[str],
        sample_rows: List[List[str]],
        schema_hint: Optional[Dict[str, str]] = None
    ) -> Dict[int, str]:
        """
        Determine field type for each column index using headers, data types, and schema hints.
        Returns mapping: {col_idx: field_name}.
        """
        col_mapping: Dict[int, str] = {}
        col_count = len(headers) if headers else (len(sample_rows[0]) if sample_rows else 0)

        # 1. Apply schema hints if provided
        if schema_hint:
            for k, field_name in schema_hint.items():
                try:
                    c_idx = int(k)
                    if 0 <= c_idx < col_count:
                        col_mapping[c_idx] = field_name
                except ValueError:
                    pass

        # 2. Match from explicit headers
        for idx in range(col_count):
            if idx in col_mapping:
                continue
            h_text = headers[idx] if idx < len(headers) else ""
            matched = cls.match_header_semantic(h_text)
            if matched:
                col_mapping[idx] = matched

        # 3. Analyze data distribution for unmapped columns
        for idx in range(col_count):
            if idx in col_mapping:
                continue

            values = [row[idx] for row in sample_rows if idx < len(row) and row[idx]]
            if not values:
                continue

            phone_matches = sum(1 for v in values if normalize_indian_phone(v)[0])
            cattle_matches = sum(1 for v in values if len(extract_cattle_ids_from_text(v)) > 0)
            milk_matches = sum(1 for v in values if parse_milk_quantity(v) is not None)

            total = max(1, len(values))
            if (cattle_matches / total) >= 0.40:
                col_mapping[idx] = "cattle_id"
            elif (phone_matches / total) >= 0.40:
                col_mapping[idx] = "phone"
            elif (milk_matches / total) >= 0.60:
                col_mapping[idx] = "milk_quantity"

        return col_mapping

    @classmethod
    def extract_candidate_records(
        cls,
        table: TableResult,
        schema_hint: Optional[Dict[str, str]] = None
    ) -> Tuple[List[FarmerCandidateRecord], List[str]]:
        """
        Convert a reconstructed table into grouped Farmer & Cattle candidate records.
        Handles continuation rows, multiple cattle IDs per cell, and uncertain associations.
        """
        headers = table.headers or []
        rows = table.rows or []
        if not rows:
            return [], []

        col_mapping = cls.infer_column_mapping(headers, rows, schema_hint=schema_hint)
        logger.info(f"Table {table.table_index} inferred semantic column mapping: {col_mapping}")

        records: List[FarmerCandidateRecord] = []
        unassigned_cattle: List[str] = []
        current_farmer: Optional[FarmerCandidateRecord] = None

        for r_idx, row in enumerate(rows):
            # Extract fields for this row
            row_farmer_name: Optional[str] = None
            row_phone: Optional[str] = None
            row_raw_phone: Optional[str] = None
            row_village: Optional[str] = None
            row_member_id: Optional[str] = None
            row_cow_count: Optional[int] = None
            row_breed: Optional[str] = None
            row_cattle_pairs: List[Tuple[str, str]] = []
            row_milk_qty: Optional[float] = None
            row_date: Optional[str] = None
            row_session: Optional[str] = None

            for c_idx, val in enumerate(row):
                field_name = col_mapping.get(c_idx)
                if not field_name or not val:
                    continue
                v_clean = str(val).strip()
                if not v_clean:
                    continue

                if field_name == "farmer_name":
                    row_farmer_name = v_clean
                elif field_name == "phone":
                    is_p, p_norm = normalize_indian_phone(v_clean)
                    row_raw_phone = v_clean
                    row_phone = p_norm if is_p else None
                elif field_name == "village":
                    row_village = v_clean
                elif field_name == "member_id":
                    row_member_id = v_clean
                elif field_name == "cow_count":
                    row_cow_count = parse_cow_count(v_clean)
                elif field_name == "breed":
                    row_breed = v_clean
                elif field_name == "cattle_id":
                    row_cattle_pairs.extend(extract_cattle_ids_with_raw_from_text(v_clean))
                elif field_name == "milk_quantity":
                    row_milk_qty = parse_milk_quantity(v_clean)
                elif field_name == "date":
                    row_date = v_clean
                elif field_name == "session":
                    row_session = v_clean

            # Check if this row is a new farmer entry or a continuation row
            is_new_farmer = bool(row_farmer_name or row_phone or (row_member_id and not row_cattle_pairs))

            if is_new_farmer:
                # Commit previous farmer record if any
                if current_farmer is not None:
                    # Synchronize declared cow count with actual cattle count (1 ID = 1 cow, N IDs = N cows)
                    if current_farmer.cattle and (current_farmer.cow_count is None or current_farmer.cow_count < len(current_farmer.cattle)):
                        current_farmer.cow_count = len(current_farmer.cattle)
                    records.append(current_farmer)

                record_id = len(records) + 1
                warnings = []
                if row_raw_phone and not row_phone:
                    warnings.append(f"Phone '{row_raw_phone}' could not be normalized to 10 digits")

                current_farmer = FarmerCandidateRecord(
                    record_id=record_id,
                    farmer_name=row_farmer_name,
                    raw_farmer_name=row_farmer_name,
                    phone=row_phone,
                    raw_phone=row_raw_phone,
                    village=row_village,
                    cow_count=row_cow_count,
                    member_id=row_member_id,
                    cattle_ids=[],
                    cattle=[],
                    milk_records=[],
                    source_row_indices=[r_idx],
                    confidence=table.confidence or 0.90,
                    review_required=True,
                    validation_warnings=warnings
                )

            # Associate cattle IDs: exactly 1 ID number = 1 cow, N IDs = N cows
            if row_cattle_pairs:
                if current_farmer is not None:
                    if r_idx not in current_farmer.source_row_indices:
                        current_farmer.source_row_indices.append(r_idx)

                    # Parse comma-separated breeds if present
                    breed_tokens = [b.strip() for b in re.split(r"[,;\n\r]+", row_breed)] if row_breed else []

                    for k, (cid, raw_cid) in enumerate(row_cattle_pairs):
                        if cid not in current_farmer.cattle_ids:
                            # 1 ID = 1 cow
                            current_farmer.cattle_ids.append(cid)
                            cow_seq = len(current_farmer.cattle) + 1

                            # Distribute breed matching 1-to-1 if multiple breeds present,
                            # or assign shared breed if single breed specified
                            if k < len(breed_tokens):
                                assigned_breed = breed_tokens[k]
                            elif len(breed_tokens) == 1:
                                assigned_breed = breed_tokens[0]
                            else:
                                assigned_breed = None

                            current_farmer.cattle.append(CattleCandidateRecord(
                                cow_index=cow_seq,
                                tag_id=cid,
                                raw_tag_id=raw_cid,
                                breed=assigned_breed,
                                status="candidate",
                                review_required=True,
                                confidence=0.92,
                                warnings=[]
                            ))

                    # Synchronize cow count (1 ID for 1 cow, N IDs for N cows)
                    if current_farmer.cattle:
                        if current_farmer.cow_count is None or current_farmer.cow_count < len(current_farmer.cattle):
                            current_farmer.cow_count = len(current_farmer.cattle)
                else:
                    # No active farmer: store as unassigned candidate
                    for cid, _ in row_cattle_pairs:
                        if cid not in unassigned_cattle:
                            unassigned_cattle.append(cid)

            # Associate milk record
            if row_milk_qty is not None and current_farmer is not None:
                current_farmer.milk_records.append(MilkCandidateRecord(
                    date=row_date,
                    session=row_session,
                    litres=row_milk_qty,
                    confidence=0.90,
                    review_required=True,
                    warnings=[]
                ))

        # Commit final farmer
        if current_farmer is not None:
            if current_farmer.cattle and (current_farmer.cow_count is None or current_farmer.cow_count < len(current_farmer.cattle)):
                current_farmer.cow_count = len(current_farmer.cattle)
            records.append(current_farmer)

        return records, unassigned_cattle


# Global singleton instance
aavin_mapper = AavinTableMapper()
