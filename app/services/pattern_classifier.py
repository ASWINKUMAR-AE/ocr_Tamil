"""
Pattern-Based Value Classification and Semantic Recognition Engine.
Provides configurable, reusable, and extensible pattern rules for identifying
data types in table cells across Tamil, English, and multilingual documents:
  - Phone numbers (10-digit Indian mobile, optional +91/91 prefix)
  - Cattle Aadhaar / Pashu Aadhaar IDs (12-digit, context-dependent)
  - Dates (DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, and contextual subtypes)
  - Serial numbers & Registration identifiers (leading-zero safe)
  - Person names (student, farmer, father, general)
  - Addresses & Localities
  - Quantities, Measurements, and Units (milk litres, weight kg, counts)
  - Amounts, Currency, and Percentages
  - Additional types: Email, Postal PIN code, GSTIN, Vehicle, Invoice numbers
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ClassificationResult:
    """Detailed classification output for a single cell value or column distribution."""
    original_text: str
    normalized_value: str
    display_text: str
    semantic_type: str
    confidence: float
    display_label_ta: str
    display_label_en: str
    inference_method: str = "pattern"
    context_hints: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# Context keywords indicating cattle/livestock/dairy domain
CATTLE_CONTEXT_KEYWORDS = {
    # Tamil
    "மாடு", "கால்நடை", "பசு", "பால்", "எருமை", "பண்ணை", "ஆதார்", "கொள்முதல்", "கன்று",
    # English
    "cattle", "cow", "bovine", "animal", "ear tag", "tag no", "pashu", "aadhaar",
    "livestock", "dairy", "buffalo", "calf", "farm"
}

# Context keywords indicating student/school domain
STUDENT_CONTEXT_KEYWORDS = {
    # Tamil
    "மாணவர்", "மாணவி", "பள்ளி", "வகுப்பு", "சேர்க்கை", "கல்வி", "தேர்ச்சி",
    # English
    "student", "school", "class", "admission", "roll", "academic", "exam"
}

# Context keywords indicating agriculture/farming domain
FARMER_CONTEXT_KEYWORDS = {
    # Tamil
    "விவசாயி", "விவசாயம்", "பயிர்", "நிலம்", "பாசனம்", "விளைச்சல்",
    # English
    "farmer", "agriculture", "crop", "land", "survey", "yield"
}


# =============================================================================
# Helper Functions for Normalization and Validation
# =============================================================================

def clean_digits(text: str) -> str:
    """Extract only numeric digits from text preserving leading zeros."""
    return re.sub(r"\D", "", text)


def is_valid_indian_phone(text: str) -> Tuple[bool, str]:
    """
    Validate 10-digit Indian mobile number with optional +91 or 91 country code.
    Returns (is_valid, normalized_10_digit_string).
    """
    cleaned = text.strip()
    digits = clean_digits(cleaned)

    # Strip +91 or 91 prefix if 12 digits
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        normalized = digits[2:]
        return True, normalized
    
    # Strip leading 0 if 11 digits
    if len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
        normalized = digits[1:]
        return True, normalized

    # Standard 10-digit mobile
    if len(digits) == 10 and digits[0] in "6789":
        return True, digits

    # Spaced pattern like '98431 23456'
    m = re.search(r"\b([6-9]\d{4})\s*(\d{5})\b", cleaned)
    if m:
        normalized = f"{m.group(1)}{m.group(2)}"
        return True, normalized

    return False, ""


def is_valid_date_format(text: str) -> Tuple[bool, str]:
    """
    Recognize common date representations: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD.
    Validates calendar boundaries (day 1-31, month 1-12).
    """
    cleaned = text.strip()
    
    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    m1 = re.search(r"\b(0?[1-9]|[12]\d|3[01])[./-](0?[1-9]|1[0-2])[./-](\d{2,4})\b", cleaned)
    if m1:
        d, m, y = m1.group(1), m1.group(2), m1.group(3)
        return True, f"{d.zfill(2)}/{m.zfill(2)}/{y}"

    # YYYY-MM-DD
    m2 = re.search(r"\b(\d{4})[./-](0?[1-9]|1[0-2])[./-](0?[1-9]|[12]\d|3[01])\b", cleaned)
    if m2:
        y, m, d = m2.group(1), m2.group(2), m2.group(3)
        return True, f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    return False, ""


def is_valid_pin_code(text: str) -> Tuple[bool, str]:
    """Validate 6-digit Indian postal PIN code (e.g., 600001)."""
    cleaned = clean_digits(text)
    if len(cleaned) == 6 and cleaned[0] in "123456789":
        return True, cleaned
    return False, ""


def is_valid_gstin(text: str) -> Tuple[bool, str]:
    """Validate 15-character Indian GSTIN format."""
    cleaned = re.sub(r"[\s-]", "", text).upper()
    if re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]{1}$", cleaned):
        return True, cleaned
    return False, ""


def is_valid_email(text: str) -> Tuple[bool, str]:
    """Validate email address format."""
    cleaned = text.strip()
    if re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", cleaned):
        return True, cleaned.lower()
    return False, ""


# =============================================================================
# Pattern Classifier Engine
# =============================================================================

class PatternClassifier:
    """
    Configurable pattern recognition engine. Inspects cell values and sets of column
    values to determine likely semantic types and labels without rigid hardcoded schemas.
    """

    def classify_single_value(
        self,
        text: str,
        column_context: Optional[str] = None,
        document_context: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify a single cell string into a semantic type candidate.
        Preserves original_text exactly without modification.
        """
        raw = text.strip()
        norm_val = raw
        combined_ctx = f"{column_context or ''} {document_context or ''}".lower()

        # 1. Serial Number (1, 2., 3), 01, etc.)
        if re.match(r"^\d{1,4}[.)]?$", raw):
            return ClassificationResult(
                original_text=raw,
                normalized_value=clean_digits(raw),
                display_text=raw,
                semantic_type="serial_number",
                confidence=0.92,
                display_label_ta="வரிசை எண்",
                display_label_en="Serial No"
            )

        # 2. Date Detection
        is_date, norm_date = is_valid_date_format(raw)
        if is_date:
            sem_type = "date"
            lbl_ta = "தேதி"
            lbl_en = "Date"

            if any(k in combined_ctx for k in ["birth", "dob", "பிறந்த", "வயது"]):
                sem_type = "birth_date"
                lbl_ta = "பிறந்த தேதி"
                lbl_en = "Date of Birth"
            elif any(k in combined_ctx for k in ["கொள்முதல்", "collection"]):
                sem_type = "collection_date"
                lbl_ta = "கொள்முதல் தேதி"
                lbl_en = "Collection Date"
            elif any(k in combined_ctx for k in ["பரிவர்த்தனை", "transaction"]):
                sem_type = "transaction_date"
                lbl_ta = "பரிவர்த்தனை தேதி"
                lbl_en = "Transaction Date"
            elif any(k in combined_ctx for k in ["சேர்க்கை", "பதிவு", "registration", "admission"]):
                sem_type = "registration_date"
                lbl_ta = "பதிவு தேதி"
                lbl_en = "Registration Date"

            return ClassificationResult(
                original_text=raw,
                normalized_value=norm_date,
                display_text=raw,
                semantic_type=sem_type,
                confidence=0.94,
                display_label_ta=lbl_ta,
                display_label_en=lbl_en
            )

        # 3. Phone Number (10 digits starting with 6-9)
        # Contextual check: Do not treat as phone if header explicitly says account / registration
        is_phone, norm_phone = is_valid_indian_phone(raw)
        if is_phone:
            # Check if context strongly implies non-phone identifier (e.g. account number)
            if any(k in combined_ctx for k in ["account", "வங்கி", "கணக்கு", "acc", "policy"]):
                return ClassificationResult(
                    original_text=raw,
                    normalized_value=norm_phone,
                    display_text=raw,
                    semantic_type="account_number",
                    confidence=0.85,
                    display_label_ta="வங்கி கணக்கு எண்",
                    display_label_en="Account Number",
                    warnings=["10-digit number matches phone pattern but context suggests account number."]
                )

            return ClassificationResult(
                original_text=raw,
                normalized_value=norm_phone,
                display_text=raw,
                semantic_type="phone_number",
                confidence=0.92,
                display_label_ta="தொலைபேசி எண்",
                display_label_en="Phone Number"
            )

        # 4. Cattle Aadhaar ID / Pashu Aadhaar (12 digits in cattle context)
        # Rule: 12-digit number in cattle context = cattle_aadhaar_id.
        # In non-cattle context = generic identifier / aadhaar.
        digits = clean_digits(raw)
        if len(digits) == 12:
            is_cattle_context = any(k in combined_ctx for k in CATTLE_CONTEXT_KEYWORDS)
            if is_cattle_context:
                return ClassificationResult(
                    original_text=raw,
                    normalized_value=digits,  # preserves leading zeros
                    display_text=raw,
                    semantic_type="cattle_aadhaar_id",
                    confidence=0.90,
                    display_label_ta="கால்நடை ஆதார் எண்",
                    display_label_en="Cattle Aadhaar ID",
                    context_hints={"cattle_context_matched": True}
                )
            else:
                return ClassificationResult(
                    original_text=raw,
                    normalized_value=digits,
                    display_text=raw,
                    semantic_type="identifier",
                    confidence=0.80,
                    display_label_ta="அடையாள எண்",
                    display_label_en="Identifier",
                    context_hints={"twelve_digit": True}
                )

        # 5. Currency / Amount
        if re.search(r"[₹$]|ரூ|\b\d+[.,]\d{2}\b|\bRs\.?\s*\d+", raw, re.I):
            return ClassificationResult(
                original_text=raw,
                normalized_value=raw,
                display_text=raw,
                semantic_type="amount",
                confidence=0.88,
                display_label_ta="தொகை",
                display_label_en="Amount"
            )

        # 6. Quantities & Measurements (Litres, kg, units, counts)
        if re.search(r"\b\d+(\.\d+)?\s*(லி|லிட்டர்|l|ltr|litres)\b", raw, re.I):
            return ClassificationResult(
                original_text=raw,
                normalized_value=raw,
                display_text=raw,
                semantic_type="quantity",
                confidence=0.90,
                display_label_ta="பால் அளவு",
                display_label_en="Milk Quantity"
            )
        if re.search(r"\b\d+(\.\d+)?\s*(kg|கிலோ|gm|கிராம்)\b", raw, re.I):
            return ClassificationResult(
                original_text=raw,
                normalized_value=raw,
                display_text=raw,
                semantic_type="weight",
                confidence=0.88,
                display_label_ta="எடை",
                display_label_en="Weight"
            )
        if re.search(r"\b\d+(\.\d+)?%", raw):
            return ClassificationResult(
                original_text=raw,
                normalized_value=raw,
                display_text=raw,
                semantic_type="percentage",
                confidence=0.90,
                display_label_ta="சதவீதம்",
                display_label_en="Percentage"
            )

        # 7. Postal PIN Code
        is_pin, norm_pin = is_valid_pin_code(raw)
        if is_pin and any(k in combined_ctx for k in ["pin", "postal", "அஞ்சல்", "முகவரி", "address"]):
            return ClassificationResult(
                original_text=raw,
                normalized_value=norm_pin,
                display_text=raw,
                semantic_type="pin_code",
                confidence=0.92,
                display_label_ta="அஞ்சல் குறியீடு",
                display_label_en="PIN Code"
            )

        # 8. Email Address
        is_em, norm_em = is_valid_email(raw)
        if is_em:
            return ClassificationResult(
                original_text=raw,
                normalized_value=norm_em,
                display_text=raw,
                semantic_type="email",
                confidence=0.98,
                display_label_ta="மின்னஞ்சல்",
                display_label_en="Email"
            )

        # 9. GSTIN Identifier
        is_gst, norm_gst = is_valid_gstin(raw)
        if is_gst:
            return ClassificationResult(
                original_text=raw,
                normalized_value=norm_gst,
                display_text=raw,
                semantic_type="gstin",
                confidence=0.96,
                display_label_ta="ஜிஎஸ்டி எண்",
                display_label_en="GSTIN"
            )

        # 10. Gender
        if any(g in raw.lower() for g in ["ஆண்", "பெண்", "ஆண", "male", "female", "transgender"]):
            return ClassificationResult(
                original_text=raw,
                normalized_value=raw,
                display_text=raw,
                semantic_type="gender",
                confidence=0.92,
                display_label_ta="பாலினம்",
                display_label_en="Gender"
            )

        # 11. Address & Locality Text
        if any(kw in raw for kw in ["தெரு", "சாலை", "நகர்", "வீதி", "கிராமம்", "காலனி", "அக்ரஹாரம்", "street", "road", "nagar", "lane", "colony", "village"]):
            return ClassificationResult(
                original_text=raw,
                normalized_value=raw,
                display_text=raw,
                semantic_type="address",
                confidence=0.88,
                display_label_ta="முகவரி",
                display_label_en="Address"
            )

        # 12. Alphanumeric / Registration Identifiers (e.g. 15A01, REG-101)
        if re.search(r"\b[0-9]{2,}[A-Za-z][A-Za-z0-9]*\b|\b[A-Za-z]+[-_0-9]+\b|\b\d{4,8}\b", raw):
            sem_type = "identifier"
            lbl_ta = "பதிவு எண்"
            lbl_en = "Register No"
            if any(k in combined_ctx for k in STUDENT_CONTEXT_KEYWORDS):
                sem_type = "student_id"
                lbl_ta = "மாணவர் பதிவு எண்"
                lbl_en = "Student ID"
            elif any(k in combined_ctx for k in CATTLE_CONTEXT_KEYWORDS):
                sem_type = "cattle_id"
                lbl_ta = "மாடு எண்"
                lbl_en = "Cattle ID"

            return ClassificationResult(
                original_text=raw,
                normalized_value=raw,
                display_text=raw,
                semantic_type=sem_type,
                confidence=0.84,
                display_label_ta=lbl_ta,
                display_label_en=lbl_en
            )

        # 13. Person Names
        # If text consists of words without numbers/symbols and has reasonable length
        if re.match(r"^[\u0b80-\u0bffA-Za-z\s.'-]{2,40}$", raw) and not any(k in raw for k in ["தெரு", "சாலை", "நகர்", "வீதி"]):
            sem_type = "person_name"
            lbl_ta = "பெயர்"
            lbl_en = "Name"

            if any(k in combined_ctx for k in ["தந்தை", "father"]):
                sem_type = "father_name"
                lbl_ta = "தந்தை பெயர்"
                lbl_en = "Father's Name"
            elif any(k in combined_ctx for k in STUDENT_CONTEXT_KEYWORDS):
                sem_type = "student_name"
                lbl_ta = "மாணவர் பெயர்"
                lbl_en = "Student Name"
            elif any(k in combined_ctx for k in FARMER_CONTEXT_KEYWORDS):
                sem_type = "farmer_name"
                lbl_ta = "விவசாயி பெயர்"
                lbl_en = "Farmer Name"

            return ClassificationResult(
                original_text=raw,
                normalized_value=raw,
                display_text=raw,
                semantic_type=sem_type,
                confidence=0.75,
                display_label_ta=lbl_ta,
                display_label_en=lbl_en
            )

        # Default Unknown / Generic
        return ClassificationResult(
            original_text=raw,
            normalized_value=raw,
            display_text=raw,
            semantic_type="unknown",
            confidence=0.50,
            display_label_ta="விவரம்",
            display_label_en="Details"
        )

    def analyze_column_distribution(
        self,
        column_values: List[str],
        header_text: Optional[str] = None,
        document_context: Optional[str] = None,
        preferred_language: str = "ta"
    ) -> ClassificationResult:
        """
        Analyze all values in a physical table column to infer the dominant semantic type.
        Calculates consistency confidence across multiple rows.
        Resolves conflicts between explicit header and pattern distributions.
        """
        clean_vals = [v.strip() for v in column_values if v and v.strip()]
        val_count = len(clean_vals)
        raw_header = (header_text or "").strip()
        lower_h = raw_header.lower()
        combined_ctx = f"{raw_header} {document_context or ''}".lower()

        # Step 1: Check Explicit Header (Priority 2)
        has_explicit_header = bool(raw_header and not re.match(r"^(column|field|col)\s*\d+$", lower_h))
        explicit_type = None
        explicit_lbl_ta = raw_header
        explicit_lbl_en = raw_header

        if has_explicit_header:
            # Map explicit keywords to types
            if any(k in lower_h for k in ["வரிசை", "வ.எண்", "sl", "s.no", "serial"]):
                explicit_type = "serial_number"
                explicit_lbl_ta, explicit_lbl_en = "வரிசை எண்", "Serial No"
            elif any(k in lower_h for k in ["பிறந்த", "dob", "birth date", "date of birth"]):
                explicit_type = "birth_date"
                explicit_lbl_ta, explicit_lbl_en = "பிறந்த தேதி", "Date of Birth"
            elif any(k in lower_h for k in ["தேதி", "date"]):
                explicit_type = "date"
                explicit_lbl_ta, explicit_lbl_en = "தேதி", "Date"
            elif any(k in lower_h for k in ["தொலைபேசி", "செல்பேசி", "கைபேசி", "phone", "mobile", "contact"]):
                explicit_type = "phone_number"
                explicit_lbl_ta, explicit_lbl_en = "தொலைபேசி எண்", "Phone Number"
            elif any(k in lower_h for k in ["மாடு எண்", "மாடு", "கால்நடை", "பசு", "cattle", "cow", "tag no", "cow id", "pashu", "aadhaar"]):
                explicit_type = "cattle_aadhaar_id"
                explicit_lbl_ta, explicit_lbl_en = "கால்நடை ஆதார் எண்", "Cattle Aadhaar ID"
            elif any(k in lower_h for k in ["பால்", "milk", "பால் அளவு", "milk qty"]):
                explicit_type = "quantity"
                explicit_lbl_ta, explicit_lbl_en = "பால் அளவு", "Milk Quantity"
            elif any(k in lower_h for k in ["அளவு", "qty", "quantity"]):
                explicit_type = "quantity"
                explicit_lbl_ta, explicit_lbl_en = "அளவு", "Quantity"
            elif any(k in lower_h for k in ["தொகை", "விலை", "amount", "price", "rate", "cost", "total"]):
                explicit_type = "amount"
                explicit_lbl_ta, explicit_lbl_en = "தொகை", "Amount"
            elif any(k in lower_h for k in ["பாலினம்", "gender", "sex"]):
                explicit_type = "gender"
                explicit_lbl_ta, explicit_lbl_en = "பாலினம்", "Gender"
            elif any(k in lower_h for k in ["முகவரி", "address"]):
                explicit_type = "address"
                explicit_lbl_ta, explicit_lbl_en = "முகவரி", "Address"
            elif any(k in lower_h for k in ["தந்தை", "father"]):
                explicit_type = "father_name"
                explicit_lbl_ta, explicit_lbl_en = "தந்தை பெயர்", "Father's Name"
            elif any(k in lower_h for k in ["மாணவர்", "student"]):
                explicit_type = "student_name"
                explicit_lbl_ta, explicit_lbl_en = "மாணவர் பெயர்", "Student Name"
            elif any(k in lower_h for k in ["பெயர்", "name"]):
                explicit_type = "person_name"
                explicit_lbl_ta, explicit_lbl_en = "பெயர்", "Name"
            elif any(k in lower_h for k in ["பதிவு", "சேர்க்கை", "reg", "admission", "id", "code"]):
                explicit_type = "identifier"
                explicit_lbl_ta, explicit_lbl_en = "பதிவு எண்", "Register No"
            elif any(k in lower_h for k in ["கணக்கு", "account", "acc"]):
                explicit_type = "account_number"
                explicit_lbl_ta, explicit_lbl_en = "வங்கி கணக்கு எண்", "Account Number"
            else:
                explicit_type = "explicit_text"
                explicit_lbl_ta, explicit_lbl_en = raw_header, raw_header

        # Step 2: Analyze Column Data Distribution (Priority 3)
        type_counts: Dict[str, int] = {}
        results_list: List[ClassificationResult] = []

        for val in clean_vals:
            res = self.classify_single_value(val, column_context=raw_header, document_context=document_context)
            results_list.append(res)
            type_counts[res.semantic_type] = type_counts.get(res.semantic_type, 0) + 1

        # Determine dominant pattern type
        pattern_dominant_type = "unknown"
        pattern_dominant_ratio = 0.0
        if val_count > 0:
            top_type, top_cnt = max(type_counts.items(), key=lambda item: item[1])
            pattern_dominant_type = top_type
            pattern_dominant_ratio = top_cnt / float(val_count)

        # Step 3: Conflict Resolution (Section 8)
        # If explicit header exists, do NOT blindly overwrite it with pattern!
        warnings: List[str] = []
        if has_explicit_header:
            final_type = explicit_type or "unknown"
            display_h = raw_header
            confidence = 0.95
            method = "explicit_header"

            # Check if explicit header conflicts with raw patterns:
            # e.g. Header is Account Number, but raw values match 10-digit phone
            raw_phone_matches = sum(1 for v in clean_vals if is_valid_indian_phone(v)[0])
            if val_count > 0 and (raw_phone_matches / float(val_count)) >= 0.6 and explicit_type != "phone_number":
                warnings.append(
                    f"Column '{raw_header}' contains data matching 'phone_number' ({int(raw_phone_matches / val_count * 100)}% consistency); preserved explicit header."
                )
            elif pattern_dominant_type not in ["unknown", "person_name", "explicit_text"] and pattern_dominant_ratio >= 0.6:
                if pattern_dominant_type != explicit_type:
                    # Record the conflicting pattern in warnings without overwriting header
                    warnings.append(
                        f"Column '{raw_header}' contains data matching '{pattern_dominant_type}' ({int(pattern_dominant_ratio * 100)}% consistency); preserved explicit header."
                    )

            lbl_ta = explicit_lbl_ta
            lbl_en = explicit_lbl_en

            return ClassificationResult(
                original_text=raw_header,
                normalized_value=raw_header,
                display_text=display_h,
                semantic_type=final_type,
                confidence=confidence,
                display_label_ta=lbl_ta,
                display_label_en=lbl_en,
                inference_method=method,
                warnings=warnings
            )

        # Step 4: Level B Semantic Inference when header is missing or generic (Priority 3 & 4)
        if pattern_dominant_ratio >= 0.45 and pattern_dominant_type != "unknown":
            sample_res = next(r for r in results_list if r.semantic_type == pattern_dominant_type)
            lbl = sample_res.display_label_en if preferred_language == "en" else sample_res.display_label_ta
            conf = min(0.92, round(0.70 + (pattern_dominant_ratio * 0.25), 2))

            return ClassificationResult(
                original_text=raw_header or f"Column",
                normalized_value="",
                display_text=lbl,
                semantic_type=pattern_dominant_type,
                confidence=conf,
                display_label_ta=sample_res.display_label_ta,
                display_label_en=sample_res.display_label_en,
                inference_method="pattern_and_context"
            )

        # Step 5: Level C Position Fallback (Priority 5)
        fallback_label = raw_header if raw_header else "Column"
        return ClassificationResult(
            original_text=raw_header or "",
            normalized_value="",
            display_text=fallback_label,
            semantic_type="unknown",
            confidence=0.50,
            display_label_ta=fallback_label,
            display_label_en=fallback_label,
            inference_method="position_fallback"
        )


# Global singleton instance
pattern_classifier = PatternClassifier()
