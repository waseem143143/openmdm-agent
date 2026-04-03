"""
openmdm-agent · mdm/loader.py
Ingests raw patient CSV records, normalizes key fields,
and returns a list of PatientRecord objects.
"""

from __future__ import annotations
import re
import uuid
import pandas as pd
from typing import Optional
from mdm.models import PatientRecord


# ─────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────

def normalize_name(value: Optional[str]) -> Optional[str]:
    """Uppercase, strip whitespace, remove punctuation."""
    if not value:
        return None
    return re.sub(r"[^A-Z\s]", "", value.upper().strip())


def normalize_phone(value: Optional[str]) -> Optional[str]:
    """Strip all non-digits → 10-digit string or None."""
    if not value:
        return None
    digits = re.sub(r"\D", "", str(value))
    return digits if len(digits) == 10 else None


def normalize_zip(value: Optional[str]) -> Optional[str]:
    """Keep first 5 digits only."""
    if not value:
        return None
    digits = re.sub(r"\D", "", str(value))
    return digits[:5] if len(digits) >= 5 else None


def normalize_address(value: Optional[str]) -> Optional[str]:
    """Uppercase, expand common abbreviations, strip extra whitespace."""
    if not value:
        return None
    abbrevs = {
        r"\bST\b":  "STREET",
        r"\bAVE\b": "AVENUE",
        r"\bRD\b":  "ROAD",
        r"\bDR\b":  "DRIVE",
        r"\bBLVD\b":"BOULEVARD",
        r"\bLN\b":  "LANE",
        r"\bCT\b":  "COURT",
        r"\bPL\b":  "PLACE",
        r"\bAPT\b": "APARTMENT",
    }
    text = value.upper().strip()
    for pattern, replacement in abbrevs.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text)


def normalize_gender(value: Optional[str]) -> Optional[str]:
    """Map common variants → M / F / O / U."""
    if not value:
        return "U"
    mapping = {"M": "M", "MALE": "M", "F": "F", "FEMALE": "F",
               "O": "O", "OTHER": "O", "U": "U", "UNKNOWN": "U"}
    return mapping.get(str(value).upper().strip(), "U")


def safe_str(value) -> Optional[str]:
    """Convert NaN / None / empty to None, otherwise string."""
    if pd.isna(value) or str(value).strip() in ("", "nan", "NaN", "None"):
        return None
    return str(value).strip()


# ─────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────

def load_patients_from_csv(filepath: str) -> list[PatientRecord]:
    """
    Read a CSV file and return normalized PatientRecord objects.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of PatientRecord instances, fully normalized.
    """
    df = pd.read_csv(filepath, dtype=str)
    records: list[PatientRecord] = []

    for _, row in df.iterrows():
        raw_first  = safe_str(row.get("first_name"))
        raw_last   = safe_str(row.get("last_name"))
        raw_phone  = safe_str(row.get("phone"))
        raw_zip    = safe_str(row.get("zip_code"))
        raw_addr   = safe_str(row.get("address_line1"))
        raw_gender = safe_str(row.get("gender"))

        # Source priority: EHR=1, CLAIMS=3, LAB=4, PHARMACY=5, REGISTRY=6
        try:
            priority = int(safe_str(row.get("source_priority")) or 5)
        except ValueError:
            priority = 5

        record = PatientRecord(
            record_id       = safe_str(row.get("record_id")) or str(uuid.uuid4()),
            source_system   = safe_str(row.get("source_system")) or "UNKNOWN",
            source_priority = priority,

            first_name      = raw_first,
            last_name       = raw_last,
            middle_name     = safe_str(row.get("middle_name")),
            date_of_birth   = safe_str(row.get("date_of_birth")),
            gender          = normalize_gender(raw_gender),
            ssn_last4       = safe_str(row.get("ssn_last4")),

            address_line1   = raw_addr,
            city            = safe_str(row.get("city")),
            state           = safe_str(row.get("state")),
            zip_code        = raw_zip,
            phone           = raw_phone,
            email           = safe_str(row.get("email")),

            mrn             = safe_str(row.get("mrn")),
            insurance_id    = safe_str(row.get("insurance_id")),

            record_created  = safe_str(row.get("record_created")),
            record_updated  = safe_str(row.get("record_updated")),

            # Normalized fields (used by matcher)
            normalized_first = normalize_name(raw_first),
            normalized_last  = normalize_name(raw_last),
            normalized_phone = normalize_phone(raw_phone),
            normalized_zip   = normalize_zip(raw_zip),
        )
        records.append(record)

    return records


def records_to_dataframe(records: list[PatientRecord]) -> pd.DataFrame:
    """
    Convert a list of PatientRecord objects into a Pandas DataFrame.
    Useful for display in the Streamlit UI.
    """
    rows = []
    for r in records:
        rows.append({
            "record_id":    r.record_id,
            "source":       r.source_system,
            "priority":     r.source_priority,
            "first_name":   r.first_name,
            "last_name":    r.last_name,
            "dob":          r.date_of_birth,
            "gender":       r.gender,
            "phone":        r.phone,
            "email":        r.email,
            "address":      r.address_line1,
            "city":         r.city,
            "state":        r.state,
            "zip":          r.zip_code,
            "mrn":          r.mrn,
            "insurance_id": r.insurance_id,
            "updated":      r.record_updated,
        })
    return pd.DataFrame(rows)
