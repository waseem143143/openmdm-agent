"""
openmdm-agent · mdm/quality.py
Scores each GoldenRecord across three dimensions:
  - Completeness : % of key fields populated
  - Validity     : % of populated fields passing format rules
  - Overall      : weighted composite score

Scores are 0.0 – 1.0. Results are stored back onto the GoldenRecord.
"""

from __future__ import annotations
import re
from mdm.models import GoldenRecord


# ─────────────────────────────────────────────
# Field definitions
# ─────────────────────────────────────────────

# Fields that must be populated for a "complete" patient identity
REQUIRED_FIELDS = [
    "first_name", "last_name", "date_of_birth", "gender",
    "phone", "address_line1", "city", "state", "zip_code",
]

OPTIONAL_FIELDS = ["middle_name", "email", "ssn_last4", "mrn", "insurance_id"]

# Validity rules: field → regex pattern
VALIDITY_RULES = {
    "date_of_birth": r"^\d{4}-\d{2}-\d{2}$",
    "zip_code":       r"^\d{5}$",
    "phone":          r"^\d{10}$",
    "email":          r"^[\w.\-+]+@[\w\-]+\.[a-z]{2,}$",
    "ssn_last4":      r"^\d{4}$",
    "gender":         r"^[MFOU]$",
    "state":          r"^[A-Z]{2}$",
}


# ─────────────────────────────────────────────
# Dimension scorers
# ─────────────────────────────────────────────

def score_completeness(record: GoldenRecord) -> float:
    """
    % of REQUIRED fields that are non-null and non-empty.
    Optional fields contribute a small bonus (up to +0.1).
    """
    required_filled = sum(
        1 for f in REQUIRED_FIELDS
        if getattr(record, f, None)
    )
    base = required_filled / len(REQUIRED_FIELDS)

    optional_filled = sum(
        1 for f in OPTIONAL_FIELDS
        if getattr(record, f, None)
    )
    bonus = (optional_filled / len(OPTIONAL_FIELDS)) * 0.1

    return round(min(base + bonus, 1.0), 3)


def score_validity(record: GoldenRecord) -> float:
    """
    Of all fields with a validation rule that are populated,
    what % pass their regex rule?
    """
    applicable = 0
    passing    = 0

    for field, pattern in VALIDITY_RULES.items():
        value = getattr(record, field, None)
        if not value:
            continue
        applicable += 1
        if re.match(pattern, str(value).strip(), re.IGNORECASE):
            passing += 1

    return round(passing / applicable, 3) if applicable > 0 else 1.0


def score_overall(completeness: float, validity: float) -> float:
    """Weighted composite: completeness 60%, validity 40%."""
    return round(completeness * 0.6 + validity * 0.4, 3)


# ─────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────

def score_golden_record(record: GoldenRecord) -> GoldenRecord:
    """
    Compute and attach quality scores to a GoldenRecord in-place.

    Args:
        record: A GoldenRecord (from survivorship).

    Returns:
        The same GoldenRecord with quality_score, completeness_score,
        and validity_score populated.
    """
    c = score_completeness(record)
    v = score_validity(record)
    o = score_overall(c, v)

    record.completeness_score = c
    record.validity_score     = v
    record.quality_score      = o
    return record


def score_all_golden_records(records: list[GoldenRecord]) -> list[GoldenRecord]:
    """Score an entire list of GoldenRecords."""
    return [score_golden_record(r) for r in records]


def quality_summary(records: list[GoldenRecord]) -> dict:
    """Return aggregate quality statistics for the dashboard."""
    if not records:
        return {}
    scores = [r.quality_score or 0.0 for r in records]
    completeness = [r.completeness_score or 0.0 for r in records]
    validity = [r.validity_score or 0.0 for r in records]

    def band(score):
        if score >= 0.85: return "high"
        if score >= 0.65: return "medium"
        return "low"

    high   = sum(1 for s in scores if band(s) == "high")
    medium = sum(1 for s in scores if band(s) == "medium")
    low    = sum(1 for s in scores if band(s) == "low")

    return {
        "total_golden_records": len(records),
        "avg_quality":          round(sum(scores) / len(scores), 3),
        "avg_completeness":     round(sum(completeness) / len(completeness), 3),
        "avg_validity":         round(sum(validity) / len(validity), 3),
        "high_quality_count":   high,
        "medium_quality_count": medium,
        "low_quality_count":    low,
    }
