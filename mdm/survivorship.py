"""
openmdm-agent · mdm/survivorship.py
Constructs Golden Records from approved merges.

Survivorship rules (in priority order):
  1. Trust rank  — lower source_priority number wins (EHR=1 beats CLAIMS=3)
  2. Recency     — most recently updated record wins ties
  3. Completeness — non-null value beats null value
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional
from mdm.models import (PatientRecord, MatchPair, MatchDecision,
                          GoldenRecord, AttributeProvenance)


# Fields that participate in survivorship
SURVIVED_FIELDS = [
    "first_name", "last_name", "middle_name", "date_of_birth",
    "gender", "ssn_last4", "address_line1", "city", "state",
    "zip_code", "phone", "email", "mrn", "insurance_id",
]


# ─────────────────────────────────────────────
# Survivorship helpers
# ─────────────────────────────────────────────

def _recency_rank(record: PatientRecord) -> str:
    """Return record_updated as ISO string for comparison; empty = oldest."""
    return record.record_updated or "0000-00-00"


def _survive_attribute(
    field: str,
    rec_a: PatientRecord,
    rec_b: PatientRecord,
) -> AttributeProvenance:
    """
    Pick the winning value for a single attribute.

    Rules applied in order:
      1. Completeness — prefer non-null over null
      2. Trust rank   — lower source_priority wins
      3. Recency      — more recently updated record wins
    """
    val_a = getattr(rec_a, field, None)
    val_b = getattr(rec_b, field, None)

    # Rule 1 — Completeness
    if val_a and not val_b:
        return AttributeProvenance(attribute=field, winning_value=val_a,
                                   winning_source=rec_a.source_system,
                                   winning_record_id=rec_a.record_id,
                                   rule_applied="completeness")
    if val_b and not val_a:
        return AttributeProvenance(attribute=field, winning_value=val_b,
                                   winning_source=rec_b.source_system,
                                   winning_record_id=rec_b.record_id,
                                   rule_applied="completeness")

    # Rule 2 — Trust rank (lower number = higher trust)
    if rec_a.source_priority < rec_b.source_priority:
        return AttributeProvenance(attribute=field, winning_value=val_a,
                                   winning_source=rec_a.source_system,
                                   winning_record_id=rec_a.record_id,
                                   rule_applied="trust")
    if rec_b.source_priority < rec_a.source_priority:
        return AttributeProvenance(attribute=field, winning_value=val_b,
                                   winning_source=rec_b.source_system,
                                   winning_record_id=rec_b.record_id,
                                   rule_applied="trust")

    # Rule 3 — Recency
    if _recency_rank(rec_a) >= _recency_rank(rec_b):
        return AttributeProvenance(attribute=field, winning_value=val_a,
                                   winning_source=rec_a.source_system,
                                   winning_record_id=rec_a.record_id,
                                   rule_applied="recency")
    return AttributeProvenance(attribute=field, winning_value=val_b,
                               winning_source=rec_b.source_system,
                               winning_record_id=rec_b.record_id,
                               rule_applied="recency")


# ─────────────────────────────────────────────
# Golden record builder
# ─────────────────────────────────────────────

def build_golden_record(pair: MatchPair) -> GoldenRecord:
    """
    Build a GoldenRecord from an APPROVED MatchPair.
    Each attribute carries provenance showing which source won and why.

    Args:
        pair: A MatchPair where decision == APPROVE.

    Returns:
        A fully attributed GoldenRecord.
    """
    rec_a = pair.record_a
    rec_b = pair.record_b

    provenance: list[AttributeProvenance] = []
    golden_attrs: dict = {}

    for field in SURVIVED_FIELDS:
        prov = _survive_attribute(field, rec_a, rec_b)
        provenance.append(prov)
        golden_attrs[field] = prov.winning_value

    golden = GoldenRecord(
        golden_id          = f"GR-{str(uuid.uuid4())[:8].upper()}",
        source_record_ids  = [rec_a.record_id, rec_b.record_id],
        source_systems     = list({rec_a.source_system, rec_b.source_system}),
        attribute_provenance = provenance,
        created_at         = datetime.utcnow().isoformat(),
        **golden_attrs,
    )
    return golden


def build_singleton_golden(record: PatientRecord) -> GoldenRecord:
    """
    Build a GoldenRecord from a single record with no matches (no merge needed).
    All attributes are sourced from this record, rule = 'sole_source'.
    """
    provenance = [
        AttributeProvenance(
            attribute        = field,
            winning_value    = getattr(record, field, None),
            winning_source   = record.source_system,
            winning_record_id= record.record_id,
            rule_applied     = "sole_source",
        )
        for field in SURVIVED_FIELDS
    ]
    golden_attrs = {field: getattr(record, field, None) for field in SURVIVED_FIELDS}
    return GoldenRecord(
        golden_id            = f"GR-{str(uuid.uuid4())[:8].upper()}",
        source_record_ids    = [record.record_id],
        source_systems       = [record.source_system],
        attribute_provenance = provenance,
        created_at           = datetime.utcnow().isoformat(),
        **golden_attrs,
    )


# ─────────────────────────────────────────────
# Full survivorship pipeline
# ─────────────────────────────────────────────

def build_all_golden_records(
    pairs: list[MatchPair],
    all_records: list[PatientRecord],
) -> list[GoldenRecord]:
    """
    Run survivorship across all pairs and unmatched records.

    Args:
        pairs:       All decided MatchPairs (APPROVE / REJECT / HUMAN_REVIEW).
        all_records: Full list of source records.

    Returns:
        List of GoldenRecord objects — one per unique patient identity.
    """
    golden_records: list[GoldenRecord] = []
    merged_record_ids: set[str] = set()

    # Process APPROVED pairs → build merged golden records
    for pair in pairs:
        if pair.decision == MatchDecision.APPROVE:
            gr = build_golden_record(pair)
            golden_records.append(gr)
            merged_record_ids.add(pair.record_a.record_id)
            merged_record_ids.add(pair.record_b.record_id)

    # Unmatched records → singleton golden records
    for record in all_records:
        if record.record_id not in merged_record_ids:
            gr = build_singleton_golden(record)
            golden_records.append(gr)

    return golden_records
