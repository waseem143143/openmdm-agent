"""
openmdm-agent · mdm/crosswalk.py

Crosswalk (Cross-Reference / XREF) table builder.

For every golden record, tracks every source record that was merged
into it — with full lineage: which system it came from, when it was
last updated, which match rule fired, and what the merge decision was.

This is the standard MDM concept used in Reltio (crosswalk),
Informatica (cross-reference), and IBM MDM (xref) — built open-source.
"""

from __future__ import annotations
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from mdm.models import GoldenRecord, MatchPair, PatientRecord, MatchDecision


# ─────────────────────────────────────────────
# Crosswalk entry — one row per source record
# ─────────────────────────────────────────────

@dataclass
class CrosswalkEntry:
    """
    A single crosswalk entry linking one source record to its golden record.
    One GoldenRecord → many CrosswalkEntries (one per merged source record).
    """
    # Golden record reference
    golden_id:          str
    golden_full_name:   str

    # Source record reference
    source_record_id:   str
    source_system:      str
    source_priority:    int

    # Identity snapshot at time of merge
    first_name:         Optional[str]
    last_name:          Optional[str]
    date_of_birth:      Optional[str]
    phone:              Optional[str]
    email:              Optional[str]
    mrn:                Optional[str]
    insurance_id:       Optional[str]
    address_line1:      Optional[str]
    city:               Optional[str]
    state:              Optional[str]

    # Merge lineage
    merge_type:         str    # "MATCHED_MERGE" | "SINGLETON" | "AUTO_APPROVED" | "AI_APPROVED"
    match_rule:         Optional[str]
    match_score:        Optional[float]
    match_decision:     Optional[str]
    record_updated:     Optional[str]
    crosswalk_created:  str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ─────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────

def build_crosswalk(
    golden_records: list[GoldenRecord],
    decided_pairs:  list[MatchPair],
    all_records:    list[PatientRecord],
) -> list[CrosswalkEntry]:
    """
    Build the full crosswalk table from golden records, decided pairs,
    and the original source records.

    Args:
        golden_records: Output of survivorship — the master golden records.
        decided_pairs:  All MatchPairs with decisions (APPROVE/REJECT/etc).
        all_records:    Full list of original source PatientRecord objects.

    Returns:
        List of CrosswalkEntry — one row per source record per golden record.
    """
    entries: list[CrosswalkEntry] = []

    # Build a lookup: record_id → PatientRecord
    record_map: dict[str, PatientRecord] = {r.record_id: r for r in all_records}

    # Build a lookup: record_id → (pair, decision) for merged records
    merge_info: dict[str, tuple[MatchPair, str]] = {}
    for pair in decided_pairs:
        if pair.decision in (MatchDecision.APPROVE,):
            rule_name = "Auto-Approved (Rule Engine)"
            if pair.decided_by and "AI_STEWARD" in (pair.decided_by or ""):
                rule_name = "AI Steward Agent"
            elif pair.decided_by and "RULE_ENGINE" in (pair.decided_by or ""):
                # Extract rule name from rationale
                rationale = pair.decision_rationale or ""
                if "Deterministic" in rationale:
                    rule_name = "Rule 1 — Deterministic Identity Match"
                elif "High Confidence" in rationale:
                    rule_name = "Rule 2 — High Confidence Name+DOB+Phone"
                elif "Full Probabilistic" in rationale:
                    rule_name = "Rule 3 — Full Probabilistic (7-Field)"

            merge_info[pair.record_a.record_id] = (pair, rule_name)
            merge_info[pair.record_b.record_id] = (pair, rule_name)

    # Build crosswalk entries for each golden record
    for gr in golden_records:
        golden_name = f"{gr.first_name or ''} {gr.last_name or ''}".strip()
        is_merged   = len(gr.source_record_ids) > 1

        for rec_id in gr.source_record_ids:
            source_rec = record_map.get(rec_id)
            if not source_rec:
                continue

            # Determine merge type
            if not is_merged:
                merge_type    = "SINGLETON"
                match_rule    = None
                match_score   = None
                match_decision= None
            elif rec_id in merge_info:
                pair, rule_name = merge_info[rec_id]
                if pair.decided_by and "AI_STEWARD" in (pair.decided_by or ""):
                    merge_type = "AI_APPROVED"
                else:
                    merge_type = "AUTO_APPROVED"
                match_rule     = rule_name
                match_score    = pair.overall_score
                match_decision = pair.decision.value if pair.decision else None
            else:
                merge_type    = "MATCHED_MERGE"
                match_rule    = None
                match_score   = None
                match_decision= "APPROVE"

            entry = CrosswalkEntry(
                golden_id        = gr.golden_id,
                golden_full_name = golden_name,
                source_record_id = rec_id,
                source_system    = source_rec.source_system,
                source_priority  = source_rec.source_priority,
                first_name       = source_rec.first_name,
                last_name        = source_rec.last_name,
                date_of_birth    = source_rec.date_of_birth,
                phone            = source_rec.phone,
                email            = source_rec.email,
                mrn              = source_rec.mrn,
                insurance_id     = source_rec.insurance_id,
                address_line1    = source_rec.address_line1,
                city             = source_rec.city,
                state            = source_rec.state,
                merge_type       = merge_type,
                match_rule       = match_rule,
                match_score      = match_score,
                match_decision   = match_decision,
                record_updated   = source_rec.record_updated,
            )
            entries.append(entry)

    return entries


# ─────────────────────────────────────────────
# Exporters
# ─────────────────────────────────────────────

def crosswalk_to_dataframe(entries: list[CrosswalkEntry]) -> pd.DataFrame:
    """Convert crosswalk entries to a flat DataFrame for display and export."""
    rows = []
    for e in entries:
        rows.append({
            "Golden ID":        e.golden_id,
            "Golden Name":      e.golden_full_name,
            "Source Record ID": e.source_record_id,
            "Source System":    e.source_system,
            "Source Priority":  e.source_priority,
            "First Name":       e.first_name or "—",
            "Last Name":        e.last_name or "—",
            "DOB":              e.date_of_birth or "—",
            "Phone":            e.phone or "—",
            "Email":            e.email or "—",
            "MRN":              e.mrn or "—",
            "Insurance ID":     e.insurance_id or "—",
            "Address":          e.address_line1 or "—",
            "City":             e.city or "—",
            "State":            e.state or "—",
            "Merge Type":       e.merge_type,
            "Match Rule":       e.match_rule or "—",
            "Match Score":      e.match_score if e.match_score else "—",
            "Decision":         e.match_decision or "—",
            "Record Updated":   e.record_updated or "—",
        })
    return pd.DataFrame(rows)


def crosswalk_summary(entries: list[CrosswalkEntry]) -> dict:
    """Summary statistics for the crosswalk dashboard."""
    total         = len(entries)
    merged        = sum(1 for e in entries if e.merge_type != "SINGLETON")
    singletons    = sum(1 for e in entries if e.merge_type == "SINGLETON")
    auto_approved = sum(1 for e in entries if e.merge_type == "AUTO_APPROVED")
    ai_approved   = sum(1 for e in entries if e.merge_type == "AI_APPROVED")

    # Unique golden records
    golden_ids    = set(e.golden_id for e in entries)
    merged_grs    = set(
        e.golden_id for e in entries if e.merge_type != "SINGLETON"
    )

    # Source system breakdown
    systems = {}
    for e in entries:
        systems[e.source_system] = systems.get(e.source_system, 0) + 1

    return {
        "total_source_records":  total,
        "total_golden_records":  len(golden_ids),
        "merged_records":        merged,
        "singleton_records":     singletons,
        "auto_approved":         auto_approved,
        "ai_approved":           ai_approved,
        "consolidation_rate":    round(merged / total * 100, 1) if total else 0.0,
        "source_systems":        systems,
    }
