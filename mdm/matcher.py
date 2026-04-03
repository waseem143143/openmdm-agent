"""
openmdm-agent · mdm/matcher.py
Candidate pair discovery using the configurable Match Rule Engine.

Replaces hardcoded Jaro-Winkler weights with a pluggable rule system
that mirrors enterprise MDM platforms like Reltio and Informatica.

Pipeline:
  1. Blocking  — Soundex(last_name) + DOB → candidate buckets
  2. Rule Engine — evaluate all 3 match rules per pair in priority order
  3. Threshold  — discard pairs below MIN_SCORE_THRESHOLD
"""

from __future__ import annotations
import uuid
import itertools
from jellyfish import soundex
from mdm.models import PatientRecord, MatchPair, MatchDecision
from mdm.match_rules import (
    MatchRule, RuleEngineResult,
    run_rule_engine, default_rules,
)

MIN_SCORE_THRESHOLD = 0.55


# ─────────────────────────────────────────────
# Blocking
# ─────────────────────────────────────────────

def blocking_key(record: PatientRecord) -> str:
    """
    Soundex(last_name) + "|" + date_of_birth
    Records in the same bucket are candidate pairs.
    """
    last = record.normalized_last or record.last_name or ""
    dob  = record.date_of_birth or ""
    sdx  = soundex(last) if last else "Z000"
    return f"{sdx}|{dob}"


# ─────────────────────────────────────────────
# Confidence banding (for display)
# ─────────────────────────────────────────────

def assign_confidence(score: float) -> str:
    if score >= 0.85: return "HIGH"
    if score >= 0.70: return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────
# Main matcher
# ─────────────────────────────────────────────

def find_candidate_pairs(
    records: list[PatientRecord],
    rules:   list[MatchRule] | None = None,
) -> list[MatchPair]:
    """
    Full matching pipeline using the configurable Match Rule Engine.

    Args:
        records: Normalised PatientRecord list from the loader.
        rules:   List of MatchRule objects. Defaults to the 3 standard rules.

    Returns:
        List of MatchPair objects sorted by overall_score descending.
    """
    if rules is None:
        rules = default_rules()

    # Step 1: Build blocking buckets
    buckets: dict[str, list[PatientRecord]] = {}
    for record in records:
        key = blocking_key(record)
        buckets.setdefault(key, []).append(record)

    # Step 2: Evaluate rule engine on every within-bucket pair
    seen_pairs: set[frozenset] = set()
    pairs: list[MatchPair] = []

    for bucket_records in buckets.values():
        if len(bucket_records) < 2:
            continue

        for rec_a, rec_b in itertools.combinations(bucket_records, 2):
            pair_key = frozenset({rec_a.record_id, rec_b.record_id})
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Run all match rules
            engine_result: RuleEngineResult = run_rule_engine(rec_a, rec_b, rules)

            # Discard below threshold
            if engine_result.overall_score < MIN_SCORE_THRESHOLD:
                continue

            # Map decision band to confidence
            confidence = assign_confidence(engine_result.overall_score)

            pair = MatchPair(
                pair_id          = str(uuid.uuid4())[:8],
                record_a         = rec_a,
                record_b         = rec_b,
                overall_score    = engine_result.overall_score,
                field_scores     = engine_result.field_scores,
                confidence_band  = confidence,
                match_rule_id    = engine_result.winning_rule_id,
                match_rule_name  = engine_result.winning_rule_name,
                decision_band    = engine_result.decision_band,
            )

            # Auto-decisions skip the LLM
            if engine_result.decision_band == "AUTO_APPROVE":
                pair.decision           = MatchDecision.APPROVE
                pair.decision_rationale = (
                    f"✅ AUTO-APPROVED by Rule: {engine_result.winning_rule_name}\n"
                    f"Score: {engine_result.overall_score:.3f} ≥ auto-approve threshold.\n"
                    f"Both anchor fields matched exactly — deterministic confirmation."
                )
                pair.decided_by = f"RULE_ENGINE/{engine_result.winning_rule_id}"

            elif engine_result.decision_band == "AUTO_REJECT":
                pair.decision           = MatchDecision.REJECT
                pair.decision_rationale = (
                    f"❌ AUTO-REJECTED by Rule: {engine_result.winning_rule_name}\n"
                    f"Score: {engine_result.overall_score:.3f} < auto-reject threshold.\n"
                    f"Insufficient evidence to support a merge."
                )
                pair.decided_by = f"RULE_ENGINE/{engine_result.winning_rule_id}"

            # LLM_REVIEW → stays PENDING, picked up by steward agent
            pairs.append(pair)

    pairs.sort(key=lambda p: p.overall_score, reverse=True)
    return pairs


# ─────────────────────────────────────────────
# Summary stats
# ─────────────────────────────────────────────

def pairs_summary(pairs: list[MatchPair]) -> dict:
    from mdm.models import MatchDecision
    high   = sum(1 for p in pairs if p.confidence_band == "HIGH")
    medium = sum(1 for p in pairs if p.confidence_band == "MEDIUM")
    low    = sum(1 for p in pairs if p.confidence_band == "LOW")
    auto_approved = sum(1 for p in pairs if p.decision == MatchDecision.APPROVE
                        and p.decided_by and "RULE_ENGINE" in (p.decided_by or ""))
    auto_rejected = sum(1 for p in pairs if p.decision == MatchDecision.REJECT
                        and p.decided_by and "RULE_ENGINE" in (p.decided_by or ""))
    return {
        "total_pairs":        len(pairs),
        "high_confidence":    high,
        "medium_confidence":  medium,
        "low_confidence":     low,
        "auto_approved":      auto_approved,
        "auto_rejected":      auto_rejected,
        "llm_needed":         len(pairs) - auto_approved - auto_rejected,
        "avg_score":          round(
            sum(p.overall_score for p in pairs) / len(pairs), 3
        ) if pairs else 0.0,
    }
