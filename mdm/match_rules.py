"""
openmdm-agent · mdm/match_rules.py

Configurable Match Rule Engine — mirrors enterprise MDM platforms
like Reltio and Informatica MDM.

Each MatchRule defines:
  - Which fields to evaluate
  - Weight per field
  - Algorithm per field (exact | jaro_winkler | phonetic)
  - Auto-approve threshold  (above = APPROVE, no LLM needed)
  - Auto-reject threshold   (below = REJECT, no LLM needed)
  - LLM band                (between = send to AI Steward)

Rules are evaluated in priority order. First rule that produces a
confident decision wins. If no rule fires confidently, the pair
falls through to the LLM.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from jellyfish import jaro_winkler_similarity, soundex
from mdm.models import PatientRecord, MatchPair, FieldScore


# ─────────────────────────────────────────────
# Field configuration within a rule
# ─────────────────────────────────────────────

@dataclass
class FieldConfig:
    """One field's contribution to a match rule."""
    field_name:  str
    weight:      float          # 0.0 – 1.0  (all weights in a rule must sum to 1.0)
    algorithm:   str            # "exact" | "jaro_winkler" | "normalized_exact"
    required:    bool = False   # if True and field is missing → rule cannot fire


# ─────────────────────────────────────────────
# Match Rule definition
# ─────────────────────────────────────────────

@dataclass
class MatchRule:
    """
    A single named match rule with configurable fields, weights, and thresholds.

    Mirrors the concept of Match Rules in Reltio, Informatica MDM,
    and other enterprise MDM platforms.
    """
    rule_id:               str
    rule_name:             str
    description:           str
    fields:                list[FieldConfig]
    auto_approve_threshold: float  = 0.90   # above → auto APPROVE (no LLM)
    auto_reject_threshold:  float  = 0.60   # below → auto REJECT (no LLM)
    enabled:               bool   = True
    priority:              int    = 1       # lower = evaluated first

    def total_weight(self) -> float:
        return sum(f.weight for f in self.fields)

    def field_names(self) -> list[str]:
        return [f.field_name for f in self.fields]


# ─────────────────────────────────────────────
# Default 3 rules (matching enterprise MDM logic)
# ─────────────────────────────────────────────

def default_rules() -> list[MatchRule]:
    """
    Three pre-configured match rules that mirror enterprise MDM platforms.

    Rule 1 — DETERMINISTIC:
        SSN-last4 + DOB exact match.
        Maps to Reltio's Deterministic Match / Informatica's Exact Match Rule.
        Auto-approves immediately — no LLM call, zero cost.

    Rule 2 — HIGH CONFIDENCE PROBABILISTIC:
        Name + DOB + Phone — high-precision focused rule.
        Maps to Reltio's "Name + DOB + Phone" probabilistic rule.
        Catches EHR↔CLAIMS duplicates efficiently.

    Rule 3 — FULL PROBABILISTIC:
        All 7 identity fields with standard weights.
        Broadest net — catches edge cases, feeds LLM for ambiguous pairs.
        Maps to Reltio's full probabilistic match rule.
    """
    return [
        MatchRule(
            rule_id    = "RULE_01",
            rule_name  = "Deterministic Identity Match",
            description= (
                "Exact match on SSN-last4 AND Date of Birth. "
                "When both anchor fields match exactly, records are definitively "
                "the same patient — no AI review needed. "
                "Equivalent to Reltio Deterministic / Informatica Exact Match."
            ),
            fields=[
                FieldConfig("ssn_last4",    weight=0.50, algorithm="exact",   required=True),
                FieldConfig("date_of_birth",weight=0.50, algorithm="exact",   required=True),
            ],
            auto_approve_threshold=0.98,  # both fields must match exactly
            auto_reject_threshold =0.49,  # only rejects if both fields clearly mismatch
            priority=1,
        ),

        MatchRule(
            rule_id    = "RULE_02",
            rule_name  = "High Confidence Name + DOB + Phone",
            description= (
                "Probabilistic match on last name, first name, date of birth, "
                "and phone number. Optimised for EHR ↔ CLAIMS matching where "
                "SSN may be absent. High weights on DOB (hard anchor) and phone "
                "(stable identifier). "
                "Equivalent to Reltio's Name+DOB+Phone probabilistic rule."
            ),
            fields=[
                FieldConfig("last_name",    weight=0.30, algorithm="jaro_winkler"),
                FieldConfig("first_name",   weight=0.20, algorithm="jaro_winkler"),
                FieldConfig("date_of_birth",weight=0.30, algorithm="exact"),
                FieldConfig("phone",        weight=0.20, algorithm="normalized_exact"),
            ],
            auto_approve_threshold=0.98,
            auto_reject_threshold =0.70,  # probabilistic — wider LLM band than Rule 1
            priority=2,
        ),

        MatchRule(
            rule_id    = "RULE_03",
            rule_name  = "Full Probabilistic (7-Field)",
            description= (
                "Comprehensive probabilistic match across all 7 identity fields: "
                "name, DOB, phone, SSN, address, and email — each with "
                "clinically motivated weights. Catches edge cases missed by "
                "Rules 1 and 2. Ambiguous pairs (medium confidence) are "
                "escalated to the AI Steward Agent for final decision. "
                "Equivalent to Reltio's full probabilistic match rule."
            ),
            fields=[
                FieldConfig("last_name",    weight=0.25, algorithm="jaro_winkler"),
                FieldConfig("first_name",   weight=0.20, algorithm="jaro_winkler"),
                FieldConfig("date_of_birth",weight=0.20, algorithm="exact"),
                FieldConfig("phone",        weight=0.15, algorithm="normalized_exact"),
                FieldConfig("ssn_last4",    weight=0.10, algorithm="exact"),
                FieldConfig("address_line1",weight=0.07, algorithm="jaro_winkler"),
                FieldConfig("email",        weight=0.03, algorithm="normalized_exact"),
            ],
            auto_approve_threshold=0.98,
            auto_reject_threshold =0.65,  # probabilistic — allows LLM review without SSN
            priority=3,
        ),
    ]


# ─────────────────────────────────────────────
# Field-level scorers
# ─────────────────────────────────────────────

def _get_field_value(record: PatientRecord, field_name: str) -> Optional[str]:
    """Retrieve a field value, using normalised version where available."""
    normalised_map = {
        "first_name":    "normalized_first",
        "last_name":     "normalized_last",
        "phone":         "normalized_phone",
        "address_line1": "address_line1",   # normalised at load time
    }
    norm_field = normalised_map.get(field_name)
    if norm_field and norm_field != field_name:
        val = getattr(record, norm_field, None)
        if val:
            return val
    return getattr(record, field_name, None)


def _score_field(
    field_cfg: FieldConfig,
    rec_a: PatientRecord,
    rec_b: PatientRecord,
) -> FieldScore:
    """Score a single field between two records using the rule's algorithm."""
    val_a = _get_field_value(rec_a, field_cfg.field_name)
    val_b = _get_field_value(rec_b, field_cfg.field_name)

    if not val_a or not val_b:
        return FieldScore(
            field_name=field_cfg.field_name,
            value_a=val_a, value_b=val_b,
            score=0.0, method="missing",
        )

    if field_cfg.algorithm == "exact":
        score = 1.0 if val_a.strip().upper() == val_b.strip().upper() else 0.0
        method = "exact"

    elif field_cfg.algorithm == "jaro_winkler":
        score  = round(jaro_winkler_similarity(val_a.upper(), val_b.upper()), 4)
        method = "jaro_winkler"

    elif field_cfg.algorithm == "normalized_exact":
        # strip formatting, compare digits/lowercase
        clean_a = ''.join(c for c in val_a if c.isalnum()).lower()
        clean_b = ''.join(c for c in val_b if c.isalnum()).lower()
        score  = 1.0 if clean_a == clean_b and clean_a != "" else 0.0
        method = "normalized_exact"

    else:
        score  = 0.0
        method = "unknown"

    return FieldScore(
        field_name=field_cfg.field_name,
        value_a=getattr(rec_a, field_cfg.field_name, None),
        value_b=getattr(rec_b, field_cfg.field_name, None),
        score=score, method=method,
    )


# ─────────────────────────────────────────────
# Rule evaluation result
# ─────────────────────────────────────────────

@dataclass
class RuleResult:
    """Result of evaluating one MatchRule against a candidate pair."""
    rule_id:         str
    rule_name:       str
    fired:           bool          # True if rule was applicable (required fields present)
    overall_score:   float
    field_scores:    list[FieldScore]
    band:            str           # "AUTO_APPROVE" | "LLM_REVIEW" | "AUTO_REJECT" | "NO_FIRE"
    confidence:      str           # "HIGH" | "MEDIUM" | "LOW"


def evaluate_rule(
    rule: MatchRule,
    rec_a: PatientRecord,
    rec_b: PatientRecord,
) -> RuleResult:
    """
    Evaluate a single MatchRule against a pair of records.

    Returns a RuleResult indicating whether the rule fired and what band
    the pair falls into.
    """
    field_scores: list[FieldScore] = []

    # Check required fields
    for fc in rule.fields:
        if fc.required:
            va = _get_field_value(rec_a, fc.field_name)
            vb = _get_field_value(rec_b, fc.field_name)
            if not va or not vb:
                return RuleResult(
                    rule_id=rule.rule_id, rule_name=rule.rule_name,
                    fired=False, overall_score=0.0,
                    field_scores=[], band="NO_FIRE", confidence="LOW",
                )

    # Score all fields
    total_weight = 0.0
    weighted_sum = 0.0
    for fc in rule.fields:
        fs = _score_field(fc, rec_a, rec_b)
        field_scores.append(fs)
        weighted_sum += fs.score * fc.weight
        total_weight += fc.weight

    overall = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

    # Determine band
    if overall >= rule.auto_approve_threshold:
        band       = "AUTO_APPROVE"
        confidence = "HIGH"
    elif overall < rule.auto_reject_threshold:
        band       = "AUTO_REJECT"
        confidence = "LOW"
    else:
        band       = "LLM_REVIEW"
        confidence = "MEDIUM"

    return RuleResult(
        rule_id=rule.rule_id, rule_name=rule.rule_name,
        fired=True, overall_score=overall,
        field_scores=field_scores, band=band, confidence=confidence,
    )


# ─────────────────────────────────────────────
# Match Rule Engine
# ─────────────────────────────────────────────

@dataclass
class RuleEngineResult:
    """Final result after running all rules against a pair."""
    winning_rule_id:   Optional[str]
    winning_rule_name: Optional[str]
    overall_score:     float
    confidence_band:   str        # "HIGH" | "MEDIUM" | "LOW"
    decision_band:     str        # "AUTO_APPROVE" | "AUTO_REJECT" | "LLM_REVIEW"
    field_scores:      list[FieldScore]
    rule_results:      list[RuleResult]   # all rule evaluations for audit


def run_rule_engine(
    rec_a:     PatientRecord,
    rec_b:     PatientRecord,
    rules:     list[MatchRule],
) -> RuleEngineResult:
    """
    Run all enabled match rules against a candidate pair in priority order.

    Logic:
      1. Evaluate rules in ascending priority order
      2. First rule that FIRES (required fields present) and produces
         AUTO_APPROVE or AUTO_REJECT → that rule wins
      3. If a rule fires but lands in LLM_REVIEW band → keep evaluating
         lower-priority rules (maybe a deterministic rule fires next)
      4. If no rule produces AUTO decision → use highest-scoring fired
         rule's result and send to LLM

    Returns:
        RuleEngineResult with the winning rule and final decision band.
    """
    sorted_rules = sorted(
        [r for r in rules if r.enabled],
        key=lambda r: r.priority
    )

    all_results: list[RuleResult] = []
    llm_candidates: list[RuleResult] = []

    for rule in sorted_rules:
        result = evaluate_rule(rule, rec_a, rec_b)
        all_results.append(result)

        if not result.fired:
            continue

        # AUTO_APPROVE always wins immediately — stop
        if result.band == "AUTO_APPROVE":
            return RuleEngineResult(
                winning_rule_id   = result.rule_id,
                winning_rule_name = result.rule_name,
                overall_score     = result.overall_score,
                confidence_band   = result.confidence,
                decision_band     = result.band,
                field_scores      = result.field_scores,
                rule_results      = all_results,
            )

        # AUTO_REJECT only wins if no LLM candidate has fired yet
        # (a higher-priority rule saying LLM_REVIEW beats a lower-priority AUTO_REJECT)
        if result.band == "AUTO_REJECT":
            if not llm_candidates:
                return RuleEngineResult(
                    winning_rule_id   = result.rule_id,
                    winning_rule_name = result.rule_name,
                    overall_score     = result.overall_score,
                    confidence_band   = result.confidence,
                    decision_band     = result.band,
                    field_scores      = result.field_scores,
                    rule_results      = all_results,
                )
            # else: a prior rule already said LLM_REVIEW — keep going

        # LLM_REVIEW — keep as candidate, continue to next rule
        if result.band == "LLM_REVIEW":
            llm_candidates.append(result)

    # No AUTO decision from any rule — use best LLM candidate
    if llm_candidates:
        best = max(llm_candidates, key=lambda r: r.overall_score)
        return RuleEngineResult(
            winning_rule_id   = best.rule_id,
            winning_rule_name = best.rule_name,
            overall_score     = best.overall_score,
            confidence_band   = "MEDIUM",
            decision_band     = "LLM_REVIEW",
            field_scores      = best.field_scores,
            rule_results      = all_results,
        )

    # Nothing fired at all (all required fields missing)
    return RuleEngineResult(
        winning_rule_id   = None,
        winning_rule_name = "No Rule Fired",
        overall_score     = 0.0,
        confidence_band   = "LOW",
        decision_band     = "AUTO_REJECT",
        field_scores      = [],
        rule_results      = all_results,
    )


# ─────────────────────────────────────────────
# Helper: serialise rules for UI display
# ─────────────────────────────────────────────

def rules_to_display(rules: list[MatchRule]) -> list[dict]:
    """Convert rules to a list of dicts for Streamlit display."""
    rows = []
    for r in rules:
        rows.append({
            "Priority":        r.priority,
            "Rule ID":         r.rule_id,
            "Rule Name":       r.rule_name,
            "Fields":          ", ".join(fc.field_name for fc in r.fields),
            "Auto-Approve ≥":  r.auto_approve_threshold,
            "Auto-Reject <":   r.auto_reject_threshold,
            "Enabled":         "✅" if r.enabled else "❌",
        })
    return rows
