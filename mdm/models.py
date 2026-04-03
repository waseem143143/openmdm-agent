"""
openmdm-agent · mdm/models.py
Core Pydantic models for Patient MDM entities.
"""

from __future__ import annotations
from datetime import date, datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import re


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class Gender(str, Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


class MatchDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    PENDING = "PENDING"


class SourceSystem(str, Enum):
    EHR = "EHR"
    CLAIMS = "CLAIMS"
    LAB = "LAB"
    PHARMACY = "PHARMACY"
    REGISTRY = "REGISTRY"


class QualityDimension(str, Enum):
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    CONFORMITY = "conformity"


# ─────────────────────────────────────────────
# Core Patient Record (raw source record)
# ─────────────────────────────────────────────

class PatientRecord(BaseModel):
    """
    A raw patient record ingested from a source system.
    Represents one row from EHR, Claims, Lab, etc.
    """
    record_id: str = Field(..., description="Unique ID within source system")
    source_system: str = Field(..., description="Originating system (EHR, CLAIMS, etc.)")
    source_priority: int = Field(default=5, ge=1, le=10,
                                  description="Trust rank: 1=highest, 10=lowest")

    # Identity fields
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    date_of_birth: Optional[str] = None          # stored as ISO string for flexibility
    gender: Optional[str] = None
    ssn_last4: Optional[str] = None

    # Contact fields
    address_line1: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

    # Clinical identifiers
    mrn: Optional[str] = None                    # Medical Record Number
    insurance_id: Optional[str] = None

    # Metadata
    record_created: Optional[str] = None
    record_updated: Optional[str] = None

    # Computed at load time
    normalized_first: Optional[str] = None
    normalized_last: Optional[str] = None
    normalized_phone: Optional[str] = None
    normalized_zip: Optional[str] = None

    class Config:
        use_enum_values = True


# ─────────────────────────────────────────────
# Match Pair (candidate duplicate pair)
# ─────────────────────────────────────────────

class FieldScore(BaseModel):
    """Similarity score for a single field comparison."""
    field_name: str
    value_a: Optional[str]
    value_b: Optional[str]
    score: float = Field(ge=0.0, le=1.0)
    method: str = Field(description="jaro_winkler | exact | phonetic | partial")


class MatchPair(BaseModel):
    """
    A candidate duplicate pair surfaced by the matcher.
    Fed into the AI Steward Agent for a merge/reject decision.
    """
    pair_id: str
    record_a: PatientRecord
    record_b: PatientRecord

    # Scoring
    overall_score: float = Field(ge=0.0, le=1.0)
    field_scores: list[FieldScore] = Field(default_factory=list)
    confidence_band: str = Field(default="LOW",
                                  description="HIGH | MEDIUM | LOW")

    # Rule attribution — which rule fired and what band it produced
    match_rule_id:   Optional[str] = None   # e.g. "RULE_01"
    match_rule_name: Optional[str] = None   # e.g. "Deterministic Identity Match"
    decision_band:   Optional[str] = None   # "AUTO_APPROVE" | "AUTO_REJECT" | "LLM_REVIEW"

    # Agent decision (populated after agent runs)
    decision: MatchDecision = Field(default=MatchDecision.PENDING)
    decision_rationale: Optional[str] = None
    decided_at: Optional[str] = None
    decided_by: str = Field(default="AI_STEWARD")

    def key_conflicts(self) -> list[FieldScore]:
        """Return fields where records disagree (score < 0.8)."""
        return [fs for fs in self.field_scores if fs.score < 0.8]

    def key_agreements(self) -> list[FieldScore]:
        """Return fields where records strongly agree (score >= 0.9)."""
        return [fs for fs in self.field_scores if fs.score >= 0.9]


# ─────────────────────────────────────────────
# Golden Record (survivor after merge)
# ─────────────────────────────────────────────

class AttributeProvenance(BaseModel):
    """Tracks which source system won survivorship for a given attribute."""
    attribute: str
    winning_value: Optional[str]
    winning_source: str
    winning_record_id: str
    rule_applied: str = Field(description="trust | recency | completeness")


class GoldenRecord(BaseModel):
    """
    The master patient identity — the single source of truth
    constructed from one or more merged source records.
    """
    golden_id: str
    source_record_ids: list[str] = Field(default_factory=list)
    source_systems: list[str] = Field(default_factory=list)

    # Survived attributes
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    ssn_last4: Optional[str] = None
    address_line1: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    mrn: Optional[str] = None
    insurance_id: Optional[str] = None

    # Provenance
    attribute_provenance: list[AttributeProvenance] = Field(default_factory=list)

    # Quality
    quality_score: Optional[float] = None
    completeness_score: Optional[float] = None
    validity_score: Optional[float] = None

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.middle_name, self.last_name]
        return " ".join(p for p in parts if p)


# ─────────────────────────────────────────────
# Audit Event
# ─────────────────────────────────────────────

class AuditEvent(BaseModel):
    """Every agent decision is recorded as an immutable audit event."""
    event_id: str
    event_type: str = Field(description="MATCH_DECISION | GOLDEN_CREATED | QUALITY_SCORED")
    pair_id: Optional[str] = None
    golden_id: Optional[str] = None
    decision: Optional[str] = None
    rationale: Optional[str] = None
    agent_model: str = Field(default="claude-haiku-4-5-20251001")
    overall_score: Optional[float] = None
    confidence_band: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
