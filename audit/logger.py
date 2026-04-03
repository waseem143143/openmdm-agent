"""
openmdm-agent · audit/logger.py
Immutable audit trail for all AI Steward Agent decisions.
Persists to a local SQLite database — zero infrastructure required.

Every merge decision, golden record creation, and quality score event
is logged with full rationale, model version, timestamp, and scores.
This log is the compliance artifact for MDM governance reviews.
"""

from __future__ import annotations
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from mdm.models import MatchPair, GoldenRecord, AuditEvent


# ─────────────────────────────────────────────
# Database setup
# ─────────────────────────────────────────────

DEFAULT_DB_PATH = "openmdm_audit.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    event_id         TEXT PRIMARY KEY,
    event_type       TEXT NOT NULL,
    pair_id          TEXT,
    record_a_id      TEXT,
    record_b_id      TEXT,
    source_a         TEXT,
    source_b         TEXT,
    golden_id        TEXT,
    decision         TEXT,
    confidence_band  TEXT,
    overall_score    REAL,
    rationale        TEXT,
    agent_model      TEXT,
    timestamp        TEXT NOT NULL
);
"""


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn


# ─────────────────────────────────────────────
# Log writers
# ─────────────────────────────────────────────

def log_match_decision(
    pair: MatchPair,
    db_path: str = DEFAULT_DB_PATH,
) -> AuditEvent:
    """
    Log an AI Steward Agent merge decision to the audit trail.

    Args:
        pair:    Decided MatchPair with decision and rationale populated.
        db_path: Path to the SQLite database.

    Returns:
        The AuditEvent that was written.
    """
    event = AuditEvent(
        event_id        = str(uuid.uuid4()),
        event_type      = "MATCH_DECISION",
        pair_id         = pair.pair_id,
        golden_id       = None,
        decision        = pair.decision.value if pair.decision else None,
        rationale       = pair.decision_rationale,
        agent_model     = pair.decided_by or "AI_STEWARD",
        overall_score   = pair.overall_score,
        confidence_band = pair.confidence_band,
        timestamp       = pair.decided_at or datetime.utcnow().isoformat(),
    )

    conn = get_connection(db_path)
    conn.execute("""
        INSERT OR REPLACE INTO audit_log
        (event_id, event_type, pair_id, record_a_id, record_b_id,
         source_a, source_b, golden_id, decision, confidence_band,
         overall_score, rationale, agent_model, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        event.event_id,
        event.event_type,
        pair.pair_id,
        pair.record_a.record_id,
        pair.record_b.record_id,
        pair.record_a.source_system,
        pair.record_b.source_system,
        None,
        event.decision,
        event.confidence_band,
        event.overall_score,
        event.rationale,
        event.agent_model,
        event.timestamp,
    ))
    conn.commit()
    conn.close()
    return event


def log_golden_created(
    golden: GoldenRecord,
    db_path: str = DEFAULT_DB_PATH,
) -> AuditEvent:
    """Log a GoldenRecord creation event."""
    event = AuditEvent(
        event_id    = str(uuid.uuid4()),
        event_type  = "GOLDEN_CREATED",
        golden_id   = golden.golden_id,
        decision    = "CREATED",
        rationale   = (f"Golden record created from sources: "
                       f"{', '.join(golden.source_systems)}. "
                       f"Merged {len(golden.source_record_ids)} records."),
        agent_model = "SURVIVORSHIP_ENGINE",
        timestamp   = golden.created_at,
    )

    conn = get_connection(db_path)
    conn.execute("""
        INSERT OR REPLACE INTO audit_log
        (event_id, event_type, pair_id, record_a_id, record_b_id,
         source_a, source_b, golden_id, decision, confidence_band,
         overall_score, rationale, agent_model, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        event.event_id, event.event_type,
        None, None, None, None, None,
        golden.golden_id, event.decision,
        None, golden.quality_score,
        event.rationale, event.agent_model, event.timestamp,
    ))
    conn.commit()
    conn.close()
    return event


def log_all_decisions(
    pairs: list[MatchPair],
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Bulk-log all match decisions. Returns count of events written."""
    count = 0
    for pair in pairs:
        if pair.decision and pair.decision.value != "PENDING":
            log_match_decision(pair, db_path)
            count += 1
    return count


# ─────────────────────────────────────────────
# Audit reader (for the UI)
# ─────────────────────────────────────────────

def read_audit_log(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Load the full audit log as a Pandas DataFrame for display.

    Returns:
        DataFrame with all audit events, most recent first.
    """
    if not Path(db_path).exists():
        return pd.DataFrame()

    conn = get_connection(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM audit_log ORDER BY timestamp DESC",
        conn
    )
    conn.close()
    return df


def clear_audit_log(db_path: str = DEFAULT_DB_PATH) -> None:
    """Wipe the audit log (for demo resets only — never do this in production)."""
    if Path(db_path).exists():
        conn = get_connection(db_path)
        conn.execute("DELETE FROM audit_log")
        conn.commit()
        conn.close()
