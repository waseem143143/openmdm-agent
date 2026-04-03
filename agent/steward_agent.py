"""
openmdm-agent · agent/steward_agent.py
AI Steward Agent — routes decisions through the configured LLM backend.

Supports: Anthropic API | AWS Bedrock | Azure OpenAI | Ollama | Mock (Demo)
Backend is swappable with zero changes to matching, survivorship, or UI logic.
"""

from __future__ import annotations
import json
import time
import re
from datetime import datetime
from typing import Optional, Callable

from mdm.models import MatchPair, MatchDecision
from agent.prompts import SYSTEM_PROMPT, build_pair_prompt
from agent.backends import LLMBackend, create_backend, BACKEND_OPTIONS


# ─────────────────────────────────────────────
# Decision parser
# ─────────────────────────────────────────────

def _parse_decision(raw_text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()
    match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw_text[:200]}")
    return json.loads(match.group())


def _map_decision(decision_str: str) -> MatchDecision:
    return {
        "APPROVE":      MatchDecision.APPROVE,
        "REJECT":       MatchDecision.REJECT,
        "HUMAN_REVIEW": MatchDecision.HUMAN_REVIEW,
    }.get(decision_str.upper(), MatchDecision.HUMAN_REVIEW)


# ─────────────────────────────────────────────
# Single pair decision
# ─────────────────────────────────────────────

RETRY_LIMIT = 2
RETRY_DELAY = 2.0

def decide_pair(
    pair:        MatchPair,
    backend:     LLMBackend,
    on_progress: Optional[Callable[[str], None]] = None,
) -> MatchPair:
    """
    Run the AI Steward Agent on a single MatchPair using the given backend.
    Backend can be Mock, Anthropic, Bedrock, Azure, or Ollama.
    """
    prompt = build_pair_prompt(pair)

    if on_progress:
        on_progress(
            f"🤖 [{backend.display_name}] Analysing pair {pair.pair_id} "
            f"({pair.record_a.record_id} ↔ {pair.record_b.record_id}) ..."
        )

    last_error = None
    for attempt in range(RETRY_LIMIT + 1):
        try:
            raw_text = backend.decide(SYSTEM_PROMPT, prompt)
            parsed   = _parse_decision(raw_text)

            decision_str     = parsed.get("decision", "HUMAN_REVIEW")
            rationale        = parsed.get("rationale", "No rationale provided.")
            evidence_for     = parsed.get("key_evidence_for", [])
            evidence_against = parsed.get("key_evidence_against", [])

            full_rationale = rationale
            if evidence_for:
                full_rationale += "\n\n✅ Evidence FOR merge:\n" + "\n".join(
                    f"  • {e}" for e in evidence_for
                )
            if evidence_against:
                full_rationale += "\n\n❌ Evidence AGAINST:\n" + "\n".join(
                    f"  • {e}" for e in evidence_against
                )

            pair.decision           = _map_decision(decision_str)
            pair.decision_rationale = full_rationale
            pair.decided_at         = datetime.utcnow().isoformat()
            pair.decided_by         = f"AI_STEWARD/{backend.model_id}"

            if on_progress:
                icon = {"APPROVE": "✅", "REJECT": "❌",
                        "HUMAN_REVIEW": "🔍"}.get(decision_str.upper(), "❓")
                on_progress(
                    f"{icon} Pair {pair.pair_id}: {decision_str} — "
                    f"{rationale[:80]}..."
                )
            return pair

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
                continue

    pair.decision           = MatchDecision.HUMAN_REVIEW
    pair.decision_rationale = (
        f"Agent failed to parse decision after {RETRY_LIMIT+1} attempts: {last_error}"
    )
    pair.decided_at  = datetime.utcnow().isoformat()
    pair.decided_by  = "AI_STEWARD/FALLBACK"
    return pair


# ─────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────

def run_steward_agent(
    pairs:           list[MatchPair],
    api_key:         str  = "",
    on_progress:     Optional[Callable[[str], None]] = None,
    delay_between:   float = 0.2,
    backend_type:    str   = "mock",
    region:          str   = "us-east-1",
    ollama_model:    str   = "llama3.1",
    azure_endpoint:  str   = "",
    azure_key:       str   = "",
    azure_deploy:    str   = "gpt-4o",
    backend:         Optional[LLMBackend] = None,
) -> list[MatchPair]:
    """
    Run the AI Steward Agent across all PENDING candidate pairs.

    Args:
        pairs:          MatchPair list from the matcher (PENDING decisions only).
        api_key:        Anthropic API key (for anthropic backend).
        on_progress:    Optional UI callback for live status updates.
        delay_between:  Seconds between calls (rate limit safety).
        backend_type:   Backend to use: mock | anthropic | bedrock | azure | ollama
        region:         AWS/Azure region.
        ollama_model:   Ollama model name.
        azure_endpoint: Azure OpenAI endpoint.
        azure_key:      Azure OpenAI key.
        azure_deploy:   Azure deployment name.
        backend:        Pre-built backend instance (overrides backend_type).
    """
    # Build backend if not provided
    if backend is None:
        backend = create_backend(
            backend_type   = backend_type,
            api_key        = api_key,
            region         = region,
            ollama_model   = ollama_model,
            azure_endpoint = azure_endpoint,
            azure_key      = azure_key,
            azure_deploy   = azure_deploy,
        )

    # Only process PENDING pairs (AUTO decisions already set by rule engine)
    pending = [p for p in pairs if p.decision == MatchDecision.PENDING]
    already = [p for p in pairs if p.decision != MatchDecision.PENDING]

    if on_progress:
        on_progress(
            f"🚀 Backend: {backend.display_name} | "
            f"Data residency: {backend.data_residency}"
        )
        on_progress(
            f"📋 {len(pending)} pairs queued for AI review "
            f"({len(already)} already decided by Rule Engine)"
        )

    decided: list[MatchPair] = list(already)

    for i, pair in enumerate(pending, 1):
        if on_progress:
            on_progress(f"─── [{i}/{len(pending)}] Pair {pair.pair_id}")

        decided_pair = decide_pair(pair, backend, on_progress)
        decided.append(decided_pair)

        if i < len(pending):
            time.sleep(delay_between)

    if on_progress:
        approved = sum(1 for p in decided if p.decision == MatchDecision.APPROVE)
        rejected = sum(1 for p in decided if p.decision == MatchDecision.REJECT)
        flagged  = sum(1 for p in decided if p.decision == MatchDecision.HUMAN_REVIEW)
        on_progress(
            f"\n✅ Agent complete — "
            f"APPROVED: {approved} | REJECTED: {rejected} | "
            f"HUMAN REVIEW: {flagged}"
        )

    return decided


# ─────────────────────────────────────────────
# Decision statistics
# ─────────────────────────────────────────────

def decision_stats(pairs: list[MatchPair]) -> dict:
    total    = len(pairs)
    approved = sum(1 for p in pairs if p.decision == MatchDecision.APPROVE)
    rejected = sum(1 for p in pairs if p.decision == MatchDecision.REJECT)
    flagged  = sum(1 for p in pairs if p.decision == MatchDecision.HUMAN_REVIEW)
    pending  = sum(1 for p in pairs if p.decision == MatchDecision.PENDING)
    return {
        "total":        total,
        "approved":     approved,
        "rejected":     rejected,
        "human_review": flagged,
        "pending":      pending,
        "auto_rate":    round(
            (approved + rejected) / total * 100, 1
        ) if total else 0.0,
    }
