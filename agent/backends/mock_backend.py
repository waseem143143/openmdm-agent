"""
openmdm-agent · agent/backends/mock_backend.py

Mock backend — pre-canned realistic AI decisions for demos.
Zero cost. Zero API key. Zero network calls.
Indistinguishable from live inference in a demo setting.

The mock analyses the actual pair data and returns contextually
appropriate decisions with realistic clinical rationale.
This is NOT random — it reads field scores and name patterns
to produce decisions that match what Claude would actually say.
"""

from __future__ import annotations
import json
import re
from agent.backends.base import LLMBackend


# ─────────────────────────────────────────────
# Decision logic — reads the prompt and decides
# based on field evidence (mirrors real LLM reasoning)
# ─────────────────────────────────────────────

# Nickname patterns the mock recognises
NICKNAME_PAIRS = {
    frozenset({"james", "jim"}),
    frozenset({"james", "jimmy"}),
    frozenset({"robert", "bob"}),
    frozenset({"robert", "rob"}),
    frozenset({"william", "bill"}),
    frozenset({"william", "will"}),
    frozenset({"michael", "mike"}),
    frozenset({"thomas", "tom"}),
    frozenset({"charles", "chuck"}),
    frozenset({"charles", "charlie"}),
    frozenset({"christopher", "chris"}),
    frozenset({"matthew", "matt"}),
    frozenset({"anthony", "tony"}),
    frozenset({"joseph", "joe"}),
    frozenset({"daniel", "dan"}),
    frozenset({"benjamin", "ben"}),
    frozenset({"alexander", "alex"}),
    frozenset({"nicholas", "nick"}),
    frozenset({"timothy", "tim"}),
    frozenset({"donald", "don"}),
    frozenset({"steven", "steve"}),
    frozenset({"andrew", "andy"}),
    frozenset({"edward", "ed"}),
    frozenset({"edward", "eddie"}),
    frozenset({"jennifer", "jen"}),
    frozenset({"jennifer", "jenny"}),
    frozenset({"elizabeth", "liz"}),
    frozenset({"elizabeth", "beth"}),
    frozenset({"katherine", "kate"}),
    frozenset({"katherine", "kathy"}),
    frozenset({"margaret", "maggie"}),
    frozenset({"margaret", "peggy"}),
    frozenset({"patricia", "pat"}),
    frozenset({"patricia", "trish"}),
    frozenset({"stephanie", "steph"}),
    frozenset({"samantha", "sam"}),
    frozenset({"rebecca", "becky"}),
    frozenset({"deborah", "deb"}),
    frozenset({"barbara", "barb"}),
    frozenset({"kimberly", "kim"}),
    frozenset({"christina", "tina"}),
    frozenset({"jacqueline", "jackie"}),
}


def _extract_field_scores(prompt: str) -> dict[str, float]:
    """Extract field scores from the formatted prompt."""
    scores = {}
    pattern = r"(\w+)\s*\|\s*A:.*?\|\s*B:.*?\|\s*score=([\d.]+)"
    for match in re.finditer(pattern, prompt):
        field = match.group(1).lower()
        score = float(match.group(2))
        scores[field] = score
    return scores


def _extract_field_values(prompt: str) -> dict[str, tuple[str, str]]:
    """Extract field values A and B from the prompt."""
    values = {}
    pattern = r"(\w+)\s*\|\s*A:\s*([^\|]+)\|\s*B:\s*([^\|]+)\|"
    for match in re.finditer(pattern, prompt):
        field = match.group(1).lower()
        val_a = match.group(2).strip()
        val_b = match.group(3).strip()
        values[field] = (val_a, val_b)
    return values


def _is_nickname(name_a: str, name_b: str) -> bool:
    pair = frozenset({name_a.lower().strip(), name_b.lower().strip()})
    return pair in NICKNAME_PAIRS


def _mock_decision(prompt: str) -> dict:
    """
    Analyse the prompt and return a contextually appropriate decision.
    Mirrors the reasoning a real LLM would apply.
    """
    scores = _extract_field_scores(prompt)
    values = _extract_field_values(prompt)

    dob_score   = scores.get("date_of_birth", 0.0)
    ssn_score   = scores.get("ssn_last4",     0.0)
    phone_score = scores.get("phone",         0.0)
    last_score  = scores.get("last_name",     0.0)
    first_score = scores.get("first_name",    0.0)
    addr_score  = scores.get("address",       0.0)
    email_score = scores.get("email",         0.0)

    first_vals = values.get("first_name", ("", ""))
    last_vals  = values.get("last_name",  ("", ""))
    is_nick    = _is_nickname(first_vals[0], first_vals[1])

    # Check MRN match from prompt
    mrn_match = "✓ MATCH" in prompt and "MRN" in prompt

    # ── Decision logic ────────────────────────────────────────────────────

    # STRONG APPROVE — SSN + DOB both exact
    if ssn_score == 1.0 and dob_score == 1.0:
        evidence_for = [
            f"ssn_last4: exact match {values.get('ssn_last4',('',''))[0]}",
            f"date_of_birth: exact match {values.get('date_of_birth',('',''))[0]}",
        ]
        if phone_score == 1.0:
            evidence_for.append("phone: normalised to identical 10-digit value")
        if last_score >= 0.95:
            evidence_for.append(f"last_name: {last_score:.2f} similarity (near-exact)")
        if is_nick:
            evidence_for.append(
                f"first_name: {first_vals[0]}/{first_vals[1]} is a known nickname variant"
            )

        return {
            "decision": "APPROVE",
            "confidence": "HIGH",
            "rationale": (
                f"SSN-last4 and date of birth both match exactly — "
                f"these are the two strongest identity anchors in clinical MDM. "
                f"{'Phone normalises to the same 10-digit value. ' if phone_score==1.0 else ''}"
                f"{'First name ' + first_vals[0] + '/' + first_vals[1] + ' is a well-known nickname pattern, not a conflict. ' if is_nick else ''}"
                f"Evidence is deterministic. Records represent the same patient."
            ),
            "key_evidence_for":     evidence_for,
            "key_evidence_against": [] if last_score >= 0.90 else [
                f"last_name: {last_vals[0]} vs {last_vals[1]} ({last_score:.2f}) — possible typo"
            ],
        }

    # APPROVE — DOB + phone both exact, strong name
    if dob_score == 1.0 and phone_score == 1.0 and last_score >= 0.90:
        evidence_for = [
            f"date_of_birth: exact match",
            f"phone: exact match after normalisation",
            f"last_name: {last_score:.2f} similarity",
        ]
        if is_nick:
            evidence_for.append(
                f"first_name: {first_vals[0]}/{first_vals[1]} recognised nickname pair"
            )
        if mrn_match:
            evidence_for.append("MRN: matching clinical identifier across source systems")

        return {
            "decision": "APPROVE",
            "confidence": "HIGH",
            "rationale": (
                f"Date of birth and phone number both match exactly. "
                f"{'MRN matches across source systems — strong clinical confirmation. ' if mrn_match else ''}"
                f"Last name similarity {last_score:.2f} is consistent with minor transcription variation. "
                f"{'First name ' + first_vals[0] + '/' + first_vals[1] + ' is a standard nickname — not a conflict. ' if is_nick else ''}"
                f"Missing SSN in secondary source is expected and does not constitute a conflict."
            ),
            "key_evidence_for":     evidence_for,
            "key_evidence_against": [
                f"ssn_last4: not present in secondary source (missing, not conflicting)"
            ] if ssn_score == 0.0 else [],
        }

    # APPROVE — last name typo + DOB exact + phone exact (the 0.95-0.98 cases)
    if dob_score == 1.0 and phone_score == 1.0 and 0.83 <= last_score < 0.97:
        typo_explanation = {
            frozenset({"walsh", "welsh"}):       "Walsh/Welsh — vowel variant, same phonetic root",
            frozenset({"price", "pryce"}):       "Price/Pryce — y/i variant, common transcription error",
            frozenset({"stone", "stoan"}):       "Stone/Stoan — vowel transposition, likely data entry error",
            frozenset({"ryan", "ryon"}):         "Ryan/Ryon — y substitution, phonetically identical",
            frozenset({"nash", "nesh"}):         "Nash/Nesh — vowel substitution, common registration error",
            frozenset({"miles", "myles"}):       "Miles/Myles — y/i variant, well-known spelling alternative",
            frozenset({"gray", "grey"}):         "Gray/Grey — legitimate spelling variants of the same name",
            frozenset({"hunt", "hant"}):         "Hunt/Hant — vowel substitution, likely OCR or data entry error",
            frozenset({"pearce", "pierce"}):     "Pearce/Pierce — vowel variant, phonetically equivalent",
            frozenset({"burns", "berns"}):       "Burns/Berns — vowel substitution, common transcription error",
            frozenset({"mitchell", "mitchel"}):  "Mitchell/Mitchel — double-l drop, frequent data entry shortcut",
            frozenset({"phillips", "philips"}):  "Phillips/Philips — double-l drop, common transcription variant",
            frozenset({"harrison", "harison"}):  "Harrison/Harison — double-r drop, likely OCR artifact",
            frozenset({"sullivan", "sulivan"}):  "Sullivan/Sulivan — double-l drop, frequent in claims systems",
            frozenset({"williams", "wiliams"}):  "Williams/Wiliams — double-l drop, very common in insurance data",
            frozenset({"anderson", "andersen"}): "Anderson/Andersen — son/sen Scandinavian spelling variant",
            frozenset({"hoffmann", "hoffman"}):  "Hoffmann/Hoffman — double-n variant, German name spelling",
            frozenset({"campbell", "cambel"}):   "Campbell/Cambel — double-p/l drop, data entry shortcut",
            frozenset({"bennett", "benet"}):     "Bennett/Benet — double-t/n drop, common transcription error",
        }
        last_pair    = frozenset({last_vals[0].lower(), last_vals[1].lower()})
        explanation  = typo_explanation.get(last_pair,
            f"{last_vals[0]}/{last_vals[1]} — phonetically similar, likely data entry variation ({last_score:.2f})"
        )

        return {
            "decision": "APPROVE",
            "confidence": "MEDIUM",
            "rationale": (
                f"Last name variation '{last_vals[0]}' vs '{last_vals[1]}' scores {last_score:.2f} — "
                f"this is consistent with: {explanation}. "
                f"Date of birth matches exactly — strong identity anchor. "
                f"Phone normalises to the same 10-digit value — confirms same individual. "
                f"{'MRN matches across systems. ' if mrn_match else ''}"
                f"SSN absent in secondary source system — this is expected in claims/pharmacy data, not a conflict. "
                f"Weight of evidence supports same patient with transcription variance in last name."
            ),
            "key_evidence_for": [
                f"date_of_birth: exact match — primary identity anchor",
                f"phone: exact match after normalisation",
                f"last_name: {explanation}",
                "ssn_last4: absent in secondary source — not a conflict",
            ],
            "key_evidence_against": [
                f"last_name: {last_score:.2f} similarity — below auto-approve threshold, warrants review",
            ],
        }

    # REJECT — DOB mismatch (hard anchor fail)
    if dob_score == 0.0 and ssn_score == 0.0:
        return {
            "decision": "REJECT",
            "confidence": "HIGH",
            "rationale": (
                f"Date of birth does not match and SSN-last4 is either absent or conflicting. "
                f"Without agreement on at least one hard identity anchor (DOB or SSN), "
                f"name similarity alone is insufficient to confirm the same patient. "
                f"These are likely different patients who share a similar name."
            ),
            "key_evidence_for":     [],
            "key_evidence_against": [
                "date_of_birth: no match — hard anchor failure",
                "ssn_last4: absent or conflicting — cannot confirm identity",
            ],
        }

    # HUMAN REVIEW — genuinely ambiguous
    return {
        "decision": "HUMAN_REVIEW",
        "confidence": "LOW",
        "rationale": (
            f"Evidence is genuinely ambiguous. "
            f"DOB score: {dob_score:.2f}. Phone score: {phone_score:.2f}. "
            f"Last name: {last_score:.2f}. SSN: {'present and matching' if ssn_score==1.0 else 'absent or conflicting'}. "
            f"Insufficient strong anchors to make a confident automated decision. "
            f"Recommend human data steward review with access to original source records."
        ),
        "key_evidence_for": [
            f"last_name: {last_score:.2f} similarity" if last_score > 0.7 else None,
            f"phone: match" if phone_score == 1.0 else None,
        ],
        "key_evidence_against": [
            f"date_of_birth: {dob_score:.2f} — not confirmed" if dob_score < 1.0 else None,
            "ssn_last4: not available for confirmation" if ssn_score == 0.0 else None,
        ],
    }


# ─────────────────────────────────────────────
# Mock Backend class
# ─────────────────────────────────────────────

class MockBackend(LLMBackend):
    """
    Mock LLM backend for demos — zero cost, zero API key, zero network.

    Analyses actual field scores and name patterns from the prompt
    to return contextually appropriate decisions with realistic rationale.
    Indistinguishable from live inference in a demo setting.
    """

    def __init__(self, region: str = "us-east-1"):
        self._region = region

    def decide(self, system_prompt: str, user_prompt: str) -> str:
        decision_dict = _mock_decision(user_prompt)
        # Clean None values from evidence lists
        decision_dict["key_evidence_for"] = [
            e for e in decision_dict.get("key_evidence_for", []) if e
        ]
        decision_dict["key_evidence_against"] = [
            e for e in decision_dict.get("key_evidence_against", []) if e
        ]
        return json.dumps(decision_dict)

    @property
    def display_name(self) -> str:
        return f"AWS Bedrock — Demo Mode ({self._region})"

    @property
    def model_id(self) -> str:
        return f"bedrock/anthropic.claude-haiku-4-5 [DEMO]"

    @property
    def data_residency(self) -> str:
        return f"AWS {self._region} (Simulated — No Data Transmitted)"

    @property
    def hipaa_eligible(self) -> bool:
        return True
