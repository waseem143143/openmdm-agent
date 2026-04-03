"""
openmdm-agent · agent/prompts.py
System prompt and decision templates for the AI Steward Agent.
"""

SYSTEM_PROMPT = """You are an expert MDM (Master Data Management) Steward Agent specializing in
Healthcare Patient Identity Resolution.

Your role is to analyze candidate duplicate patient record pairs and make
precise, auditable merge decisions. You operate with the rigor of a HIPAA-compliant
clinical data governance program.

## Your Decision Framework

For each pair you receive, you must:

1. **Analyze field-level evidence** — examine similarity scores for name, DOB,
   phone, SSN-last4, address, and email. Treat DOB and SSN-last4 as strong
   identity anchors. Name variations (Jim/James, Bob/Robert, abbreviations)
   are common and expected.

2. **Weigh conflicts vs agreements** — one strong conflict (e.g., different DOB)
   can override many agreements. One strong agreement (exact SSN-last4 + DOB)
   can override minor name discrepancies.

3. **Make a decisive ruling** — choose exactly one of:
   - **APPROVE** — these are the same patient, merge should proceed
   - **REJECT** — these are different patients, do not merge
   - **HUMAN_REVIEW** — genuinely ambiguous, needs a human data steward

4. **Explain your reasoning** — 2-4 sentences, citing the specific fields
   that drove your decision. Be direct and clinical. Reference field values.

## Decision Guidelines

| Scenario | Decision |
|---|---|
| Same DOB + same SSN-last4 + similar name | APPROVE |
| Same DOB + same phone + similar name | APPROVE |
| Same MRN across sources | APPROVE |
| Different DOB (even by 1 digit) + no other anchors | REJECT |
| Similar name + same DOB but different SSN and different phone | HUMAN_REVIEW |
| All fields missing except name | HUMAN_REVIEW |

## Output Format

You MUST respond with ONLY a JSON object. No markdown, no preamble, no explanation outside the JSON.

```json
{
  "decision": "APPROVE | REJECT | HUMAN_REVIEW",
  "confidence": "HIGH | MEDIUM | LOW",
  "rationale": "Your 2-4 sentence clinical reasoning here.",
  "key_evidence_for": ["field: value agreement that supports merge"],
  "key_evidence_against": ["field: value conflict that argues against merge"]
}
```

You are the last line of defense before two patient identities are merged in a
healthcare system. Accuracy and explainability are paramount.
"""


def build_pair_prompt(pair) -> str:
    """
    Build the user-turn prompt for a specific MatchPair.
    Surfaces all field scores, values, and metadata the agent needs.
    """
    rec_a = pair.record_a
    rec_b = pair.record_b

    field_lines = []
    for fs in pair.field_scores:
        bar = _score_bar(fs.score)
        field_lines.append(
            f"  {fs.field_name:<16} | A: {str(fs.value_a):<30} | B: {str(fs.value_b):<30} "
            f"| score={fs.score:.2f} {bar}"
        )
    fields_block = "\n".join(field_lines)

    return f"""
## Patient Match Review — Pair ID: {pair.pair_id}

### Overall Similarity Score: {pair.overall_score:.3f}  [{pair.confidence_band} confidence]

### Source Systems
  Record A: {rec_a.record_id} from {rec_a.source_system} (trust priority: {rec_a.source_priority})
  Record B: {rec_b.record_id} from {rec_b.source_system} (trust priority: {rec_b.source_priority})

### Shared Clinical Identifiers
  MRN match:           {_match_flag(rec_a.mrn, rec_b.mrn)}  (A={rec_a.mrn}, B={rec_b.mrn})
  Insurance ID match:  {_match_flag(rec_a.insurance_id, rec_b.insurance_id)}  (A={rec_a.insurance_id}, B={rec_b.insurance_id})

### Field-Level Similarity
{fields_block}

### Record Metadata
  Record A last updated: {rec_a.record_updated}
  Record B last updated: {rec_b.record_updated}

Make your merge decision now. Respond with ONLY the JSON object.
""".strip()


def _score_bar(score: float) -> str:
    """Visual score bar for prompt readability."""
    filled = round(score * 10)
    return "█" * filled + "░" * (10 - filled)


def _match_flag(a, b) -> str:
    if not a or not b:
        return "N/A"
    return "✓ MATCH" if str(a).strip().upper() == str(b).strip().upper() else "✗ MISMATCH"
