# 🏥 OpenMDM Agent

**AI-Powered Patient Identity Resolution — No Proprietary MDM Platform Required**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io)
[![Anthropic Claude](https://img.shields.io/badge/AI-Claude%20Sonnet-6366f1.svg)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An open-source MDM Stewardship Agent that resolves duplicate patient records,
> constructs auditable golden records, and scores data quality — powered by
> LLM-driven decision-making with full provenance and an immutable audit trail.

---

## ✨ What It Does

Most enterprise MDM implementations cost $500K–$2M+ in platform licenses and
require years of tuning. **OpenMDM Agent** demonstrates that the core intelligence
of MDM stewardship — match review, merge decisions, survivorship, and data quality
— can be built on open-source Python and a large language model.

| Capability | How |
|---|---|
| **Entity Resolution** | Phonetic blocking + Jaro-Winkler similarity across 7 identity fields |
| **AI Steward Agent** | Claude evaluates each candidate pair and returns APPROVE / REJECT / HUMAN_REVIEW with clinical rationale |
| **Survivorship** | Trust-rank + recency rules build the golden record; every attribute carries provenance |
| **Data Quality Scoring** | Completeness, validity, and composite quality score per golden record |
| **Audit Trail** | Every agent decision logged to SQLite — exportable CSV for compliance |

---

## 🚀 Quickstart (2 minutes)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/openmdm-agent.git
cd openmdm-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Anthropic API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...

# 4. Launch the demo
streamlit run app.py
```

Open http://localhost:8501 — enter your API key in the sidebar and click **▶ Run Pipeline**.

---

## 🏗 Architecture

```
Raw Patient Records (CSV)
        │
        ▼
┌─────────────────────┐
│  Loader & Normalizer │  ← standardize names, DOB, phone, address
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Matcher             │  ← phonetic blocking + Jaro-Winkler scoring
│  (Blocking + Fuzzy)  │     surfaces candidate duplicate pairs
└─────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│         AI STEWARD AGENT (Claude)            │  ← THE CORE
│                                              │
│  Reviews field-level evidence for each pair  │
│  Returns: APPROVE | REJECT | HUMAN_REVIEW    │
│  With:    clinical rationale + key evidence  │
└──────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Survivorship Engine │  ← trust + recency → golden record
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Quality Scorer      │  ← completeness + validity per record
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Audit Logger        │  ← SQLite: every decision, immutable
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Streamlit Dashboard │  ← live demo surface
└─────────────────────┘
```

---

## 📁 Project Structure

```
openmdm-agent/
├── app.py                     # Streamlit dashboard (demo entrypoint)
├── requirements.txt
├── .env.example
├── data/
│   └── sample_patients.csv    # 50 synthetic patients with intentional duplicates
├── mdm/
│   ├── models.py              # Pydantic: PatientRecord, MatchPair, GoldenRecord
│   ├── loader.py              # Ingest + normalize raw records
│   ├── matcher.py             # Blocking + Jaro-Winkler scoring
│   ├── survivorship.py        # Golden record construction
│   └── quality.py             # Completeness + validity scoring
├── agent/
│   ├── steward_agent.py       # LLM batch decision engine
│   └── prompts.py             # System prompt + field-level prompt builder
└── audit/
    └── logger.py              # SQLite audit trail
```

---

## 🤖 Sample Agent Decision

The AI Steward Agent receives structured field-level evidence and returns:

```json
{
  "decision": "APPROVE",
  "confidence": "HIGH",
  "rationale": "Record A (EHR) and Record B (CLAIMS) share an identical date of birth
    (1978-04-12), matching SSN-last4 (4521), and the same MRN (MRN-78234) across
    source systems. The name variation James/Jim is a well-known nickname pattern.
    Phone numbers normalize to the same 10-digit value. Evidence strongly supports
    these as the same patient.",
  "key_evidence_for": [
    "date_of_birth: exact match 1978-04-12",
    "ssn_last4: exact match 4521",
    "mrn: MRN-78234 appears in both sources",
    "phone: normalizes to 3125550192 in both records"
  ],
  "key_evidence_against": [
    "first_name: James vs Jim (0.82 similarity — nickname variation)"
  ]
}
```

---

## 📊 Demo Dataset

`data/sample_patients.csv` contains **50 synthetic patient records** across 5 source systems
(EHR, CLAIMS, LAB, PHARMACY, REGISTRY) with intentionally engineered duplicate scenarios:

- Exact duplicates with address abbreviation differences
- Nickname variations (James/Jim, Bob/Robert, Jennifer/Jen)
- Typos in last name (Rodriguez/Rodriquez)
- Records spanning 2–3 source systems per patient
- Missing fields in non-primary sources

The pipeline surfaces **~34 candidate pairs** and the AI agent auto-resolves ~90%+.

---

## 🔧 Configuration

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key from console.anthropic.com |

Only one key required. No cloud infrastructure, no databases to spin up, no platform licenses.

---

## 🗺 Roadmap

- [ ] **v1.1** — REST API layer (FastAPI) for programmatic access
- [ ] **v1.2** — Databricks-native version (PySpark + Delta Lake audit log)
- [ ] **v1.3** — Snowflake Native App packaging
- [ ] **v2.0** — LangGraph multi-agent workflow (Matcher Agent + Steward Agent + QA Agent)
- [ ] **v2.1** — FHIR R4 patient resource input/output support
- [ ] **v2.2** — HCP/HCO entity type (Life Sciences vertical)

---

## 🏛 Why This Matters

Traditional MDM platforms (Reltio, Informatica, Tibco, IBM) charge $500K–$2M+/year.
The intelligence they provide — match scoring, steward workflows, survivorship rules —
is increasingly replicable with open-source tooling and LLMs.

**OpenMDM Agent** is a proof-of-concept that the data governance community can
build, audit, and own their MDM intelligence layer without vendor lock-in.

---

## 📄 License

MIT License — use freely, contribute openly.

---

## 🙋 Author

Built by **Waseem** — Principal MDM & Data Governance Architect  
20 years in Healthcare, Life Sciences, and Finance MDM  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)
