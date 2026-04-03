"""
openmdm-agent · app.py
Streamlit dashboard — full live demo surface.

Run with:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import pandas as pd
import time
from pathlib import Path

st.set_page_config(
    page_title           = "OpenMDM Agent — Patient Identity Resolution",
    page_icon            = "🏥",
    layout               = "wide",
    initial_sidebar_state= "expanded",
)

from mdm.loader          import load_patients_from_csv, records_to_dataframe
from mdm.matcher         import find_candidate_pairs, pairs_summary
from mdm.match_rules     import (default_rules, MatchRule, FieldConfig, rules_to_display)
from mdm.survivorship    import build_all_golden_records
from mdm.quality         import score_all_golden_records, quality_summary
from mdm.crosswalk       import build_crosswalk, crosswalk_to_dataframe, crosswalk_summary
from agent.steward_agent import run_steward_agent, decision_stats
from agent.backends      import BACKEND_OPTIONS, BACKEND_RESIDENCY, create_backend
from audit.logger        import (log_all_decisions, log_golden_created,
                                  read_audit_log, clear_audit_log)
from mdm.models          import MatchDecision

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Decision badges ── */
  .decision-approve { color: #15803d; font-weight: 600; }
  .decision-reject  { color: #b91c1c; font-weight: 600; }
  .decision-human   { color: #b45309; font-weight: 600; }
  .decision-pending { color: #6b7280; font-weight: 600; }

  /* ── Rule attribution badges ── */
  .rule-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 6px;
    vertical-align: middle;
  }
  .rule-deterministic { background: #d1fae5; color: #065f46; }
  .rule-name-dob      { background: #dbeafe; color: #1e40af; }
  .rule-probabilistic { background: #ede9fe; color: #4c1d95; }
  .rule-none          { background: #f3f4f6; color: #374151; }

  /* ── Confidence pills ── */
  .conf-high { background: #d1fae5; color: #065f46; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
  .conf-med  { background: #fef3c7; color: #92400e; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
  .conf-low  { background: #fee2e2; color: #991b1b; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

  /* ── Band chip ── */
  .band-auto-approve { background: #d1fae5; color: #065f46; padding: 2px 10px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }
  .band-auto-reject  { background: #fee2e2; color: #991b1b; padding: 2px 10px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }
  .band-llm-review   { background: #fef3c7; color: #92400e; padding: 2px 10px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }

  /* ── Rationale box ── */
  .rationale-box {
    background: #f8fafc;
    border-left: 3px solid #6366f1;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.86rem;
    color: #374151;
    white-space: pre-wrap;
    line-height: 1.6;
    margin-top: 6px;
  }

  /* ── Rule explain box ── */
  .rule-explain {
    background: #f5f3ff;
    border-left: 3px solid #7c3aed;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.84rem;
    color: #4c1d95;
    margin-top: 8px;
    line-height: 1.6;
  }
  .rule-explain b { color: #3730a3; }

  /* ── Field match/conflict highlights ── */
  .field-match    { color: #15803d; font-weight: 500; }
  .field-conflict { color: #b91c1c; font-weight: 500; }
  .field-missing  { color: #9ca3af; font-style: italic; }

  /* ── Section headers ── */
  .section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 4px;
  }

  /* ── Tab font ── */
  .stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 600; }

  /* ── KPI card tweak ── */
  [data-testid="metric-container"] { background: #f9fafb; border-radius: 8px; padding: 8px 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

RULE_CSS = {
    "RULE_01": "rule-deterministic",
    "RULE_02": "rule-name-dob",
    "RULE_03": "rule-probabilistic",
}
RULE_ICONS = {
    "RULE_01": "🔒",
    "RULE_02": "🧬",
    "RULE_03": "🔮",
}

def rule_badge_html(rule_id: str | None, rule_name: str | None) -> str:
    """Return a coloured HTML badge for a match rule."""
    if not rule_id or not rule_name:
        return "<span class='rule-badge rule-none'>No rule fired</span>"
    css  = RULE_CSS.get(rule_id, "rule-none")
    icon = RULE_ICONS.get(rule_id, "📋")
    short = rule_name.split("(")[0].strip()   # drop parenthetical
    return f"<span class='rule-badge {css}'>{icon} {short}</span>"

def conf_badge_html(band: str) -> str:
    css = {"HIGH": "conf-high", "MEDIUM": "conf-med", "LOW": "conf-low"}.get(band, "conf-low")
    return f"<span class='{css}'>{band}</span>"

def band_badge_html(band: str | None) -> str:
    if not band: return ""
    css = {
        "AUTO_APPROVE": "band-auto-approve",
        "AUTO_REJECT":  "band-auto-reject",
        "LLM_REVIEW":   "band-llm-review",
    }.get(band, "band-llm-review")
    labels = {
        "AUTO_APPROVE": "✅ Auto-approved",
        "AUTO_REJECT":  "❌ Auto-rejected",
        "LLM_REVIEW":   "🤖 LLM reviewed",
    }
    return f"<span class='{css}'>{labels.get(band, band)}</span>"

def field_score_df(pair) -> pd.DataFrame:
    """Build a colour-annotated field comparison dataframe."""
    rows = []
    for fs in pair.field_scores:
        if fs.score >= 0.9:
            signal = "✅ Match"
            cls    = "match"
        elif fs.score >= 0.6:
            signal = "⚠️ Partial"
            cls    = "partial"
        else:
            signal = "❌ Conflict"
            cls    = "conflict"
        bar_filled = round(fs.score * 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        rows.append({
            "Field":    fs.field_name,
            "Record A": fs.value_a or "—",
            "Record B": fs.value_b or "—",
            "Score":    round(fs.score, 3),
            "Match":    bar,
            "Signal":   signal,
        })
    return pd.DataFrame(rows)

def extract_rule_from_decided_by(decided_by: str | None) -> tuple[str | None, str | None]:
    """Parse 'RULE_ENGINE/RULE_01' → ('RULE_01', css_class)."""
    if decided_by and "RULE_ENGINE/" in decided_by:
        rule_id = decided_by.split("/")[-1]
        return rule_id, RULE_CSS.get(rule_id, "rule-none")
    return None, None


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
_DEFAULTS = {
    "records":        None,
    "pairs":          None,
    "decided_pairs":  None,
    "golden_records": None,
    "agent_log":      [],
    "pipeline_done":  False,
    "pipeline_error": None,
    "match_rules":    None,
    "crosswalk":      None,
    "backend_type":   "mock",
    "backend_region": "us-east-1",
    "pair_rule_filter": "All",
    "decision_rule_filter": "All",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 OpenMDM Agent")
    st.caption("AI-Powered Patient Identity Resolution")
    st.divider()

    st.subheader("⚙️ Configuration")

    st.markdown("**🔒 LLM Deployment Mode**")
    backend_choice = st.selectbox(
        "Select backend",
        options     = list(BACKEND_OPTIONS.keys()),
        format_func = lambda k: BACKEND_OPTIONS[k],
        index       = list(BACKEND_OPTIONS.keys()).index(
            st.session_state.get("backend_type", "mock")
        ),
        label_visibility = "collapsed",
        key = "backend_selector",
    )
    st.session_state["backend_type"] = backend_choice

    residency_info = BACKEND_RESIDENCY.get(backend_choice, ("Unknown", False, ""))
    residency_label, is_hipaa, phi_status = residency_info

    st.markdown(
        f"<div style='background:{'#f0fdf4' if is_hipaa else '#fef2f2'};"
        f"border:1px solid {'#bbf7d0' if is_hipaa else '#fecaca'};"
        f"border-radius:8px;padding:8px 12px;font-size:0.82rem;margin-top:4px'>"
        f"{phi_status}<br>"
        f"<span style='color:#6b7280'>Region: {residency_label}</span>"
        f"{'<br><span style=\"color:#16a34a;font-weight:700\">✅ HIPAA Eligible</span>' if is_hipaa else '<br><span style=\"color:#dc2626;font-weight:700\">⚠️ Requires BAA for PHI</span>'}"
        f"</div>",
        unsafe_allow_html=True,
    )

    api_key = azure_endpoint = azure_key = ""
    azure_deploy = "gpt-4o"
    ollama_model = "llama3.1"
    region       = "us-east-1"

    if backend_choice == "mock":
        region = st.selectbox("Simulated AWS Region",
            ["us-east-1","us-west-2","eu-west-1","ap-southeast-1"],
            help="Simulates the data residency region shown in audit logs.")
        st.session_state["backend_region"] = region
        st.success("✅ No API key needed — demo mode active", icon="🎯")

    elif backend_choice == "anthropic":
        env_key = os.environ.get("ANTHROPIC_API_KEY", "")
        api_key = st.text_input("Anthropic API Key", value=env_key, type="password", placeholder="sk-ant-...")
        if env_key:   st.success("✅ Key loaded from .env", icon="🔑")
        elif not api_key: st.warning("Enter API key to enable AI Steward", icon="⚠️")

    elif backend_choice == "bedrock":
        region = st.selectbox("AWS Region", ["us-east-1","us-west-2","eu-west-1","ap-southeast-1"])
        st.session_state["backend_region"] = region
        st.info("Requires AWS credentials (IAM role or env vars).\nModel: anthropic.claude-haiku-4-5", icon="ℹ️")

    elif backend_choice == "azure":
        azure_endpoint = st.text_input("Azure OpenAI Endpoint", placeholder="https://your-resource.openai.azure.com/")
        azure_key      = st.text_input("Azure API Key", type="password", placeholder="your-azure-key")
        azure_deploy   = st.text_input("Deployment Name", value="gpt-4o")
        region         = st.selectbox("Azure Region", ["eastus","westus2","westeurope","southeastasia"])

    elif backend_choice == "ollama":
        ollama_model = st.selectbox("Local Model", ["llama3.1","mistral","phi3:medium","llama3.2"],
            help="Model must be pulled via: ollama pull <model>")
        st.info("Ollama must be running locally.\nInstall: ollama.com → pull model → fully air-gapped.", icon="💻")

    st.divider()
    st.subheader("📂 Dataset")
    data_source = st.radio("Source",
        ["Built-in sample (50 patients)", "Upload your CSV"],
        label_visibility="collapsed")

    data_path = None
    if data_source == "Built-in sample (50 patients)":
        default_csv = Path(__file__).parent / "data" / "sample_patients.csv"
        if default_csv.exists():
            data_path = str(default_csv)
            st.caption("📄 `data/sample_patients.csv` — 50 synthetic patients, 5 source systems")
        else:
            st.error("sample_patients.csv not found in data/")
    else:
        uploaded  = st.file_uploader("Upload patient CSV", type=["csv"],
            help="Must include: record_id, source_system, first_name, last_name, date_of_birth")
        data_path = uploaded

    st.divider()
    st.subheader("🔄 Pipeline Control")
    run_col, rst_col = st.columns(2)
    with run_col:
        run_btn = st.button("▶ Run Pipeline", use_container_width=True, type="primary",
            disabled=(data_path is None))
    with rst_col:
        reset_btn = st.button("↺ Reset", use_container_width=True)

    if reset_btn:
        for key in list(st.session_state.keys()): del st.session_state[key]
        clear_audit_log()
        st.rerun()

    st.divider()
    if backend_choice == "mock":
        st.info("🎯 **Demo Mode** — AI decisions are pre-canned and realistic.\n\nNo API key, no cost, no data transmitted.\n\nSwitch to Bedrock/Azure for live enterprise deployment.", icon="💡")
    elif backend_choice == "anthropic" and not api_key:
        st.warning("No API key — AI Steward will be skipped.", icon="⚠️")
    else:
        st.info(f"🤖 **{BACKEND_OPTIONS[backend_choice]}** active.\n\nData residency: {residency_label}", icon="✅")

    st.divider()
    st.caption("**openmdm-agent** v1.1  \nClaude · Streamlit · Python  \n\n[GitHub ↗](https://github.com/yourusername/openmdm-agent)")


# ─────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────
st.markdown("## 🏥 OpenMDM Agent — Patient Identity Resolution")
st.markdown(
    "An **open-source AI MDM Stewardship Agent** that resolves duplicate patient records, "
    "constructs auditable golden records, and scores data quality — "
    "**without any proprietary MDM platform**."
)
st.divider()


# ─────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────
if run_btn and data_path is not None:
    st.session_state["pipeline_error"] = None
    progress_bar = st.progress(0, text="Starting pipeline...")
    status_box   = st.empty()

    try:
        status_box.info("**Step 1 / 5** — Loading and normalising patient records...")
        progress_bar.progress(10, text="Loading records...")
        records = load_patients_from_csv(data_path)
        st.session_state["records"] = records
        time.sleep(0.3)

        status_box.info(f"**Step 2 / 5** — Blocking & scoring {len(records)} records...")
        progress_bar.progress(25, text="Finding duplicate candidates...")
        active_rules = st.session_state.get("match_rules") or default_rules()
        pairs = find_candidate_pairs(records, rules=active_rules)
        st.session_state["pairs"] = pairs
        time.sleep(0.3)

        skip_agent = (backend_choice == "anthropic" and not api_key)
        if skip_agent:
            status_box.warning(
                f"**Step 3 / 5** — AI Steward Agent skipped (no API key). "
                f"{sum(1 for p in pairs if p.decision.value == 'PENDING')} pairs remain **PENDING**.")
            decided_pairs = pairs
            st.session_state["agent_log"] = []
        else:
            pending_count = sum(1 for p in pairs if p.decision.value == "PENDING")
            status_box.info(
                f"**Step 3 / 5** — AI Steward Agent reviewing "
                f"**{pending_count} pending pairs** via **{BACKEND_OPTIONS[backend_choice]}**...")
            progress_bar.progress(40, text="AI Agent reasoning over pairs...")
            log_lines: list[str] = []

            def _capture(msg: str): log_lines.append(msg)

            decided_pairs = run_steward_agent(
                pairs,
                api_key        = api_key,
                on_progress    = _capture,
                backend_type   = backend_choice,
                region         = region,
                ollama_model   = ollama_model,
                azure_endpoint = azure_endpoint,
                azure_key      = azure_key,
                azure_deploy   = azure_deploy,
            )
            st.session_state["agent_log"] = log_lines
            log_all_decisions(decided_pairs)

        st.session_state["decided_pairs"] = decided_pairs
        progress_bar.progress(65, text="Agent decisions complete...")

        status_box.info("**Step 4 / 5** — Building golden records via survivorship rules...")
        progress_bar.progress(75, text="Constructing golden records...")
        golden_records = build_all_golden_records(decided_pairs, records)
        time.sleep(0.3)

        status_box.info("**Step 5 / 5** — Scoring data quality...")
        progress_bar.progress(90, text="Scoring quality...")
        golden_records = score_all_golden_records(golden_records)
        for gr in golden_records: log_golden_created(gr)
        st.session_state["golden_records"] = golden_records

        crosswalk_entries = build_crosswalk(golden_records, decided_pairs, records)
        st.session_state["crosswalk"]     = crosswalk_entries
        st.session_state["pipeline_done"] = True

        progress_bar.progress(100, text="Complete!")
        status_box.success("✅ Pipeline complete!")
        time.sleep(1.0)

    except Exception as exc:
        st.session_state["pipeline_error"] = str(exc)
        status_box.error(f"❌ Pipeline error: {exc}")
        st.exception(exc)
    finally:
        progress_bar.empty()
        status_box.empty()
        if st.session_state["pipeline_done"]:
            st.rerun()


if st.session_state.get("pipeline_error"):
    st.error(f"⚠️ Last run failed: {st.session_state['pipeline_error']}")


# ─────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────
if st.session_state["pipeline_done"]:

    records        = st.session_state["records"]
    pairs          = st.session_state["pairs"]
    decided_pairs  = st.session_state["decided_pairs"]
    golden_records = st.session_state["golden_records"]

    p_sum  = pairs_summary(pairs)
    d_stat = decision_stats(decided_pairs)
    q_sum  = quality_summary(golden_records)

    # KPI strip
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📋 Source Records",  len(records))
    k2.metric("🔗 Candidate Pairs", p_sum["total_pairs"])
    k3.metric("✅ Approved Merges", d_stat["approved"])
    k4.metric("⭐ Golden Records",  q_sum.get("total_golden_records", "—"))
    k5.metric("📊 Avg Quality",     f"{q_sum.get('avg_quality', 0)*100:.1f}%")
    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "⚙️ Match Rules",
        "📋 Source Records",
        "🔗 Candidate Pairs",
        "🤖 Agent Decisions",
        "⭐ Golden Records",
        "🔀 Crosswalk",
        "📜 Audit Log",
    ])

    # ══════════════════════════════════════════
    # Tab 1 — Match Rules
    # ══════════════════════════════════════════
    with tab1:
        st.subheader("⚙️ Match Rule Configuration")
        st.caption(
            "Configure the 3 match rules used by the pipeline. "
            "Rules are evaluated in priority order — first rule that produces "
            "a confident AUTO decision wins. Ambiguous pairs go to the AI Steward Agent. "
            "This mirrors how Reltio and Informatica MDM define match rules."
        )

        # Rule legend
        st.markdown("""
<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px'>
  <span class='rule-badge rule-deterministic'>🔒 RULE_01 — Deterministic</span>
  <span class='rule-badge rule-name-dob'>🧬 RULE_02 — Name + DOB + Phone</span>
  <span class='rule-badge rule-probabilistic'>🔮 RULE_03 — Full Probabilistic</span>
</div>
""", unsafe_allow_html=True)

        if st.session_state["match_rules"] is None:
            st.session_state["match_rules"] = default_rules()
        current_rules: list[MatchRule] = st.session_state["match_rules"]

        st.markdown("### Current Rules")
        rule_display = rules_to_display(current_rules)
        st.dataframe(pd.DataFrame(rule_display), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Configure Rules")

        updated_rules = []
        for rule in sorted(current_rules, key=lambda r: r.priority):
            rule_css  = RULE_CSS.get(rule.rule_id, "rule-none")
            rule_icon = RULE_ICONS.get(rule.rule_id, "📋")
            with st.expander(
                f"{rule_icon} Rule {rule.priority}: {rule.rule_name}   "
                f"{'✅ Enabled' if rule.enabled else '❌ Disabled'}",
                expanded=False,
            ):
                st.markdown(
                    f"<span class='rule-badge {rule_css}'>{rule_icon} {rule.rule_id}</span>",
                    unsafe_allow_html=True,
                )
                col_en, col_pri = st.columns([3, 1])
                with col_en:
                    enabled = st.toggle("Enable this rule", value=rule.enabled, key=f"enabled_{rule.rule_id}")
                with col_pri:
                    st.metric("Priority", rule.priority)

                st.caption(rule.description)
                st.markdown("**Fields & Weights:**")
                field_df = pd.DataFrame([
                    {"Field": fc.field_name, "Weight": fc.weight,
                     "Algorithm": fc.algorithm, "Required": "✅" if fc.required else "—"}
                    for fc in rule.fields
                ])
                st.dataframe(field_df, use_container_width=True, hide_index=True)

                st.markdown("**Decision Thresholds:**")
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    approve_thresh = st.slider("Auto-Approve threshold (≥)",
                        min_value=0.70, max_value=1.00,
                        value=float(rule.auto_approve_threshold), step=0.01,
                        key=f"approve_{rule.rule_id}",
                        help="Pairs scoring at or above this threshold are AUTO-APPROVED — no LLM call needed.")
                with t_col2:
                    reject_thresh = st.slider("Auto-Reject threshold (<)",
                        min_value=0.30, max_value=0.80,
                        value=float(rule.auto_reject_threshold), step=0.01,
                        key=f"reject_{rule.rule_id}",
                        help="Pairs scoring below this threshold are AUTO-REJECTED — no LLM call needed.")

                if approve_thresh <= reject_thresh:
                    st.warning("⚠️ Auto-Approve threshold must be higher than Auto-Reject threshold.")

                st.info(
                    f"🤖 **LLM Review band:** scores between "
                    f"`{reject_thresh:.2f}` and `{approve_thresh:.2f}` → sent to AI Steward Agent",
                    icon="ℹ️",
                )
                updated_rules.append(MatchRule(
                    rule_id=rule.rule_id, rule_name=rule.rule_name,
                    description=rule.description, fields=rule.fields,
                    auto_approve_threshold=approve_thresh,
                    auto_reject_threshold=reject_thresh,
                    enabled=enabled, priority=rule.priority,
                ))

        st.divider()
        apply_col, reset_col = st.columns([2, 1])
        with apply_col:
            if st.button("💾 Apply Rule Changes & Re-Run Pipeline", type="primary", use_container_width=True):
                st.session_state["match_rules"]    = updated_rules
                st.session_state["pipeline_done"]  = False
                st.session_state["pairs"]          = None
                st.session_state["decided_pairs"]  = None
                st.session_state["golden_records"] = None
                st.rerun()
        with reset_col:
            if st.button("↺ Reset to Defaults", use_container_width=True):
                st.session_state["match_rules"] = default_rules()
                st.rerun()

        if st.session_state["pairs"]:
            st.divider()
            st.markdown("### 💰 Cost Impact of Current Rules")
            s = pairs_summary(st.session_state["pairs"])
            ci1, ci2, ci3, ci4 = st.columns(4)
            ci1.metric("Total Pairs",      s["total_pairs"])
            ci2.metric("✅ Auto-Approved", s["auto_approved"], help="Rule Engine decision — $0 LLM cost")
            ci3.metric("❌ Auto-Rejected", s["auto_rejected"], help="Rule Engine decision — $0 LLM cost")
            ci4.metric("🤖 LLM Needed",   s["llm_needed"],   help=f"~${s['llm_needed'] * 0.003:.2f} estimated API cost")
            st.caption(
                f"Auto-decision rate: **{((s['auto_approved']+s['auto_rejected'])/max(s['total_pairs'],1)*100):.1f}%** "
                f"of pairs resolved without LLM. "
                f"Estimated AI cost: **~${s['llm_needed']*0.003:.2f}**"
            )

    # ══════════════════════════════════════════
    # Tab 2 — Source Records
    # ══════════════════════════════════════════
    with tab2:
        st.subheader("Source Patient Records")
        st.caption(
            f"{len(records)} records across "
            f"{len(set(r.source_system for r in records))} source systems. "
            "Fields normalised: name, phone, address, gender."
        )
        df_records = records_to_dataframe(records)
        st.dataframe(df_records, use_container_width=True, height=420)

        src_counts = (
            df_records["source"].value_counts().reset_index()
            .rename(columns={"source": "Source System", "count": "Records"})
        )
        st.subheader("Source System Breakdown")
        st.dataframe(src_counts, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════
    # Tab 3 — Candidate Pairs  (REDESIGNED)
    # ══════════════════════════════════════════
    with tab3:
        st.subheader("Candidate Duplicate Pairs")
        st.caption(
            "Surfaced via phonetic blocking (Soundex + DOB) + "
            "Jaro-Winkler similarity across identity fields. "
            "Each pair shows which match rule fired and why."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Pairs",           p_sum["total_pairs"])
        c2.metric("🔴 High (≥0.85)",       p_sum["high_confidence"])
        c3.metric("🟡 Medium (0.70–0.85)", p_sum["medium_confidence"])
        c4.metric("🟢 Low (0.55–0.70)",    p_sum["low_confidence"])
        st.divider()

        # ── Filters ──
        fcol1, fcol2 = st.columns([1, 1])
        with fcol1:
            show_band = st.multiselect(
                "Filter by confidence band",
                ["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM", "LOW"],
            )
        with fcol2:
            rule_opts = ["All", "🔒 RULE_01 — Deterministic", "🧬 RULE_02 — Name+DOB+Phone", "🔮 RULE_03 — Probabilistic", "No rule fired"]
            rule_filter_label = st.selectbox("Filter by match rule", rule_opts, key="pair_rule_filter_sel")

        # Map label back to rule ID
        rule_filter_map = {
            "All": None,
            "🔒 RULE_01 — Deterministic":    "RULE_01",
            "🧬 RULE_02 — Name+DOB+Phone":   "RULE_02",
            "🔮 RULE_03 — Probabilistic":     "RULE_03",
            "No rule fired":                  "__NONE__",
        }
        active_rule_filter = rule_filter_map.get(rule_filter_label)

        st.divider()

        # ── Pair cards ──
        visible = 0
        for pair in sorted(pairs, key=lambda p: p.overall_score, reverse=True):
            if pair.confidence_band not in show_band:
                continue
            if active_rule_filter:
                if active_rule_filter == "__NONE__" and pair.match_rule_id:
                    continue
                elif active_rule_filter != "__NONE__" and pair.match_rule_id != active_rule_filter:
                    continue
            visible += 1

            conf_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(pair.confidence_band, "⚪")

            # Build expander title with rule badge inline
            rule_short = ""
            if pair.match_rule_id:
                icon = RULE_ICONS.get(pair.match_rule_id, "📋")
                rule_short = f" · {icon} {pair.match_rule_id}"

            with st.expander(
                f"{conf_icon} **{pair.pair_id}** — "
                f"{pair.record_a.first_name} {pair.record_a.last_name} "
                f"({pair.record_a.source_system}) ↔ "
                f"{pair.record_b.first_name} {pair.record_b.last_name} "
                f"({pair.record_b.source_system})"
                f" | Score: **{pair.overall_score:.3f}**{rule_short}",
                expanded=False,
            ):
                # ── Header row: rule badge + decision band + conf ──
                header_html = (
                    rule_badge_html(pair.match_rule_id, pair.match_rule_name)
                    + " &nbsp; "
                    + band_badge_html(pair.decision_band)
                    + " &nbsp; "
                    + conf_badge_html(pair.confidence_band)
                )
                st.markdown(header_html, unsafe_allow_html=True)
                st.markdown("")

                # ── Source record metadata ──
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(
                        f"**Record A** &nbsp; `{pair.record_a.record_id}`  \n"
                        f"Source: `{pair.record_a.source_system}` "
                        f"(trust priority: {pair.record_a.source_priority})  \n"
                        f"Updated: `{pair.record_a.record_updated or '—'}`"
                    )
                with col_b:
                    st.markdown(
                        f"**Record B** &nbsp; `{pair.record_b.record_id}`  \n"
                        f"Source: `{pair.record_b.source_system}` "
                        f"(trust priority: {pair.record_b.source_priority})  \n"
                        f"Updated: `{pair.record_b.record_updated or '—'}`"
                    )

                # ── Field score table ──
                st.markdown("<div class='section-label' style='margin-top:12px'>Field-level comparison</div>",
                    unsafe_allow_html=True)
                st.dataframe(
                    field_score_df(pair),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score": st.column_config.NumberColumn(format="%.3f"),
                        "Match": st.column_config.TextColumn(width="medium"),
                    }
                )

                # ── Rule explanation ──
                if pair.match_rule_name:
                    rule_id = pair.match_rule_id or ""
                    rule_descriptions = {
                        "RULE_01": (
                            "This pair scored high on <b>SSN last-4 + Date of Birth</b> — "
                            "both deterministic anchor fields. When both match exactly, identity "
                            "is confirmed without needing the AI Steward. "
                            "Equivalent to Reltio Deterministic / Informatica Exact Match."
                        ),
                        "RULE_02": (
                            "Matched via <b>Name + DOB + Phone</b> probabilistic rule — "
                            "optimised for EHR ↔ Claims where SSN is absent. "
                            "High weights on DOB (hard anchor) and phone (stable identifier)."
                        ),
                        "RULE_03": (
                            "Matched via <b>Full 7-Field Probabilistic</b> rule — "
                            "broadest net covering name, DOB, phone, SSN, address, and email. "
                            "Ambiguous pairs in the LLM band are escalated to the AI Steward."
                        ),
                    }
                    rule_desc = rule_descriptions.get(rule_id, "Rule evaluation details not available.")
                    st.markdown(
                        f"<div class='rule-explain'>"
                        f"<b>Why this rule fired ({pair.match_rule_id} · {pair.match_rule_name})</b><br>"
                        f"{rule_desc}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        if visible == 0:
            st.info("No pairs match the current filters.")
        else:
            st.caption(f"Showing {visible} pair(s)")

    # ══════════════════════════════════════════
    # Tab 4 — Agent Decisions  (REDESIGNED)
    # ══════════════════════════════════════════
    with tab4:
        st.subheader("🤖 AI Steward Agent Decisions")

        if d_stat["pending"] == d_stat["total"]:
            st.warning(
                "All pairs are **PENDING** — AI Steward Agent did not run.  \n"
                "Add your Anthropic API key in the sidebar and re-run.",
                icon="🔑",
            )
        else:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("✅ Approved",     d_stat["approved"])
            d2.metric("❌ Rejected",     d_stat["rejected"])
            d3.metric("🔍 Human Review", d_stat["human_review"])
            d4.metric("⚡ Auto-Rate",    f"{d_stat['auto_rate']}%")

            # Rule breakdown
            st.divider()
            st.markdown("<div class='section-label'>Decisions by match rule</div>", unsafe_allow_html=True)
            rule_counts = {}
            for p in decided_pairs:
                if p.decision == MatchDecision.PENDING:
                    continue
                key = p.match_rule_name or "Unknown"
                rule_counts[key] = rule_counts.get(key, 0) + 1
            if rule_counts:
                rc_df = pd.DataFrame([{"Match Rule": k, "Decisions": v} for k, v in rule_counts.items()])
                st.dataframe(rc_df, use_container_width=True, hide_index=True)

            if st.session_state["agent_log"]:
                with st.expander("📋 Agent Run Log", expanded=False):
                    st.code("\n".join(st.session_state["agent_log"]), language=None)

            st.divider()

            # ── Filters ──
            fcol1, fcol2 = st.columns([1, 1])
            with fcol1:
                show_dec = st.multiselect(
                    "Filter by decision",
                    ["APPROVE", "REJECT", "HUMAN_REVIEW"],
                    default=["APPROVE", "REJECT", "HUMAN_REVIEW"],
                )
            with fcol2:
                drule_opts = ["All", "🔒 RULE_01 — Deterministic", "🧬 RULE_02 — Name+DOB+Phone", "🔮 RULE_03 — Probabilistic"]
                drule_label = st.selectbox("Filter by match rule", drule_opts, key="dec_rule_filter")

            drule_map = {
                "All": None,
                "🔒 RULE_01 — Deterministic":  "RULE_01",
                "🧬 RULE_02 — Name+DOB+Phone": "RULE_02",
                "🔮 RULE_03 — Probabilistic":  "RULE_03",
            }
            active_drule = drule_map.get(drule_label)

            # ── Decision cards ──
            visible_d = 0
            for pair in decided_pairs:
                if pair.decision == MatchDecision.PENDING:
                    continue
                if pair.decision.value not in show_dec:
                    continue
                if active_drule and pair.match_rule_id != active_drule:
                    continue
                visible_d += 1

                dec_icon = {"APPROVE": "✅", "REJECT": "❌", "HUMAN_REVIEW": "🔍"}.get(pair.decision.value, "❓")
                css_dec  = {"APPROVE": "decision-approve", "REJECT": "decision-reject",
                            "HUMAN_REVIEW": "decision-human"}.get(pair.decision.value, "decision-pending")

                rule_short = ""
                if pair.match_rule_id:
                    rule_short = f" · {RULE_ICONS.get(pair.match_rule_id, '📋')} {pair.match_rule_id}"

                # Determine who decided
                is_auto = pair.decided_by and "RULE_ENGINE" in (pair.decided_by or "")
                decider = "Rule Engine" if is_auto else "AI Steward"

                with st.expander(
                    f"{dec_icon} {pair.pair_id} — "
                    f"{pair.record_a.first_name} {pair.record_a.last_name} "
                    f"({pair.record_a.source_system}) ↔ "
                    f"{pair.record_b.first_name} {pair.record_b.last_name} "
                    f"({pair.record_b.source_system})"
                    f" | **{pair.decision.value}**{rule_short}",
                    expanded=False,
                ):
                    # Status row
                    status_html = (
                        f"<b>Decision:</b> <span class='{css_dec}'>{pair.decision.value}</span>"
                        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                        f"Score: <code>{pair.overall_score:.3f}</code>"
                        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                        f"Decided by: <b>{decider}</b>"
                        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                        + conf_badge_html(pair.confidence_band)
                    )
                    st.markdown(status_html, unsafe_allow_html=True)

                    # Rule badge row
                    rule_html = (
                        rule_badge_html(pair.match_rule_id, pair.match_rule_name)
                        + "&nbsp;&nbsp;"
                        + band_badge_html(pair.decision_band)
                    )
                    st.markdown(rule_html, unsafe_allow_html=True)

                    # Rationale
                    if pair.decision_rationale:
                        st.markdown(
                            f"<div class='rationale-box'>{pair.decision_rationale}</div>",
                            unsafe_allow_html=True,
                        )

                    # Compact field table (only key agreements / conflicts)
                    if pair.field_scores:
                        agree   = [fs for fs in pair.field_scores if fs.score >= 0.9]
                        conflict= [fs for fs in pair.field_scores if fs.score < 0.6]
                        if agree or conflict:
                            st.markdown("")
                            col_ag, col_cf = st.columns(2)
                            with col_ag:
                                st.markdown("<div class='section-label'>Key agreements</div>", unsafe_allow_html=True)
                                for fs in agree:
                                    st.markdown(
                                        f"<span class='field-match'>✅ {fs.field_name}</span>"
                                        f"<span style='font-size:0.8rem;color:#6b7280'> — {fs.value_a}</span>",
                                        unsafe_allow_html=True,
                                    )
                            with col_cf:
                                st.markdown("<div class='section-label'>Key conflicts</div>", unsafe_allow_html=True)
                                for fs in conflict:
                                    st.markdown(
                                        f"<span class='field-conflict'>❌ {fs.field_name}</span>"
                                        f"<span style='font-size:0.8rem;color:#6b7280'> — {fs.value_a} vs {fs.value_b}</span>",
                                        unsafe_allow_html=True,
                                    )

            if visible_d == 0:
                st.info("No decisions match the current filters.")

    # ══════════════════════════════════════════
    # Tab 5 — Golden Records
    # ══════════════════════════════════════════
    with tab5:
        st.subheader("⭐ Golden Records — Single Source of Truth")
        st.caption(
            "Constructed via trust-rank + recency survivorship. "
            "Every attribute carries provenance showing which source won and why."
        )
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total Golden Records", q_sum.get("total_golden_records", "—"))
        q2.metric("Avg Completeness",     f"{q_sum.get('avg_completeness', 0)*100:.1f}%")
        q3.metric("Avg Validity",         f"{q_sum.get('avg_validity', 0)*100:.1f}%")
        q4.metric("Avg Quality",          f"{q_sum.get('avg_quality', 0)*100:.1f}%")
        st.divider()

        gr_rows = []
        for gr in golden_records:
            gr_rows.append({
                "Golden ID":      gr.golden_id,
                "Full Name":      gr.full_name,
                "DOB":            gr.date_of_birth,
                "Gender":         gr.gender,
                "Phone":          gr.phone,
                "City":           gr.city,
                "State":          gr.state,
                "Sources":        ", ".join(sorted(set(gr.source_systems))),
                "Merged Records": len(gr.source_record_ids),
                "Quality %":      f"{(gr.quality_score or 0)*100:.1f}",
                "Complete %":     f"{(gr.completeness_score or 0)*100:.1f}",
                "Valid %":        f"{(gr.validity_score or 0)*100:.1f}",
            })
        gr_df = pd.DataFrame(gr_rows)

        min_q = st.slider("Min quality score (%)", 0, 100, 0, 5)
        filtered = gr_df[gr_df["Quality %"].astype(float) >= min_q]
        st.caption(f"Showing {len(filtered)} / {len(gr_df)} golden records")
        st.dataframe(filtered, use_container_width=True, height=380)
        st.download_button("⬇ Export Golden Records (CSV)", data=gr_df.to_csv(index=False),
            file_name="openmdm_golden_records.csv", mime="text/csv")

        st.subheader("🔍 Attribute Provenance Explorer")
        st.caption("See exactly which source system won each attribute and why.")
        label_map = {f"{gr.golden_id} — {gr.full_name}": gr for gr in golden_records}
        sel = st.selectbox("Select Golden Record", list(label_map.keys()))
        sel_gr = label_map.get(sel)
        if sel_gr:
            prov_rows = [
                {"Attribute": p.attribute, "Winning Value": p.winning_value or "—",
                 "Source": p.winning_source, "Record ID": p.winning_record_id,
                 "Rule": p.rule_applied}
                for p in sel_gr.attribute_provenance if p.winning_value
            ]
            if prov_rows:
                st.dataframe(pd.DataFrame(prov_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No populated attributes to display.")

    # ══════════════════════════════════════════
    # Tab 6 — Crosswalk
    # ══════════════════════════════════════════
    with tab6:
        st.subheader("🔀 Crosswalk — Source to Golden Record Mapping")
        st.caption(
            "Every source record mapped to its golden record. "
            "This is the MDM cross-reference (XREF) table — "
            "equivalent to Reltio Crosswalk, Informatica Cross-Reference, "
            "and IBM MDM XREF. Use this to trace any source record "
            "back to its master identity."
        )

        crosswalk_entries = st.session_state.get("crosswalk", [])

        if not crosswalk_entries:
            st.info("Run the pipeline to generate the crosswalk.")
        else:
            xw_sum = crosswalk_summary(crosswalk_entries)
            xw_df  = crosswalk_to_dataframe(crosswalk_entries)

            x1, x2, x3, x4, x5 = st.columns(5)
            x1.metric("Source Records",     xw_sum["total_source_records"])
            x2.metric("Golden Records",     xw_sum["total_golden_records"])
            x3.metric("Merged Records",     xw_sum["merged_records"],    help="Source records merged into a shared golden record")
            x4.metric("Singletons",         xw_sum["singleton_records"], help="Source records with no duplicate — their own golden record")
            x5.metric("Consolidation Rate", f"{xw_sum['consolidation_rate']}%")

            st.divider()
            st.markdown("**Source System Breakdown**")
            sys_cols = st.columns(len(xw_sum["source_systems"]))
            for col, (sys_name, count) in zip(sys_cols, sorted(xw_sum["source_systems"].items())):
                col.metric(sys_name, count)

            st.divider()
            st.markdown("### 🔍 Explore the Crosswalk")
            f1, f2, f3 = st.columns(3)
            with f1:
                filter_system = st.multiselect("Filter by Source System",
                    options=sorted(xw_df["Source System"].unique()),
                    default=sorted(xw_df["Source System"].unique()))
            with f2:
                filter_merge = st.multiselect("Filter by Merge Type",
                    options=sorted(xw_df["Merge Type"].unique()),
                    default=sorted(xw_df["Merge Type"].unique()))
            with f3:
                search_name = st.text_input("Search by name", placeholder="e.g. Wilson, James...")

            filtered_xw = xw_df[
                xw_df["Source System"].isin(filter_system) &
                xw_df["Merge Type"].isin(filter_merge)
            ]
            if search_name:
                mask = (
                    filtered_xw["Golden Name"].str.contains(search_name, case=False, na=False) |
                    filtered_xw["First Name"].str.contains(search_name, case=False, na=False) |
                    filtered_xw["Last Name"].str.contains(search_name, case=False, na=False)
                )
                filtered_xw = filtered_xw[mask]

            st.caption(f"Showing **{len(filtered_xw)}** of **{len(xw_df)}** crosswalk entries")
            st.dataframe(filtered_xw, use_container_width=True, height=420, hide_index=True)
            st.download_button("⬇ Export Full Crosswalk (CSV)", data=xw_df.to_csv(index=False),
                file_name="openmdm_crosswalk.csv", mime="text/csv")

            st.divider()
            st.markdown("### 🔎 Golden Record Drill-Down")
            st.caption("Select any golden record to see all source records merged under it.")

            gr_options = (
                xw_df.groupby(["Golden ID", "Golden Name"]).size().reset_index(name="Records")
                .apply(lambda row: f"{row['Golden ID']} — {row['Golden Name']} ({row['Records']} source record{'s' if row['Records'] > 1 else ''})", axis=1)
                .tolist()
            )
            selected_gr_label = st.selectbox("Select Golden Record", sorted(gr_options))

            if selected_gr_label:
                selected_gid = selected_gr_label.split(" — ")[0].strip()
                subset = xw_df[xw_df["Golden ID"] == selected_gid]
                merge_count = len(subset)

                if merge_count > 1:
                    st.success(f"✅ **{merge_count} source records merged** into Golden Record `{selected_gid}`", icon="🔀")
                else:
                    st.info("ℹ️ Singleton — 1 source record, no merge performed.", icon="👤")

                display_cols = [
                    "Source Record ID", "Source System", "Source Priority",
                    "First Name", "Last Name", "DOB", "Phone", "Email",
                    "MRN", "Insurance ID", "Address", "Merge Type",
                    "Match Rule", "Match Score", "Decision", "Record Updated",
                ]
                st.dataframe(subset[display_cols], use_container_width=True, hide_index=True)

                # ── Rule badge for this golden record ──
                match_rules_used = subset["Match Rule"].dropna().unique()
                if len(match_rules_used) > 0:
                    st.markdown("<div class='section-label' style='margin-top:10px'>Match rules used</div>", unsafe_allow_html=True)
                    badges = ""
                    for mr in match_rules_used:
                        if mr and mr != "—":
                            # Try to map rule name to ID
                            rid = None
                            for r in (st.session_state.get("match_rules") or default_rules()):
                                if r.rule_name in mr or r.rule_id in mr:
                                    rid = r.rule_id
                                    break
                            badges += rule_badge_html(rid, mr) + " &nbsp;"
                    if badges:
                        st.markdown(badges, unsafe_allow_html=True)

                if merge_count == 2:
                    st.markdown("**Field Comparison (side by side):**")
                    rec1 = subset.iloc[0]
                    rec2 = subset.iloc[1]
                    compare_fields = ["First Name","Last Name","DOB","Phone","Email","MRN","Insurance ID","Address"]
                    comp_rows = []
                    for cf in compare_fields:
                        v1 = str(rec1.get(cf, "—"))
                        v2 = str(rec2.get(cf, "—"))
                        match = "✅" if v1.lower() == v2.lower() else "⚠️"
                        comp_rows.append({"Field": cf, rec1["Source System"]: v1, rec2["Source System"]: v2, "Match": match})
                    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════
    # Tab 7 — Audit Log
    # ══════════════════════════════════════════
    with tab7:
        st.subheader("📜 Immutable Audit Trail")
        st.caption(
            "Every AI Steward decision and golden record creation is logged "
            "with full rationale, model, and timestamp. "
            "Your MDM governance compliance artifact."
        )
        audit_df = read_audit_log()

        if audit_df.empty:
            st.info("No audit events yet. Run the pipeline with an API key to generate them.")
        else:
            a1, a2, a3 = st.columns(3)
            a1.metric("Total Events",          len(audit_df))
            a2.metric("Match Decisions",        len(audit_df[audit_df["event_type"] == "MATCH_DECISION"]))
            a3.metric("Golden Records Created", len(audit_df[audit_df["event_type"] == "GOLDEN_CREATED"]))
            st.divider()

            evt_filter = st.multiselect("Filter event type",
                ["MATCH_DECISION", "GOLDEN_CREATED"],
                default=["MATCH_DECISION", "GOLDEN_CREATED"])
            st.dataframe(audit_df[audit_df["event_type"].isin(evt_filter)],
                use_container_width=True, height=450)
            st.download_button("⬇ Export Audit Log (CSV)", data=audit_df.to_csv(index=False),
                file_name="openmdm_audit_log.csv", mime="text/csv")


# ─────────────────────────────────────────────
# Welcome screen
# ─────────────────────────────────────────────
else:
    st.info("👈 Add your API key in the sidebar and click **▶ Run Pipeline**.", icon="🚀")
    st.markdown("### How it works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**① Load & Normalise**  \nIngests records from any source system CSV. "
            "Standardises names, phones, addresses, DOB.")
    with col2:
        st.markdown("**② Entity Resolution**  \nPhonetic blocking narrows the search space. "
            "Three configurable match rules score each pair.")
    with col3:
        st.markdown("**③ AI Steward Agent**  \nClaude reviews each ambiguous pair, weighs all field evidence, "
            "returns APPROVE / REJECT / HUMAN_REVIEW with clinical rationale.")
    with col4:
        st.markdown("**④ Golden Records**  \nTrust-ranked survivorship builds the master identity. "
            "Full provenance. Quality scored. Audit logged.")
    st.divider()

    # Rule legend on welcome screen
    st.markdown("### Match Rule Engine")
    st.markdown("""
<div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px'>
  <div style='background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:10px 14px;min-width:200px'>
    <span class='rule-badge rule-deterministic'>🔒 RULE_01</span><br>
    <b style='font-size:0.9rem'>Deterministic Identity</b><br>
    <span style='font-size:0.82rem;color:#374151'>SSN-last4 + DOB exact match → AUTO-APPROVE, zero LLM cost</span>
  </div>
  <div style='background:#eff6ff;border:1px solid #93c5fd;border-radius:8px;padding:10px 14px;min-width:200px'>
    <span class='rule-badge rule-name-dob'>🧬 RULE_02</span><br>
    <b style='font-size:0.9rem'>Name + DOB + Phone</b><br>
    <span style='font-size:0.82rem;color:#374151'>High-confidence probabilistic — EHR ↔ Claims without SSN</span>
  </div>
  <div style='background:#f5f3ff;border:1px solid #c4b5fd;border-radius:8px;padding:10px 14px;min-width:200px'>
    <span class='rule-badge rule-probabilistic'>🔮 RULE_03</span><br>
    <b style='font-size:0.9rem'>Full Probabilistic (7-field)</b><br>
    <span style='font-size:0.82rem;color:#374151'>Broadest net — ambiguous pairs escalated to AI Steward</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**No Reltio. No Informatica. No proprietary platform.**  \nOne command: `streamlit run app.py`")
