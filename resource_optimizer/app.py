"""
Resource Optimizer Agent – Streamlit UI
========================================
Cluster scope: pac-mlpcluster01 only.
Namespace scope:
  • Namespaces starting with "odmlp" → accepted automatically.
  • All others → user must confirm the service is deployed via Shuttle.

Powered by LangGraph + LangChain + Gemini 2.5 Flash (internal endpoint).
"""

from __future__ import annotations

import os
import sys

# Ensure imports resolve from the project root
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ── page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Resource Optimizer Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Gradient primary button */
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #f13ab1, #e72744, #fd913c, #f05524);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 12px;
    font-size: 15px;
    font-weight: bold;
    width: 100%;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
div.stButton > button[kind="primary"]:hover {
    filter: brightness(1.15);
    transform: scale(1.03);
    box-shadow: 0 6px 18px rgba(255,82,82,0.45);
}

/* Metric cards */
.rec-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    margin-bottom: 12px;
}
.rec-card.good  { border-left: 5px solid #2ecc71; }
.rec-card.warn  { border-left: 5px solid #e67e22; }
.rec-card.error { border-left: 5px solid #e74c3c; }

/* Section headers */
h2 { color: #f05524; }
</style>
""",
    unsafe_allow_html=True,
)

# ── lazy import of the graph (deferred so Streamlit renders fast) ─────────────
@st.cache_resource(show_spinner=False)
def _load_graph():
    from agent.graph import optimization_graph
    return optimization_graph


# ── helper functions ──────────────────────────────────────────────────────────

SUPPORTED_CLUSTER = "pac-mlpcluster01"


def _validate_namespace(ns: str) -> tuple[bool | None, str]:
    """
    Returns:
      (True,  "")          → namespace accepted (odmlp* prefix)
      (None,  "")          → needs shuttle confirmation (non-odmlp)
      (False, reason_msg)  → invalid input
    """
    ns = ns.strip()
    if not ns:
        return False, "Please enter a namespace."
    if ns.startswith("odmlp"):
        return True, ""
    return None, ""


def _bytes_to_human(b):
    if b is None:
        return "N/A"
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if b < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} TiB"


def _color_band(pct: float | None) -> str:
    """Return a CSS class name based on utilisation percentage."""
    if pct is None:
        return "warn"
    if pct >= 60:
        return "good"
    if pct >= 30:
        return "warn"
    return "error"


def _run_agent(namespace: str, cluster: str, days: int) -> dict:
    graph = _load_graph()
    initial_state = {
        "namespace": namespace,
        "cluster": cluster,
        "days_back": days,
        "bifrost_data": None,
        "grafana_data": None,
        "recommendations": None,
        "ai_suggestions": None,
        "error": None,
    }
    return graph.invoke(initial_state)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Myntra_logo.svg/512px-Myntra_logo.svg.png",
        width=120,
    )
    st.markdown("## ⚙️ Configuration")
    st.markdown(f"**Cluster** (fixed): `{SUPPORTED_CLUSTER}`")
    days_back = st.slider(
        "Historical data window (days)",
        min_value=7,
        max_value=30,
        value=30,
        step=1,
        help="How many days of Bifrost data to analyse for peak usage.",
    )
    st.divider()
    st.markdown(
        """
**How it works**
1. Enter the Kubernetes namespace
2. Agent fetches peak CPU/mem from **Bifrost**
3. Agent fetches current limits from **Grafana**
4. Optimization math: `peak × 1.30` → round up to integer core / 256 MiB
5. **Gemini 2.5 Flash** writes the recommendation report
""",
        unsafe_allow_html=False,
    )
    st.divider()
    st.caption("Powered by LangGraph · LangChain · Gemini 2.5 Flash")


# ── main header ───────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align:center;color:#f05524;'>⚡ Kubernetes Resource Optimizer</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#555;font-size:16px;'>"
    "AI-powered right-sizing for pac-mlpcluster01 services</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── session state initialisation ──────────────────────────────────────────────

for key, default in [
    ("shuttle_confirmed", False),
    ("show_shuttle_prompt", False),
    ("pending_namespace", ""),
    ("last_result", None),
    ("analysed_namespace", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── input form ────────────────────────────────────────────────────────────────

col_input, col_btn = st.columns([4, 1])
with col_input:
    namespace_input = st.text_input(
        "Service Namespace",
        placeholder="e.g. odmlpaistylist",
        help=(
            "Kubernetes namespace on pac-mlpcluster01. "
            "Must start with 'odmlp' or be deployed via Shuttle."
        ),
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Analyze", type="primary")

# ── validation & shuttle gate ─────────────────────────────────────────────────

if analyze_clicked and namespace_input:
    ns = namespace_input.strip()
    valid, msg = _validate_namespace(ns)

    if valid is False:
        st.error(msg)
    elif valid is True:
        # odmlp* → proceed directly
        st.session_state.shuttle_confirmed = False
        st.session_state.show_shuttle_prompt = False
        st.session_state.pending_namespace = ns
        st.session_state.trigger_run = True
    else:
        # Non-odmlp → ask shuttle confirmation
        st.session_state.shuttle_confirmed = False
        st.session_state.show_shuttle_prompt = True
        st.session_state.pending_namespace = ns
        st.session_state.trigger_run = False

elif analyze_clicked and not namespace_input:
    st.warning("Please enter a namespace before clicking Analyze.")

# Initialise trigger flag if missing
if "trigger_run" not in st.session_state:
    st.session_state.trigger_run = False

# ── shuttle confirmation dialog ───────────────────────────────────────────────

if st.session_state.show_shuttle_prompt and not st.session_state.shuttle_confirmed:
    pns = st.session_state.pending_namespace
    st.warning(
        f"⚠️  The namespace **`{pns}`** does not start with `odmlp`.  \n"
        f"This agent only supports services on **pac-mlpcluster01** deployed via **Shuttle** "
        f"or with the `odmlp` prefix."
    )
    st.info("Is this service deployed via **Shuttle**?")

    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        if st.button("✅ Yes, via Shuttle", use_container_width=True):
            st.session_state.shuttle_confirmed = True
            st.session_state.show_shuttle_prompt = False
            st.session_state.trigger_run = True
            st.rerun()
    with c2:
        if st.button("❌ No", use_container_width=True):
            st.session_state.show_shuttle_prompt = False
            st.session_state.trigger_run = False
            st.error(
                "Only namespaces starting with `odmlp` or services deployed via Shuttle "
                "are supported. Please verify with your team."
            )

# ── agent execution ───────────────────────────────────────────────────────────

if st.session_state.trigger_run:
    target_ns = st.session_state.pending_namespace
    st.session_state.trigger_run = False  # reset so it doesn't re-fire on rerun

    st.markdown(f"---\n### 🔄 Analysing `{target_ns}` on `{SUPPORTED_CLUSTER}`")

    # Progress steps shown sequentially via status
    with st.status("Running resource optimization agent…", expanded=True) as status_box:
        st.write("📡 **Step 1/4** — Fetching historical usage from Bifrost…")
        st.write("📊 **Step 2/4** — Fetching current resource limits from Grafana…")
        st.write("🧮 **Step 3/4** — Computing optimal allocations…")
        st.write("🤖 **Step 4/4** — Generating AI analysis with Gemini 2.5 Flash…")

        try:
            result = _run_agent(target_ns, SUPPORTED_CLUSTER, days_back)
            st.session_state.last_result = result
            st.session_state.analysed_namespace = target_ns
            status_box.update(label="✅ Analysis complete!", state="complete", expanded=False)
        except Exception as exc:
            status_box.update(label=f"❌ Agent failed: {exc}", state="error")
            st.error(f"Agent raised an exception: {exc}")
            st.stop()

# ── results rendering ─────────────────────────────────────────────────────────

result = st.session_state.get("last_result")
target_ns = st.session_state.get("analysed_namespace", "")

if result is None:
    st.markdown(
        "<div style='text-align:center;color:#aaa;margin-top:80px;font-size:18px;'>"
        "Enter a namespace above and click <b>Analyze</b> to begin.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── error display ─────────────────────────────────────────────────────────────

if result.get("error"):
    st.error(f"❌  **{result['error']}**")
    st.stop()

st.divider()
st.markdown(f"## 📋 Results for `{target_ns}`")

bifrost = result.get("bifrost_data") or {}
grafana = result.get("grafana_data") or {}
recs = result.get("recommendations") or {}
cpu_rec = recs.get("cpu") or {}
mem_rec = recs.get("memory") or {}
savings = recs.get("savings") or {}

# ── top KPI strip ──────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)

with k1:
    peak_cpu = bifrost.get("max_cpu_cores")
    st.metric("Peak CPU Usage", f"{round(peak_cpu, 3)} cores" if peak_cpu else "N/A")

with k2:
    opt_cores = cpu_rec.get("optimal_cores")
    cur_cores = savings.get("cpu_current_cores")
    delta_cores = f"-{savings.get('cpu_saved_cores', 0)} cores" if savings.get("cpu_saved_cores") else None
    st.metric("Recommended CPU", f"{opt_cores} core(s)" if opt_cores else "N/A", delta=delta_cores, delta_color="inverse")

with k3:
    peak_mem_bytes = bifrost.get("max_mem_bytes")
    st.metric("Peak Memory Usage", _bytes_to_human(peak_mem_bytes))

with k4:
    opt_gib = mem_rec.get("optimal_gib")
    mem_saved = savings.get("mem_saved_gib")
    delta_mem = f"-{mem_saved} GiB" if mem_saved else None
    st.metric("Recommended Memory", f"{opt_gib} GiB" if opt_gib else "N/A", delta=delta_mem, delta_color="inverse")

st.divider()

# ── split layout: left = numbers, right = AI report ──────────────────────────

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    # ── Current configuration ─────────────────────────────────────────────
    st.markdown("### 🔧 Current Configuration")
    current_data = {
        "Metric": ["CPU Limit", "CPU Request", "Memory Limit", "Memory Request"],
        "Value": [
            f"{round(grafana.get('cpu_limit_cores', 0) or 0, 2)} cores",
            f"{round(grafana.get('cpu_request_cores', 0) or 0, 2)} cores",
            _bytes_to_human(grafana.get("mem_limit_bytes")),
            _bytes_to_human(grafana.get("mem_request_bytes")),
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(current_data), use_container_width=True, hide_index=True)

    # ── Recommended configuration ─────────────────────────────────────────
    st.markdown("### ✅ Recommended Configuration")

    cpu_util = cpu_rec.get("utilisation_pct")
    cpu_card_class = _color_band(cpu_util)

    st.markdown(
        f"""
<div class="rec-card {cpu_card_class}">
  <b>CPU</b><br>
  Optimal limit : <b>{cpu_rec.get('optimal_cores', 'N/A')} core(s)</b>
  &nbsp;|&nbsp; Buffer applied: {cpu_rec.get('buffer_applied_pct', 30)}%
  &nbsp;|&nbsp; Peak usage: {cpu_rec.get('max_usage_cores', 'N/A')} cores
  &nbsp;|&nbsp; Expected utilisation: <b>{cpu_util}%</b>
  {"&nbsp;✅ ≥60%" if cpu_rec.get('meets_60pct_threshold') else "&nbsp;⚠️ below 60% (integer minimum)"}
</div>
""",
        unsafe_allow_html=True,
    )

    mem_util = mem_rec.get("utilisation_pct")
    mem_card_class = _color_band(mem_util)

    st.markdown(
        f"""
<div class="rec-card {mem_card_class}">
  <b>Memory</b><br>
  Optimal limit : <b>{mem_rec.get('optimal_gib', 'N/A')} GiB ({mem_rec.get('optimal_mib', 'N/A')} MiB)</b>
  &nbsp;|&nbsp; Buffer: {mem_rec.get('buffer_applied_pct', 30)}%
  &nbsp;|&nbsp; Peak usage: {mem_rec.get('max_usage_gib', 'N/A')} GiB
  &nbsp;|&nbsp; Expected utilisation: <b>{mem_util}%</b>
  {"&nbsp;✅ ≥60%" if mem_rec.get('meets_60pct_threshold') else "&nbsp;⚠️ below 60% (256 MiB minimum)"}
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Savings summary ───────────────────────────────────────────────────
    if savings.get("cpu_saved_cores") is not None:
        st.markdown("### 💰 Estimated Savings")
        sav_data = {
            "Resource": ["CPU", "Memory"],
            "Current": [
                f"{savings.get('cpu_current_cores', 'N/A')} cores",
                f"{savings.get('mem_current_gib', 'N/A')} GiB",
            ],
            "Recommended": [
                f"{savings.get('cpu_optimal_cores', 'N/A')} cores",
                f"{savings.get('mem_optimal_gib', 'N/A')} GiB",
            ],
            "Saved": [
                f"{savings.get('cpu_saved_cores', 'N/A')} cores  ({savings.get('cpu_reduction_pct', 'N/A')}%)",
                f"{savings.get('mem_saved_gib', 'N/A')} GiB  ({savings.get('mem_reduction_pct', 'N/A')}%)",
            ],
        }
        st.dataframe(pd.DataFrame(sav_data), use_container_width=True, hide_index=True)

    # ── Bifrost data window ───────────────────────────────────────────────
    with st.expander("📅 Bifrost Data Details"):
        st.json(
            {
                "namespace": bifrost.get("namespace"),
                "cluster": bifrost.get("cluster"),
                "data_days": bifrost.get("data_days"),
                "date_range": bifrost.get("date_range"),
                "max_cpu_cores": bifrost.get("max_cpu_cores"),
                "avg_cpu_cores": bifrost.get("avg_cpu_cores"),
                "max_mem_bytes": bifrost.get("max_mem_bytes"),
                "avg_mem_bytes": bifrost.get("avg_mem_bytes"),
                "avg_pods": bifrost.get("avg_pods"),
                "max_pods": bifrost.get("max_pods"),
            }
        )

with right_col:
    st.markdown("### 🤖 AI Analysis (Gemini 2.5 Flash)")

    ai_report = result.get("ai_suggestions")
    if ai_report:
        # Check if it's an error response
        if ai_report.startswith("["):
            st.warning("AI report generation encountered an issue:")
            st.code(ai_report)
        else:
            st.markdown(ai_report)
    else:
        st.info("No AI report generated. Check for errors above.")

# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<p style='text-align:center;font-size:13px;color:#aaa;'>"
    "Resource Optimizer Agent · pac-mlpcluster01 · "
    "MLP Platform · Internal use only</p>",
    unsafe_allow_html=True,
)
