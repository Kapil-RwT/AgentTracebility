"""
K8s Debug Agent — Streamlit UI
Run with: streamlit run src/interfaces/ui.py
"""

import asyncio
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

_IST = timezone(timedelta(hours=5, minutes=30))
from urllib.parse import parse_qs, urlparse

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── context builder ───────────────────────────────────────────────────────────

def _build_context(service: str, time_from: int, time_to: int) -> dict:
    """
    Build the investigation context from the service name, time range, and the
    GRAFANA_DASHBOARD_BASE_URL env var.

    All `var-*` params from the base URL are parsed into grafana_vars so that
    Grafana panel expressions (e.g. `$cluster_small`, `$region`) are substituted
    correctly.  "All" values and Grafana template expressions (starting with `$`)
    are intentionally skipped — they carry no useful filter information.
    """
    base = os.getenv("GRAFANA_DASHBOARD_BASE_URL", "")
    grafana_vars: dict[str, str] = {}

    if base:
        params = parse_qs(urlparse(base).query)
        for key, val_list in params.items():
            if not key.startswith("var-") or not val_list:
                continue
            val = val_list[0]
            # Skip: empty, Grafana template expressions ($__...), multi-select "All"
            if not val or val.startswith("$") or val == "All":
                continue
            grafana_vars[key[4:]] = val  # strip "var-" prefix

    # Service-specific vars always override the base URL values
    grafana_vars["namespace"] = service
    grafana_vars["backend"]   = service

    cluster = grafana_vars.get("cluster", os.getenv("DEFAULT_CLUSTER", "unknown"))

    from_label = datetime.fromtimestamp(time_from / 1000, tz=_IST).strftime("%Y-%m-%d %H:%M IST")
    to_label   = datetime.fromtimestamp(time_to   / 1000, tz=_IST).strftime("%Y-%m-%d %H:%M IST")

    return {
        "service":      service,
        "namespace":    service,
        "grafana_vars": grafana_vars,
        "time_from":    time_from,
        "time_to":      time_to,
        "from_label":   from_label,
        "to_label":     to_label,
        "cluster":      cluster,
        "region":       grafana_vars.get("region", ""),
    }


# ── agent runner ──────────────────────────────────────────────────────────────

def run_diagnosis(ctx: dict, symptom: str) -> dict:
    from src.agent import create_agent_from_env
    agent = create_agent_from_env()
    return asyncio.run(
        agent.investigate(
            service=ctx["service"],
            namespace=ctx["namespace"],
            time_from=ctx["time_from"],
            time_to=ctx["time_to"],
            grafana_vars=ctx["grafana_vars"],
            symptom=symptom,
            verbose=False,
            return_raw=True,
        )
    )


# ── panel ID → readable name ──────────────────────────────────────────────────

_PANEL_NAMES = {
    "86": "Inbound Latency (pod)",
    "69": "Outbound Latency",
    "42": "Ingress Latency",
    "98": "Disk IOPS",
    "88": "Inbound Latency (deployment)",
    "28": "DNS Latency / Inbound (CoreDNS)",
    "72": "CPU Throttle",
    "62": "CPU Request",
}

def _panel_label(url: str) -> str:
    m = re.search(r"viewPanel=(\d+)", url)
    if m:
        return _PANEL_NAMES.get(m.group(1), f"Panel {m.group(1)}")
    return "Grafana Panel"


# ── CSS ───────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
  /* tighten top padding */
  .block-container { padding-top: 1.5rem; }

  /* anomaly cards */
  .anomaly-card {
    background: rgba(239,68,68,0.08);
    border-left: 4px solid #ef4444;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 6px;
    font-size: 0.88rem;
    line-height: 1.5;
  }
  .anomaly-card.rpm  { border-color: #f97316; background: rgba(249,115,22,0.08); }
  .anomaly-card.iops { border-color: #eab308; background: rgba(234,179,8,0.08); }
  .anomaly-card.cpu  { border-color: #a78bfa; background: rgba(167,139,250,0.08); }

  /* healthy pills */
  .healthy-pill {
    display: inline-block;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.35);
    color: #4ade80;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.82rem;
    margin: 3px 4px;
  }

  /* narrative box */
  .narrative-box {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.25);
    padding: 16px 20px;
    border-radius: 8px;
    line-height: 1.7;
    font-size: 0.95rem;
  }

  /* root cause box */
  .cause-box {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.4);
    padding: 16px 20px;
    border-radius: 8px;
    line-height: 1.7;
    font-size: 0.95rem;
  }

  /* escalation banner */
  .escalate-yes {
    background: rgba(239,68,68,0.12);
    border: 2px solid #ef4444;
    padding: 16px 20px;
    border-radius: 8px;
  }
  .escalate-no {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.3);
    padding: 14px 20px;
    border-radius: 8px;
  }

  /* step card */
  .step-prose {
    font-size: 0.93rem;
    line-height: 1.6;
    margin-bottom: 6px;
  }

  /* context bar */
  .ctx-bar {
    background: #1e1e2e;
    border-radius: 8px;
    padding: 14px 20px;
    margin-bottom: 8px;
    border-left: 4px solid #6366f1;
    font-size: 0.9rem;
  }

  /* tool status row */
  .tool-row {
    font-size: 0.83rem;
    padding: 2px 0;
    color: #94a3b8;
  }
  .tool-ok  { color: #4ade80; }
  .tool-err { color: #f87171; }
</style>
"""


# ── render helpers ────────────────────────────────────────────────────────────

def _anomaly_class(a: str) -> str:
    if "RPM" in a or "IMBALANCE" in a:
        return "anomaly-card rpm"
    if "IOPS" in a:
        return "anomaly-card iops"
    if "CPU" in a:
        return "anomaly-card cpu"
    return "anomaly-card"


def _render_step(i: int, step: str):
    lines = step.split("\n")
    # Split into prose lines and kubectl lines
    prose, kubectl = [], []
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("kubectl") or (stripped.startswith("--") and kubectl):
            kubectl.append(stripped)
        else:
            prose.append(stripped)

    prose_text = " ".join(l for l in prose if l)
    kubectl_text = "\n".join(kubectl)

    st.markdown(f"**Step {i}**")
    if prose_text:
        st.markdown(f'<div class="step-prose">{prose_text}</div>', unsafe_allow_html=True)
    if kubectl_text:
        st.code(kubectl_text, language="bash")


def render_results(result: dict):
    # ── Context bar ───────────────────────────────────────────────────────────
    region_part = f"&nbsp; 🌐 {result.get('cluster', '')} / {result.get('from_iso', '')[:10]}" if result.get("cluster") else ""
    st.markdown(
        f'<div class="ctx-bar">'
        f'📍 <strong>{result["service"]}</strong> &nbsp;/&nbsp; {result["namespace"]}'
        f'&nbsp;&nbsp;&nbsp; 🖥️ {result["cluster"]}'
        f'&nbsp;&nbsp;&nbsp; 🕐 {result["from_iso"]} → {result["to_iso"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Metric collection summary ─────────────────────────────────────────────
    with st.expander("📡 Metrics collected", expanded=False):
        for t in result.get("tool_summary", []):
            hop    = f"({t['hop']})" if t["hop"] else ""
            icon   = "✓" if t["ok"] else "✗"
            cls    = "tool-ok" if t["ok"] else "tool-err"
            st.markdown(
                f'<div class="tool-row"><span class="{cls}">{icon}</span> '
                f'<code>{t["name"]}{hop}</code> — {t["one_line"]}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Two-column layout: anomalies | healthy ────────────────────────────────
    col_a, col_h = st.columns([3, 2])

    with col_a:
        st.markdown("### ⚠️ What's Wrong")
        if result["anomalies"]:
            for a in result["anomalies"]:
                st.markdown(f'<div class="{_anomaly_class(a)}">{a}</div>', unsafe_allow_html=True)
        else:
            st.success("No anomalies detected — service is healthy.")

    with col_h:
        st.markdown("### ✅ Healthy Signals")
        if result["healthy"]:
            pills = "".join(
                f'<span class="healthy-pill">{h}</span>' for h in result["healthy"]
            )
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("No healthy signal data available.")

    st.divider()

    # ── Narrative ─────────────────────────────────────────────────────────────
    st.markdown("### 📋 What Is Happening")
    st.markdown(f'<div class="narrative-box">{result["narrative"]}</div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Most Likely Cause")
    st.markdown(f'<div class="cause-box">{result["root_cause"]}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Steps ─────────────────────────────────────────────────────────────────
    st.markdown("### 🔧 What To Do")
    for i, step in enumerate(result["steps"], 1):
        _render_step(i, step)

    # ── Secondary steps ───────────────────────────────────────────────────────
    if result.get("secondary_steps"):
        with st.expander("📌 Also Address", expanded=True):
            for step in result["secondary_steps"]:
                st.markdown(f'<div class="anomaly-card">{step}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Escalation ────────────────────────────────────────────────────────────
    if result["escalate"]:
        st.markdown(
            f'<div class="escalate-yes">'
            f'<h4 style="color:#ef4444;margin:0 0 8px 0;">🚨 ESCALATE TO SRE</h4>'
            f'<p style="margin:0;font-size:0.95rem;">{result["escalate_reason"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="escalate-no">'
            '<strong>✅ No SRE escalation required</strong> — DS team can resolve this directly.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Grafana links ─────────────────────────────────────────────────────────
    links = result.get("grafana_links", [])
    if links:
        st.markdown("### 📊 Grafana Links")
        cols = st.columns(min(len(links), 4))
        for i, (link, col) in enumerate(zip(links, cols * 10)):
            with col:
                st.link_button(
                    f"📈 {_panel_label(link)}",
                    link,
                    use_container_width=True,
                )


# ── time helpers ─────────────────────────────────────────────────────────────

def _parse_hhmm(s: str, base_dt: datetime) -> datetime:
    """Parse an HH:MM string and return a datetime on base_dt's date (IST)."""
    m = re.match(r"^(\d{1,2}):(\d{2})$", s.strip())
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if 0 <= h < 24 and 0 <= mn < 60:
            return base_dt.replace(hour=h, minute=mn, second=0, microsecond=0)
    return base_dt


# ── main app ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="K8s Debug Agent",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("## 🔍 K8s Debug Agent")
    st.caption("Diagnose MLP Kubernetes service health issues from a Grafana dashboard URL.")

    st.divider()

    # ── Input form ────────────────────────────────────────────────────────────
    col_svc, col_sym = st.columns([1, 2])
    with col_svc:
        service = st.text_input(
            "Service / Namespace",
            placeholder="e.g. odmlpcore",
            help="Deployment name (same as namespace for MLP services).",
        )
    with col_sym:
        symptom = st.text_input(
            "Symptom (optional)",
            placeholder="e.g. p99 latency spiked to 2s starting at 10:08 IST",
            help="Briefly describe what you observed. Helps focus the diagnosis.",
        )

    # Time range — preset selectbox + separate custom range toggle
    TIME_PRESETS = {
        "Last 30 min":   30 * 60,
        "Last 1 hour":   1  * 3600,
        "Last 3 hours":  3  * 3600,
        "Last 6 hours":  6  * 3600,
        "Last 12 hours": 12 * 3600,
        "Last 24 hours": 24 * 3600,
        "Last 2 days":   2  * 86400,
        "Last 7 days":   7  * 86400,
        "Last 30 days":  30 * 86400,
        "Last 90 days":  90 * 86400,
        "Last 6 months": 183 * 86400,
        "Last 1 year":   365 * 86400,
        "Last 2 years":  730 * 86400,
        "Last 5 years":  1825 * 86400,
        "Yesterday":     -1,
    }

    col_tr, col_custom = st.columns([3, 1])
    with col_tr:
        preset_label = st.selectbox("Time Range", list(TIME_PRESETS.keys()), index=1)
    with col_custom:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)  # align with selectbox
        use_custom = st.toggle("Custom range", value=False)

    preset_secs = TIME_PRESETS[preset_label]

    time_from: int | None = None
    time_to:   int | None = None

    if use_custom:
        # Seed session_state defaults only on first entry so reruns don't wipe user input.
        ss = st.session_state
        if "cr_from_date" not in ss:
            _now = datetime.now(_IST).replace(second=0, microsecond=0)
            ss["cr_from_date"] = (_now - timedelta(minutes=5)).date()
            ss["cr_from_time"] = (_now - timedelta(minutes=5)).strftime("%H:%M")
            ss["cr_to_date"]   = _now.date()
            ss["cr_to_time"]   = _now.strftime("%H:%M")

        col_fd, col_ft, col_td, col_tt = st.columns(4)
        with col_fd:
            from_date = st.date_input("From date (IST)", key="cr_from_date")
        with col_ft:
            from_time_str = st.text_input(
                "From time (IST, HH:MM)", key="cr_from_time",
                help="Type a time in HH:MM format, e.g. 10:08",
            )
        with col_td:
            to_date = st.date_input("To date (IST)", key="cr_to_date")
        with col_tt:
            to_time_str = st.text_input(
                "To time (IST, HH:MM)", key="cr_to_time",
                help="Type a time in HH:MM format, e.g. 10:13",
            )

        from_dt_base = datetime(from_date.year, from_date.month, from_date.day, tzinfo=_IST)
        to_dt_base   = datetime(to_date.year,   to_date.month,   to_date.day,   tzinfo=_IST)
        from_dt = _parse_hhmm(from_time_str, from_dt_base)
        to_dt   = _parse_hhmm(to_time_str,   to_dt_base)

        time_from = int(from_dt.timestamp() * 1000)
        time_to   = int(to_dt.timestamp()   * 1000)
    elif preset_secs == -1:  # Yesterday
        now_ist   = datetime.now(_IST)
        yesterday = now_ist.date() - timedelta(days=1)
        time_from = int(datetime(yesterday.year, yesterday.month, yesterday.day,
                                 0, 0, 0, tzinfo=_IST).timestamp() * 1000)
        time_to   = int(datetime(yesterday.year, yesterday.month, yesterday.day,
                                 23, 59, 59, tzinfo=_IST).timestamp() * 1000)
    else:
        time_to   = int(time.time() * 1000)
        time_from = time_to - preset_secs * 1000

    ctx = None
    if service and time_from and time_to:
        if time_from >= time_to:
            st.warning("⚠️ Start time must be before end time.")
        else:
            ctx = _build_context(service.strip(), time_from, time_to)
            # For preset ranges show the label — end time is recomputed at click time.
            # For custom range show the exact IST timestamps the user entered.
            if use_custom:
                time_display = f"{ctx['from_label']} → {ctx['to_label']}"
            elif preset_secs == -1:
                time_display = f"Yesterday ({ctx['from_label'][:10]})"
            else:
                approx_from = datetime.fromtimestamp(time_from / 1000, tz=_IST).strftime("%Y-%m-%d")
                time_display = f"{preset_label} ({approx_from} → now)"
            st.markdown(
                f'<div style="font-size:0.85rem;color:#94a3b8;padding:4px 0;">'
                f'Ready: <strong>{ctx["service"]}</strong> &nbsp;·&nbsp; '
                f'cluster: {ctx["cluster"]} &nbsp;·&nbsp; '
                f'{time_display}'
                f'</div>',
                unsafe_allow_html=True,
            )

    diagnose = st.button(
        "🔍 Diagnose",
        type="primary",
        disabled=(ctx is None),
        use_container_width=False,
    )

    st.divider()

    # ── Run ───────────────────────────────────────────────────────────────────
    if diagnose and ctx:
        # Recompute the time window at the exact moment of click so preset ranges
        # always end at "right now". Custom range uses the user-entered timestamps as-is.
        if not use_custom and preset_secs and preset_secs > 0:
            _now_ms = int(time.time() * 1000)
            ctx["time_to"]    = _now_ms
            ctx["time_from"]  = _now_ms - preset_secs * 1000
            ctx["to_label"]   = datetime.fromtimestamp(_now_ms / 1000, tz=_IST).strftime("%Y-%m-%d %H:%M IST")
            ctx["from_label"] = datetime.fromtimestamp(ctx["time_from"] / 1000, tz=_IST).strftime("%Y-%m-%d %H:%M IST")

        placeholder = st.empty()

        with placeholder.container():
            with st.status("Running diagnostics…", expanded=True) as status:
                st.write(f"📡 Querying Grafana for **{ctx['service']}** ({ctx['from_label']} → {ctx['to_label']})…")
                st.write("⚙️  Running 11 metric checks in parallel…")

                try:
                    t0 = time.time()
                    result = run_diagnosis(ctx, symptom)
                    elapsed = round(time.time() - t0, 1)

                    n = len(result.get("anomalies", []))
                    st.write(f"🤖 LLM diagnosis generated — {n} anomaly(ies) found ({elapsed}s total)")
                    status.update(
                        label=f"Done — {n} anomaly(ies) found in {elapsed}s",
                        state="complete",
                        expanded=False,
                    )
                except Exception as e:
                    status.update(label="Diagnosis failed", state="error", expanded=True)
                    st.error(f"**Error:** {e}")
                    st.stop()

        placeholder.empty()
        render_results(result)


if __name__ == "__main__":
    main()
