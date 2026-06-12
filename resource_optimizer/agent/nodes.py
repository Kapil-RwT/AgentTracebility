"""
LangGraph node functions.

Each function:
  • receives the full OptimizationState
  • returns a dict of only the keys it updates  (LangGraph merges partial updates)

Graph topology
--------------
  fetch_bifrost_data
        │
  fetch_grafana_data
        │
  compute_recommendations
        │
  generate_ai_report
        │
       END

Error handling: nodes write `error` into state; the graph checks this key to
decide whether to skip downstream nodes and go directly to END.
"""

from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import gemini_llm
from agent.state import OptimizationState
from agent.tools import (
    fetch_usage_from_bifrost,
    fetch_resource_limits_from_grafana,
    compute_optimal_resources,
)


# ── Node 1: fetch historical usage from Bifrost ──────────────────────────────

def fetch_bifrost_data(state: OptimizationState) -> dict:
    """Invoke the Bifrost LangChain tool and store results in state."""
    result = fetch_usage_from_bifrost.invoke(
        {
            "namespace": state["namespace"],
            "cluster": state["cluster"],
            "days": state.get("days_back", 30),
        }
    )

    if "error" in result:
        return {"error": result["error"], "bifrost_data": None}

    return {"bifrost_data": result, "error": None}


# ── Node 2: fetch current resource limits from Grafana ───────────────────────

def fetch_grafana_data(state: OptimizationState) -> dict:
    """Invoke the Grafana LangChain tool and store results in state."""
    result = fetch_resource_limits_from_grafana.invoke(
        {
            "namespace": state["namespace"],
            "cluster": state["cluster"],
        }
    )

    if "error" in result:
        # Grafana failure is non-fatal: we can still compute recommendations
        # without current limits; we just won't show savings.
        return {"grafana_data": None}

    return {"grafana_data": result}


# ── Node 3: compute optimal resource allocation ───────────────────────────────

def compute_recommendations(state: OptimizationState) -> dict:
    """
    Use the compute LangChain tool to calculate optimal CPU (integer cores)
    and memory (256 MiB-aligned) from peak bifrost usage.
    """
    bifrost = state.get("bifrost_data") or {}
    grafana = state.get("grafana_data") or {}

    max_cpu = bifrost.get("max_cpu_cores")
    max_mem = bifrost.get("max_mem_bytes")

    if max_cpu is None:
        return {"error": "max_cpu_cores not available in Bifrost data; cannot compute recommendations."}

    # Memory might be unavailable; fall back to a minimal placeholder so the
    # rest of the pipeline still works.
    if max_mem is None:
        max_mem = 0.0

    result = compute_optimal_resources.invoke(
        {
            "max_cpu_cores": max_cpu,
            "max_mem_bytes": max_mem,
            "current_cpu_limit_cores": grafana.get("cpu_limit_cores"),
            "current_mem_limit_bytes": grafana.get("mem_limit_bytes"),
        }
    )

    return {"recommendations": result}


# ── Node 4: generate AI report via Gemini ────────────────────────────────────

def generate_ai_report(state: OptimizationState) -> dict:
    """
    Use the internal Gemini 2.5 Flash LLM (via LangChain BaseChatModel) to
    produce a rich, actionable optimization report including advice to discuss
    low traffic levels with the product team.
    """
    namespace = state["namespace"]
    bifrost = state.get("bifrost_data") or {}
    grafana = state.get("grafana_data") or {}
    recs = state.get("recommendations") or {}

    cpu_rec = recs.get("cpu") or {}
    mem_rec = recs.get("memory") or {}
    savings = recs.get("savings") or {}

    # ── helpers ──────────────────────────────────────────────────────────────
    def _fmt(v, decimals=2):
        return "N/A" if v is None else round(v, decimals)

    def _bytes_to_gib(b):
        return "N/A" if b is None else round(b / (1024 ** 3), 3)

    # ── system prompt ─────────────────────────────────────────────────────────
    system_prompt = """You are an expert Site Reliability Engineer and Kubernetes cost-optimisation specialist at Myntra (Flipkart group).
Your job is to analyse service resource utilisation metrics and produce concise, actionable recommendations.
Always be data-driven. Use markdown formatting with clear headers and bullet points.
When utilisation is very low, proactively advise the team to also engage with their product/business stakeholders."""

    # ── user prompt ───────────────────────────────────────────────────────────
    user_prompt = f"""
## Service: `{namespace}`  |  Cluster: `{state['cluster']}`

### Current Resource Configuration (from Grafana / kube-state-metrics)
| Resource | Limit | Request |
|----------|-------|---------|
| CPU      | {_fmt(grafana.get('cpu_limit_cores'))} cores | {_fmt(grafana.get('cpu_request_cores'))} cores |
| Memory   | {_bytes_to_gib(grafana.get('mem_limit_bytes'))} GiB | {_bytes_to_gib(grafana.get('mem_request_bytes'))} GiB |

### Actual Peak Usage — Last {state.get('days_back', 30)} days (from Bifrost / sre_cost_analysis)
| Metric | Value |
|--------|-------|
| Peak CPU usage | {_fmt(bifrost.get('max_cpu_cores'), 3)} cores |
| Avg CPU usage  | {_fmt(bifrost.get('avg_cpu_cores'), 3)} cores |
| Peak Memory    | {_bytes_to_gib(bifrost.get('max_mem_bytes'))} GiB |
| Avg Memory     | {_bytes_to_gib(bifrost.get('avg_mem_bytes'))} GiB |
| Avg pod count  | {_fmt(bifrost.get('avg_pods'))} |
| Max pod count  | {_fmt(bifrost.get('max_pods'))} |
| Data points    | {bifrost.get('data_days', 'N/A')} days |
| Date range     | {bifrost.get('date_range', 'N/A')} |

### Computed Recommendations (30 % buffer, 60 % utilisation target)
| Resource | Current Limit | Recommended | Savings | Utilisation |
|----------|--------------|-------------|---------|-------------|
| CPU (cores, integer) | {_fmt(savings.get('cpu_current_cores'))} | **{cpu_rec.get('optimal_cores', 'N/A')}** | {_fmt(savings.get('cpu_saved_cores'))} cores ({_fmt(savings.get('cpu_reduction_pct'))} %) | {cpu_rec.get('utilisation_pct', 'N/A')} % |
| Memory (GiB) | {savings.get('mem_current_gib', 'N/A')} | **{mem_rec.get('optimal_gib', 'N/A')}** | {savings.get('mem_saved_gib', 'N/A')} GiB ({savings.get('mem_reduction_pct', 'N/A')} %) | {mem_rec.get('utilisation_pct', 'N/A')} % |

### Task for you

Please produce a concise, actionable report with the following sections:

1. **Executive Summary** — what these numbers mean in plain English.
2. **CPU Recommendation** — why {cpu_rec.get('optimal_cores', 'N/A')} core(s) is optimal; note the 30 % buffer and integer constraint.
3. **Memory Recommendation** — why {mem_rec.get('optimal_gib', 'N/A')} GiB is optimal; note 256 MiB rounding.
4. **Traffic & Business Insight** — the service is running at only {cpu_rec.get('utilisation_pct', 'N/A')} % CPU utilisation. Clearly advise the team to **discuss with their product manager / business stakeholders** why traffic is so low:
   - Is this expected (low-traffic service)?
   - Is there a growth plan that would justify the current allocation?
   - If traffic is unlikely to grow, downsizing resources frees cluster capacity for other teams.
5. **Risk Assessment** — any risks when reducing limits (OOM, CPU throttle spikes during peak events).
6. **Action Items** — a numbered checklist of what the team should do next (update Helm values, test in staging, monitor after rollout, etc.).

Keep the tone professional but friendly. Be specific with numbers.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = gemini_llm.invoke(messages)
        report = response.content
    except Exception as exc:
        report = f"[AI report generation failed: {exc}]"

    return {"ai_suggestions": report}
