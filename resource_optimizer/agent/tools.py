"""
LangChain tools used by the resource-optimizer agent.

Each tool is decorated with @tool so it can be bound to a LangChain
agent executor or called directly from LangGraph nodes.
"""

from __future__ import annotations

import sys
import os

# Make sure the project root is on the path regardless of how we're invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


from langchain_core.tools import tool

from clients.bifrost_client import BifrostClient
from clients.grafana_client import GrafanaClient
from optimizer.calculator import (
    calculate_optimal_cpu,
    calculate_optimal_memory,
    compute_savings,
)

# ── singletons (avoid re-creating on every tool call) ───────────────────────
_bifrost = BifrostClient()
_grafana = GrafanaClient()


# ── Tool 1: fetch usage from Bifrost ────────────────────────────────────────

@tool
def fetch_usage_from_bifrost(
    namespace: str,
    cluster: str = "pac-mlpcluster01",
    days: int = 30,
) -> dict:
    """
    Query the Bifrost data platform for historical CPU and memory usage of a
    Kubernetes namespace over the last `days` days.

    Returns peak max_cpu (cores) and max_mem (bytes) across all data points,
    plus pod-count stats and the raw row count for transparency.
    """
    try:
        df = _bifrost.get_service_usage(namespace, cluster, days)
        if df.empty:
            return {"error": f"No data found in Bifrost for namespace='{namespace}', cluster='{cluster}'"}

        def _safe_max(col):
            return float(df[col].max()) if col in df.columns else None

        def _safe_mean(col):
            return float(df[col].mean()) if col in df.columns else None

        return {
            "namespace": namespace,
            "cluster": cluster,
            "data_days": len(df),
            "date_range": f"{df['date'].min()} → {df['date'].max()}" if "date" in df.columns else "N/A",
            "max_cpu_cores": _safe_max("max_cpu"),
            "avg_cpu_cores": _safe_mean("avg_cpu"),
            "max_mem_bytes": _safe_max("max_mem"),
            "avg_mem_bytes": _safe_mean("avg_mem"),
            "max_cores_allocated": _safe_max("max_cores"),
            "avg_pods": _safe_mean("avg_pod"),
            "max_pods": _safe_max("max_pod"),
        }
    except Exception as exc:
        return {"error": f"Bifrost fetch failed: {exc}"}


# ── Tool 2: fetch current limits from Grafana ────────────────────────────────

@tool
def fetch_resource_limits_from_grafana(
    namespace: str,
    cluster: str = "pac-mlpcluster01",
) -> dict:
    """
    Query the Grafana Prometheus datasource for the current Kubernetes CPU and
    memory resource limits and requests configured for a namespace.

    Returns CPU (cores) and memory (bytes) values for both limit and request.
    """
    try:
        data = _grafana.get_resource_config(namespace, cluster)
        return {
            "namespace": namespace,
            "cluster": cluster,
            "cpu_limit_cores": data.get("cpu_limit"),
            "cpu_request_cores": data.get("cpu_request"),
            "mem_limit_bytes": data.get("mem_limit"),
            "mem_request_bytes": data.get("mem_request"),
        }
    except Exception as exc:
        return {"error": f"Grafana fetch failed: {exc}"}


# ── Tool 3: compute optimal resource allocation ──────────────────────────────

@tool
def compute_optimal_resources(
    max_cpu_cores: float,
    max_mem_bytes: float,
    current_cpu_limit_cores: float | None = None,
    current_mem_limit_bytes: float | None = None,
) -> dict:
    """
    Given peak CPU (cores) and memory (bytes) usage, compute the recommended
    resource limits applying a 30 % safety buffer.

    CPU recommendation is always an integer (minimum 1 core).
    Memory is rounded up to the nearest 256 MiB block.
    Also computes savings compared to the current limits if provided.
    """
    cpu_rec = calculate_optimal_cpu(max_cpu_cores)
    mem_rec = calculate_optimal_memory(max_mem_bytes)
    savings = compute_savings(
        current_cpu_limit_cores,
        cpu_rec["optimal_cores"],
        current_mem_limit_bytes,
        mem_rec["optimal_mib"],
    )
    return {
        "cpu": cpu_rec,
        "memory": mem_rec,
        "savings": savings,
    }
