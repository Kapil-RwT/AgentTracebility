"""
LangGraph StateGraph definition for the resource-optimizer pipeline.

Graph topology
──────────────
                      ┌──────────────────────┐
         START ──────▶│  fetch_bifrost_data  │
                      └──────────┬───────────┘
                                 │  error? ──────────────────────▶ END
                                 ▼
                      ┌──────────────────────┐
                      │  fetch_grafana_data  │  (non-fatal; continues on fail)
                      └──────────┬───────────┘
                                 ▼
                      ┌──────────────────────────┐
                      │  compute_recommendations  │
                      └──────────┬───────────────┘
                                 │  error? ──────────────────────▶ END
                                 ▼
                      ┌──────────────────────┐
                      │  generate_ai_report  │
                      └──────────┬───────────┘
                                 ▼
                                END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.state import OptimizationState
from agent.nodes import (
    compute_recommendations,
    fetch_bifrost_data,
    fetch_grafana_data,
    generate_ai_report,
)


# ── routing helpers ──────────────────────────────────────────────────────────

def _route_after_bifrost(state: OptimizationState) -> str:
    """Skip to END if bifrost failed; otherwise proceed normally."""
    if state.get("error"):
        return "end"
    return "fetch_grafana"


def _route_after_compute(state: OptimizationState) -> str:
    if state.get("error"):
        return "end"
    return "generate_report"


# ── graph builder ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    workflow = StateGraph(OptimizationState)

    # Register nodes
    workflow.add_node("fetch_bifrost", fetch_bifrost_data)
    workflow.add_node("fetch_grafana", fetch_grafana_data)
    workflow.add_node("compute", compute_recommendations)
    workflow.add_node("generate_report", generate_ai_report)

    # Entry point
    workflow.set_entry_point("fetch_bifrost")

    # Edges with conditional error routing
    workflow.add_conditional_edges(
        "fetch_bifrost",
        _route_after_bifrost,
        {"end": END, "fetch_grafana": "fetch_grafana"},
    )
    workflow.add_edge("fetch_grafana", "compute")
    workflow.add_conditional_edges(
        "compute",
        _route_after_compute,
        {"end": END, "generate_report": "generate_report"},
    )
    workflow.add_edge("generate_report", END)

    return workflow.compile()


# Compiled graph singleton – import this in app.py
optimization_graph = build_graph()
