"""LangGraph TypedDict state shared across all nodes."""

from typing import Any, TypedDict


class OptimizationState(TypedDict):
    # ── inputs ───────────────────────────────────────────────────────────────
    namespace: str
    cluster: str
    days_back: int

    # ── data fetched by nodes ─────────────────────────────────────────────────
    bifrost_data: dict[str, Any] | None
    grafana_data: dict[str, Any] | None

    # ── computed recommendations ──────────────────────────────────────────────
    recommendations: dict[str, Any] | None

    # ── AI-generated report ───────────────────────────────────────────────────
    ai_suggestions: str | None

    # ── error propagation ─────────────────────────────────────────────────────
    error: str | None
