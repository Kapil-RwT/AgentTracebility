"""
Grafana HTTP API client.

Handles dashboard metadata fetching and Prometheus datasource queries.
All methods are async and safe to call concurrently.

Key design: panel PromQL queries are fetched DIRECTLY from the dashboard JSON
(get_panel_expr) rather than being hardcoded. This ensures every service is
filtered correctly via Grafana's own template variables ($namespace, $backend, etc.).
"""

import os
import re
import httpx
from typing import Any


def _grafana_time_vars() -> dict[str, str]:
    """
    Replacements for Grafana built-in time variables in PromQL expressions.
    The real Grafana computes these dynamically from the query window; here we
    substitute a fixed window appropriate for a 5-minute rolling aggregation.
    Override via GRAFANA_RATE_INTERVAL env var (e.g. "1m", "5m", "15m").
    """
    interval = os.getenv("GRAFANA_RATE_INTERVAL", "5m")
    interval_ms = str(int(interval[:-1]) * (60000 if interval.endswith("m") else 3600000))
    window_s = os.getenv("GRAFANA_WINDOW_S", "3600")
    return {
        "__rate_interval": interval,
        "__interval": interval,
        "__interval_ms": interval_ms,
        "__auto_interval_interval": interval,
        "interval": interval,
        "__range": f"{window_s}s",
        "__range_s": window_s,
        "__range_ms": str(int(window_s) * 1000),
    }


class GrafanaClient:
    def __init__(self, base_url: str, token: str, datasource_uid: str = ""):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self.datasource_uid = datasource_uid
        self._dashboard_cache: dict[str, Any] = {}
        # Per-investigation Grafana template variables (cluster, region, ingress_class …).
        # Set by the agent before running tools so every query uses the right context.
        # Tools merge their own vars (namespace, backend) on top of these.
        self.vars: dict[str, str] = {}

    @property
    def _timeout(self) -> int:
        return int(os.getenv("GRAFANA_TIMEOUT_S", "30"))

    async def get_dashboard(self, uid: str) -> dict:
        """Fetch and cache a dashboard definition (includes all panel queries)."""
        if uid in self._dashboard_cache:
            return self._dashboard_cache[uid]
        async with httpx.AsyncClient(headers=self.headers, timeout=self._timeout) as client:
            r = await client.get(f"{self.base_url}/api/dashboards/uid/{uid}")
            r.raise_for_status()
            data = r.json()
        self._dashboard_cache[uid] = data
        return data

    def _find_panel(self, panels: list[dict], panel_id: int) -> dict | None:
        """Recursively search panels and row-collapsed panels for a given panel ID."""
        for panel in panels:
            if panel.get("id") == panel_id:
                return panel
            # Panels can be nested inside rows
            if panel.get("type") == "row":
                found = self._find_panel(panel.get("panels", []), panel_id)
                if found:
                    return found
        return None

    async def get_panel_expr(
        self,
        dashboard_uid: str,
        panel_id: int | None,
        variables: dict[str, str],
        target_index: int = 0,
        prefer_p99: bool = False,
    ) -> str | None:
        """
        Extract the PromQL expression from a specific panel and substitute
        all Grafana template variables with real values.

        prefer_p99: when True, scan all targets and return the one whose
            expression contains '0.99' or 'p99' — used for latency panels
            that have separate p50/p90/p95/p99 targets.
        """
        if panel_id is None:
            return None
        dashboard = await self.get_dashboard(dashboard_uid)
        all_panels = dashboard.get("dashboard", {}).get("panels", [])
        panel = self._find_panel(all_panels, panel_id)
        if panel is None:
            return None

        targets = [t for t in panel.get("targets", []) if t.get("expr", "").strip()]
        if not targets:
            return None

        if prefer_p99:
            # Find the target whose expression contains a 0.99 quantile
            for t in targets:
                expr_lower = t["expr"].lower()
                if "0.99" in expr_lower or ", 99," in expr_lower:
                    expr = t["expr"]
                    break
            else:
                idx = min(target_index, len(targets) - 1)
                expr = targets[idx]["expr"]
        else:
            idx = min(target_index, len(targets) - 1)
            expr = targets[idx]["expr"]

        # Merge: client-level infra vars first, then per-tool vars (service/namespace) override
        merged_vars = {**self.vars, **variables}
        return self.apply_variables(expr, merged_vars)

    def apply_variables(self, expr: str, variables: dict[str, str]) -> str:
        """
        Substitute Grafana template variables in a PromQL expression.

        Handles all three Grafana syntaxes:
          $var   ${var}   ${var:pipe}   ${var:regex}
        Also replaces Grafana built-in time variables (__rate_interval etc.)
        """
        merged = {**_grafana_time_vars(), **variables}

        def _lookup(key: str, fallback: str) -> str:
            # Try exact match, then lowercase, then capitalize — handles $Namespace vs $namespace
            return (
                merged.get(key)
                if merged.get(key) is not None else
                merged.get(key.lower())
                if merged.get(key.lower()) is not None else
                merged.get(key.capitalize(), fallback)
            )

        # Replace ${var:modifier} and ${var} forms first (longer match wins)
        expr = re.sub(
            r"\$\{(\w+)(?::[^}]*)?\}",
            lambda m: _lookup(m.group(1), m.group(0)),
            expr,
        )
        # Replace plain $var form
        expr = re.sub(
            r"\$(\w+)",
            lambda m: _lookup(m.group(1), m.group(0)),
            expr,
        )
        return expr

    async def query_range(
        self,
        expr: str,
        time_from: int,
        time_to: int,
        step_ms: int | None = None,
        datasource_uid: str | None = None,
    ) -> list[dict]:
        """
        Execute a Prometheus range query via the Grafana datasource proxy.

        step_ms: resolution in milliseconds. Defaults to GRAFANA_STEP_MS env var (default 60000).
        datasource_uid: override the default datasource (used for panels with their own source).

        Returns the raw list of data frames from Grafana's /api/ds/query response.
        """
        uid = datasource_uid or self.datasource_uid
        if not uid:
            raise ValueError("GRAFANA_DATASOURCE_UID must be set to run queries")

        resolved_step_ms = step_ms or int(os.getenv("GRAFANA_STEP_MS", "60000"))
        max_data_points = int(os.getenv("GRAFANA_MAX_DATAPOINTS", "720"))

        payload = {
            "queries": [
                {
                    "refId": "A",
                    "expr": expr,
                    "datasource": {"uid": uid, "type": "prometheus"},
                    "intervalMs": resolved_step_ms,
                    "maxDataPoints": max_data_points,
                    "range": True,
                    "instant": False,
                }
            ],
            "from": str(time_from),
            "to": str(time_to),
        }
        async with httpx.AsyncClient(headers=self.headers, timeout=self._timeout) as client:
            r = await client.post(f"{self.base_url}/api/ds/query", json=payload)
            r.raise_for_status()
        return r.json().get("results", {}).get("A", {}).get("frames", [])

    def iter_panels(self, dashboard_uid: str) -> list[tuple[str, dict]]:
        """
        Return (section_title, panel_dict) pairs from a cached dashboard.
        Must call get_dashboard first. Flattens row-nested panels.
        """
        dash = self._dashboard_cache.get(dashboard_uid, {})
        raw_panels = dash.get("dashboard", {}).get("panels", [])
        out: list[tuple[str, dict]] = []
        section = "Overview"
        for p in raw_panels:
            if p.get("type") == "row":
                section = p.get("title", "Overview").strip()
                for nested in p.get("panels", []):
                    out.append((section, nested))
            else:
                out.append((section, p))
        return out

    def panel_datasource_uid(self, panel: dict) -> str:
        """Return the datasource UID for a panel, falling back to the default."""
        ds = panel.get("datasource")
        if isinstance(ds, dict):
            uid = ds.get("uid", "")
            if uid and not uid.startswith("$"):
                return uid
        elif isinstance(ds, str) and ds and not ds.startswith("$"):
            return ds
        return self.datasource_uid

    def panel_url(
        self, dashboard_uid: str, panel_id: int | None, time_from: int, time_to: int
    ) -> str | None:
        if panel_id is None:
            return None
        # Include all template vars (cluster, namespace, region, …) so the link
        # opens the panel pre-filtered to the right service — not all services.
        var_params = "".join(
            f"&var-{k}={v}" for k, v in sorted(self.vars.items()) if v
        )
        return (
            f"{self.base_url}/d/{dashboard_uid}"
            f"?orgId=1{var_params}&viewPanel={panel_id}&from={time_from}&to={time_to}"
        )
