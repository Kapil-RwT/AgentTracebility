"""
Grafana / Prometheus client for fetching Kubernetes resource limits and requests.

Re-uses the same Grafana endpoint and auth token observed in:
  MLPLAT_metrx-master/pages/01_Kubernetes_Metrics.py
  MLPLAT_metrx-master/core/api_client.py

The metrx repo is NOT modified or imported here.
"""

import os
import time
import warnings

import requests

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

GRAFANA_URL = os.getenv("GRAFANA_URL", "https://grafana-k8s-ci.myntra.com/api/ds/query?ds_type=prometheus")
GRAFANA_TOKEN = f"Bearer {os.getenv('GRAFANA_TOKEN', '')}"
DATASOURCE_UID = os.getenv("GRAFANA_DATASOURCE_UID", "QUS1C844k")
DATASOURCE_ID = 5


class GrafanaClient:
    """Thin wrapper around the Grafana Prometheus datasource API."""

    def __init__(self) -> None:
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": GRAFANA_TOKEN,
        }

    # ── low-level helpers ────────────────────────────────────────────────────

    def _time_range_ms(self, minutes_back: int = 30) -> tuple[str, str]:
        now_ms = int(time.time() * 1000)
        return str(now_ms - minutes_back * 60 * 1000), str(now_ms)

    def _build_payload(
        self,
        expr: str,
        ref_id: str,
        instant: bool = True,
        minutes_back: int = 30,
    ) -> dict:
        from_ms, to_ms = self._time_range_ms(minutes_back)
        return {
            "queries": [
                {
                    "datasource": {"type": "prometheus", "uid": DATASOURCE_UID},
                    "editorMode": "code",
                    "exemplar": False,
                    "expr": expr,
                    "hide": False,
                    "instant": instant,
                    "interval": "5m",
                    "refId": ref_id,
                    "utcOffsetSec": 19800,
                    "datasourceId": DATASOURCE_ID,
                    "intervalMs": 60000,
                    "maxDataPoints": 1 if instant else 968,
                }
            ],
            "from": from_ms,
            "to": to_ms,
        }

    def _post(self, payload: dict) -> dict | None:
        try:
            resp = requests.post(
                GRAFANA_URL,
                headers=self._headers,
                json=payload,
                verify=False,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            print(f"[GrafanaClient] Request failed: {exc}")
            return None

    def _extract_scalar(self, response: dict | None, ref_id: str) -> float | None:
        """Pull the maximum non-null value from an instant-query result frame."""
        if not response:
            return None
        try:
            frames = response["results"][ref_id]["frames"]
            for frame in frames:
                values = frame.get("data", {}).get("values", [])
                if len(values) >= 2 and values[1]:
                    non_null = [v for v in values[1] if v is not None]
                    if non_null:
                        return max(non_null)
        except (KeyError, IndexError, TypeError):
            pass
        return None

    # ── public API ───────────────────────────────────────────────────────────

    def instant_query(self, expr: str, ref_id: str = "A") -> float | None:
        payload = self._build_payload(expr, ref_id, instant=True)
        resp = self._post(payload)
        return self._extract_scalar(resp, ref_id)

    def get_resource_config(self, namespace: str, cluster: str) -> dict:
        """
        Return the current CPU / memory limits and requests for a namespace.

        Queries used are derived from the Kubernetes Metrics page in metrx
        (kube-state-metrics series).  CPU in cores, memory in bytes.
        """
        # fmt: off
        queries = {
            "cpu_limit":    f'max(kube_pod_container_resource_limits{{resource="cpu", namespace="{namespace}", cluster="{cluster}", container="{namespace}"}})',
            "cpu_request":  f'avg(kube_pod_container_resource_requests{{resource="cpu", namespace="{namespace}", cluster="{cluster}", container="{namespace}"}})',
            "mem_limit":    f'max(kube_pod_container_resource_limits{{resource="memory", namespace="{namespace}", cluster="{cluster}", container="{namespace}"}})',
            "mem_request":  f'avg(kube_pod_container_resource_requests{{resource="memory", namespace="{namespace}", cluster="{cluster}", container="{namespace}"}})',
        }
        # fmt: on

        result: dict = {}
        for ref_id, expr in queries.items():
            result[ref_id] = self.instant_query(expr, ref_id=ref_id)

        return result
