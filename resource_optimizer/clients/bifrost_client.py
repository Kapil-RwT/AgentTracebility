"""
Bifrost API client.

API contract derived from the real implementation in:
  MLPLAT_bifrost/…/bifrost/bifrostapi/bifrostquery.py

Exact flow (matches BiFrostQuery class):
  1. POST  BIFROST_SUBMIT_URL
         headers: {Authorization: <token>, Content-Type: application/json}
         body:    json.dumps(payload)
         → response.json()["_links"]["status"]["href"]   ← status-poll URL

  2. GET   <status_url>
         headers: {Id-token: <token>}                    ← NOTE: different header
         → response.json()["status"] == "COMPLETED"
         → response.json()["externalLocation"]           ← gzip download URL

  3. GET   externalLocation  → gzip-decompress → CSV → DataFrame

This client is standalone; the bifrost package is NOT modified or imported.
"""

import gzip
import io
import json
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

# ── Constants (from MLPLAT_bifrost/…/bifrost/constants.py) ──────────────────
BIFROST_SUBMIT_URL = "http://bifrostx-gateway.myntra.com/api/v1/client/async/queries"
BIFROST_TOKEN      = "ZGlrc2hhLmt1bWFyaToxemJuNU9QTGhzUEY"
BIFROST_USER       = "diksha.kumari"
BIFROST_ENGINE     = "PRESTO"
BIFROST_SLEEP_SECS = 10
BIFROST_TIMEOUT_SECS = 600
BIFROST_MAX_RETRY  = 5      # mirrors const.BIFROST_QUERY_MAX_RETRY


class BifrostClient:
    """HTTP client for Bifrost async query API (pac-mlpcluster01 usage data)."""

    def __init__(
        self,
        token: str = BIFROST_TOKEN,
        user: str = BIFROST_USER,
        engine: str = BIFROST_ENGINE,
    ) -> None:
        self.token = token
        self.user = user
        self.engine = engine

    # ── helpers ───────────────────────────────────────────────────────────────

    def _submit_headers(self) -> dict:
        return {"Authorization": self.token, "Content-Type": "application/json"}

    def _status_headers(self) -> dict:
        # Status polling uses 'Id-token', NOT 'Authorization'
        return {"Id-token": self.token}

    def _build_payload(self, query: str) -> dict:
        return {
            "query": query,
            "engine": self.engine,
            "username": self.user,
            "output": {
                "format": "CSV",
                "compression": "GZIP",   # bifrost always returns gzip
            },
        }

    # ── step 1: submit query ─────────────────────────────────────────────────

    def _submit(self, query: str) -> dict:
        """POST the query; retry up to BIFROST_MAX_RETRY times on failure."""
        payload = self._build_payload(query)
        last_exc: Exception | None = None

        for attempt in range(1, BIFROST_MAX_RETRY + 1):
            try:
                resp = requests.request(
                    method="POST",
                    url=BIFROST_SUBMIT_URL,
                    headers=self._submit_headers(),
                    data=json.dumps(payload),   # matches make_request in bifrostquery.py
                    timeout=30,
                )
                if resp.ok:
                    return resp.json()
                last_exc = RuntimeError(
                    f"Submit HTTP {resp.status_code}: {resp.text[:300]}"
                )
            except requests.RequestException as exc:
                last_exc = exc

            wait = BIFROST_SLEEP_SECS * attempt
            time.sleep(wait)

        raise RuntimeError(f"Bifrost submit failed after {BIFROST_MAX_RETRY} retries: {last_exc}")

    # ── step 2: extract status URL ───────────────────────────────────────────

    @staticmethod
    def _status_url(submit_resp: dict) -> str:
        """
        Real bifrost response shape:
          {"_links": {"status": {"href": "http://..."}}}
        """
        try:
            return submit_resp["_links"]["status"]["href"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Unexpected submit response (no _links.status.href): {submit_resp}"
            ) from exc

    # ── step 3: poll until COMPLETED ─────────────────────────────────────────

    def _poll(self, status_url: str) -> dict:
        """
        GET status_url with Id-token header.
        Waits until json["status"] == "COMPLETED".
        Returns the final status JSON (contains externalLocation).
        """
        elapsed = 0
        while elapsed < BIFROST_TIMEOUT_SECS:
            resp = requests.request(
                method="GET",
                url=status_url,
                headers=self._status_headers(),
                data="",
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            status = str(data.get("status", "")).upper()
            if status == "COMPLETED":
                return data
            if status in {"FAILED", "ERROR", "CANCELLED"}:
                raise RuntimeError(
                    f"Bifrost query failed (status={status}): {data}"
                )
            time.sleep(BIFROST_SLEEP_SECS)
            elapsed += BIFROST_SLEEP_SECS

        raise TimeoutError(f"Bifrost query did not complete within {BIFROST_TIMEOUT_SECS}s")

    # ── step 4: download & parse ──────────────────────────────────────────────

    def _download(self, status_resp: dict) -> pd.DataFrame:
        """
        Download the gzip CSV from externalLocation and return as DataFrame.
        Mirrors get_file_content() in bifrostquery.py.
        """
        url = status_resp.get("externalLocation")
        if not url:
            raise ValueError(
                f"No externalLocation in status response: {status_resp}"
            )

        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        # Always gzip-compressed (DEFAULT_OUTPUT_COMPRESSION = "GZIP")
        try:
            content = gzip.decompress(resp.content)
        except (OSError, gzip.BadGzipFile):
            content = resp.content  # fallback: try raw

        return pd.read_csv(io.BytesIO(content))

    # ── public API ────────────────────────────────────────────────────────────

    def run_query(self, query: str) -> pd.DataFrame:
        """Submit a Bifrost SQL query and return the full result as a DataFrame."""
        submit_resp = self._submit(query)
        status_url  = self._status_url(submit_resp)
        final_resp  = self._poll(status_url)
        return self._download(final_resp)

    def get_service_usage(
        self,
        namespace: str,
        cluster: str = "pac-mlpcluster01",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch daily CPU + memory usage for a namespace over the last `days` days.
        Uses max_cpu (cores) and max_mem (bytes) as peak-usage figures.
        """
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_str  = start_date.strftime("%Y-%m-%d")
        end_str    = end_date.strftime("%Y-%m-%d")

        query = f"""
SELECT date, cluster, namespace, department,
       min_pod, avg_pod, max_pod,
       min_cores, avg_cores, max_cores,
       min_cpu,  avg_cpu,  max_cpu,
       p50, p90, p95, p99,
       min_mem,  avg_mem,  max_mem,
       m50, m90, m95, m99
FROM (
    SELECT a.date,
           a.cluster,
           a.namespace,
           COALESCE(b.dept, 'NA') AS department,
           c.min_cores, c.avg_cores, c.max_cores,
           a.min_cpu,   a.avg_cpu,   a.max_cpu,
           a.p50, a.p90, a.p95, a.p99,
           d.min_mem,   d.avg_mem,   d.max_mem,
           d.m50, d.m90, d.m95, d.m99,
           e.min_pod,   e.avg_pod,   e.max_pod
    FROM  sre_cost_analysis.cpu_usage a
    LEFT JOIN sre_cost_analysis.core_usage c
           ON a.date = c.date AND a.cluster = c.cluster AND a.namespace = c.namespace
    LEFT JOIN sre_cost_analysis.mem_usage d
           ON a.date = d.date AND a.cluster = d.cluster AND a.namespace = d.namespace
    LEFT JOIN sre_cost_analysis.pod_usage e
           ON a.date = e.date AND a.cluster = e.cluster AND a.namespace = e.namespace
    LEFT JOIN sre_cost_analysis.service_dept_map b
           ON a.namespace = b.service
) AS virtual_table
WHERE cluster   IN ('{cluster}')
  AND namespace  = '{namespace}'
  AND date      >= DATE '{start_str}'
  AND date       < DATE '{end_str}'
LIMIT 1000
""".strip()

        return self.run_query(query)
