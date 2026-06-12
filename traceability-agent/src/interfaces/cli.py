"""
K8s Debug Agent — paste a Grafana dashboard URL and get a diagnosis.

Usage (minimal — just paste the URL):
  python -m src.interfaces.cli "https://grafana-k8s-ci.myntra.com/d/.../...?var-cluster=pac-mlpcluster01&var-namespace=odmlpppr&from=1778819928386&to=1778820193092"

With optional overrides:
  python -m src.interfaces.cli "<url>" --symptom "p99 spiked to 2s on checkout"
  python -m src.interfaces.cli "<url>" --service catalog-service --backend anthropic

Everything else (service, namespace, cluster, region, time window) is parsed
from the URL's var-* query parameters and from/to timestamps.
"""

import argparse
import asyncio
import os
import re
import sys
import time
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse


def _parse_time(value: str) -> int:
    """Parse relative (1h, 30m, 7d, now-6h, now-7d) or ISO8601 or Unix epoch → Unix ms."""
    now_ms = int(time.time() * 1000)

    if re.match(r"^now-\d+[dhm]", value):
        return _parse_time(value[4:])
    if value == "now":
        return now_ms

    if re.match(r"^\d+[dhm]", value):
        delta_ms = 0
        for num, unit in re.findall(r"(\d+)([dhm])", value):
            multiplier = {"d": 86400, "h": 3600, "m": 60}[unit]
            delta_ms += int(num) * multiplier * 1000
        return now_ms - delta_ms

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass

    try:
        ts = float(value)
        return int(ts * 1000) if ts < 1e12 else int(ts)
    except ValueError:
        pass

    raise ValueError(
        f"Cannot parse time '{value}'. Use: 1h, 30m, 2024-01-15T14:00:00, or Unix ms epoch."
    )


def _parse_dashboard_url(url: str) -> dict:
    """
    Extract all context from a Grafana dashboard URL.

    Pulls out:
      var-namespace  → namespace (and default service name if var-backend is empty)
      var-backend    → service name (if set)
      var-cluster    → cluster
      var-region     → region
      var-ingress_class → ingress_class
      var-cluster_small → cluster_small
      from / to      → Unix ms time range

    Returns dict with all found values plus "_from" and "_to".
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    result: dict[str, str] = {}
    for key, values in params.items():
        value = values[0] if values else ""
        if key.startswith("var-"):
            var_name = key[4:]
            if value and not value.startswith("$") and value != "All":
                result[var_name] = value
        elif key in ("from", "to"):
            result[f"_{key}"] = value

    return result


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.agent import create_agent_from_env

    parser = argparse.ArgumentParser(
        description="K8s service health diagnostic agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="Grafana dashboard URL (contains all cluster/namespace/time context)",
    )
    parser.add_argument(
        "--service",
        default=None,
        help="Service name override (default: var-backend from URL, or var-namespace if backend is empty)",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Namespace override (default: var-namespace from URL)",
    )
    parser.add_argument(
        "--from", dest="time_from", default=None,
        help="Time range start override (e.g. 1h, 30m, ISO8601). Default: from URL or 1h ago.",
    )
    parser.add_argument(
        "--to", dest="time_to", default=None,
        help="Time range end override. Default: from URL or now.",
    )
    parser.add_argument(
        "--symptom", default="",
        help="Optional: describe what you observed (e.g. 'p99 spiked to 2s for 5 minutes')",
    )
    parser.add_argument(
        "--backend", default=None,
        choices=["anthropic", "ollama"],
        help="LLM backend override (default: LLM_BACKEND env var)",
    )

    args = parser.parse_args()

    if not args.url:
        parser.print_help()
        print("\nError: Grafana dashboard URL is required.")
        sys.exit(1)

    # ── Parse all context from URL ────────────────────────────────────────────
    url_data = _parse_dashboard_url(args.url)

    url_from = url_data.pop("_from", None)
    url_to   = url_data.pop("_to",   None)

    # service: explicit flag > var-backend > var-namespace (fallback when backend is blank)
    service = (
        args.service
        or url_data.get("backend")
        or url_data.get("namespace")
    )
    if not service:
        parser.error(
            "Could not determine service name from URL. "
            "Add --service <name> or ensure the URL contains var-namespace."
        )

    # namespace: explicit flag > var-namespace from URL
    namespace = args.namespace or url_data.get("namespace")
    if not namespace:
        parser.error(
            "Could not determine namespace from URL. "
            "Add --namespace <name> or ensure the URL contains var-namespace=<ns>."
        )

    # Remaining URL vars go straight to Grafana template substitution
    grafana_vars: dict[str, str] = {k: v for k, v in url_data.items() if k != "backend"}

    # ── Resolve time range ────────────────────────────────────────────────────
    raw_from = args.time_from or url_from or "1h"
    raw_to   = args.time_to   or url_to

    time_from = _parse_time(raw_from)
    time_to   = int(time.time() * 1000) if raw_to is None else _parse_time(raw_to)

    if time_from >= time_to:
        parser.error("Start time must be before end time.")

    # ── Banner ────────────────────────────────────────────────────────────────
    from_label = datetime.fromtimestamp(time_from / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    to_label   = datetime.fromtimestamp(time_to   / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"\n{'━' * 60}")
    print("  K8s Debug Agent")
    print(f"{'━' * 60}")
    print(f"  Service:    {service} / {namespace}")
    if grafana_vars.get("cluster"):
        print(f"  Cluster:    {grafana_vars['cluster']}")
    if grafana_vars.get("region"):
        print(f"  Region:     {grafana_vars['region']}")
    print(f"  Window:     {from_label} → {to_label}")
    if args.symptom:
        print(f"  Symptom:    {args.symptom}")
    print(f"  Backend:    {args.backend or os.getenv('LLM_BACKEND', 'anthropic')}")
    print(f"{'━' * 60}\n")

    agent = create_agent_from_env(llm_backend=args.backend)

    result = asyncio.run(
        agent.investigate(
            service=service,
            namespace=namespace,
            time_from=time_from,
            time_to=time_to,
            grafana_vars=grafana_vars,
            symptom=args.symptom,
        )
    )

    print(f"\n{'━' * 60}  DIAGNOSIS  {'━' * 60}")
    print(result)
    print(f"{'━' * 60}\n")


if __name__ == "__main__":
    main()
