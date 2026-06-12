"""
Tool implementations for the K8s debug agent.

Query strategy (two-layer):
  1. PRIMARY  — fetch the actual PromQL from the Grafana dashboard panel and substitute
                the real template variables ($namespace, $backend, $cluster …).
                This guarantees every service is filtered correctly because we use
                exactly the same queries the dashboard uses.
  2. FALLBACK — if the panel fetch fails (auth, panel moved, etc.) use the
                pre-written templates from config/queries.yaml.
"""

import asyncio
import os
from typing import Any

import yaml

from .grafana_client import GrafanaClient

# ─── Config loading (lazy, cached) ────────────────────────────────────────────

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
_panels: dict | None = None
_thresholds: dict | None = None
_queries: dict | None = None


def _cfg() -> tuple[dict, dict, dict]:
    global _panels, _thresholds, _queries
    if _panels is None:
        with open(os.path.join(_CONFIG_DIR, "panels.yaml")) as f:
            _panels = yaml.safe_load(f)
        with open(os.path.join(_CONFIG_DIR, "thresholds.yaml")) as f:
            _thresholds = yaml.safe_load(f)
        with open(os.path.join(_CONFIG_DIR, "queries.yaml")) as f:
            _queries = yaml.safe_load(f)

        # Allow dashboard UIDs to be overridden via env vars without editing YAML
        if os.getenv("GRAFANA_MAIN_DASHBOARD_UID"):
            _panels["main_dashboard"]["uid"] = os.environ["GRAFANA_MAIN_DASHBOARD_UID"]
        if os.getenv("GRAFANA_COREDNS_DASHBOARD_UID"):
            _panels["coredns_dashboard"]["uid"] = os.environ["GRAFANA_COREDNS_DASHBOARD_UID"]
        if os.getenv("GRAFANA_NODE_DASHBOARD_UID"):
            _panels["node_dashboard"]["uid"] = os.environ["GRAFANA_NODE_DASHBOARD_UID"]

    return _panels, _thresholds, _queries


async def _query_with_fallback(
    client: GrafanaClient,
    dashboard_uid: str,
    panel_id: int | None,
    variables: dict[str, str],
    fallback: str,
    time_from: int,
    time_to: int,
    target_index: int = 0,
    prefer_p99: bool = False,
) -> tuple[list[dict], str]:
    """
    Two-stage query strategy:
      1. Fetch the panel's PromQL from Grafana and query Prometheus.
         If the panel exists AND returns data → use it.
      2. If the panel is missing, fails, or returns zero series →
         apply the YAML fallback template and query again.

    Returns (frames, expression_used) so callers get data in one await.
    """
    fallback_expr = client.apply_variables(fallback, variables)

    if panel_id is not None:
        try:
            panel_expr = await client.get_panel_expr(
                dashboard_uid, panel_id, variables, target_index, prefer_p99
            )
            if panel_expr:
                frames = await client.query_range(panel_expr, time_from, time_to)
                if _parse_frames(frames):
                    return frames, panel_expr
                # Panel expression returned no data → fall through to YAML fallback
        except Exception:
            pass

    frames = await client.query_range(fallback_expr, time_from, time_to)
    return frames, fallback_expr


def _render_fallback(template: str, **kwargs) -> str:
    """Replace UPPER_CASE placeholder tokens in a fallback PromQL template."""
    q = template.strip()
    for key, value in kwargs.items():
        q = q.replace(key, str(value))
    return q


# ─── Frame parsing helpers ─────────────────────────────────────────────────────

def _parse_frames(frames: list[dict]) -> dict[str, list[float]]:
    """
    Convert Grafana data frames into {label_string: [float_values]} dict.
    Null values are dropped. Label is built from field labels or field name.
    """
    result: dict[str, list[float]] = {}
    for frame in frames:
        fields = frame.get("schema", {}).get("fields", [])
        values = frame.get("data", {}).get("values", [])
        if len(fields) < 2 or len(values) < 2:
            continue
        value_field = fields[1]
        labels = value_field.get("labels", {})
        if labels:
            label = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        else:
            label = value_field.get("name", "value")
        result[label] = [v for v in values[1] if v is not None]
    return result


def _stats(values: list[float]) -> dict:
    """Return p50/p90/p99/max/avg for a list of values."""
    if not values:
        return {"p50": None, "p90": None, "p99": None, "max": None, "avg": None}
    s = sorted(values)
    n = len(s)
    return {
        "p50": round(s[int(n * 0.50)], 2),
        "p90": round(s[int(n * 0.90)], 2),
        "p99": round(s[min(int(n * 0.99), n - 1)], 2),
        "max": round(s[-1], 2),
        "avg": round(sum(s) / n, 2),
    }


# ─── Tool implementations ──────────────────────────────────────────────────────

async def fetch_latency_at_hop(
    client: GrafanaClient,
    service: str,
    namespace: str,
    hop: str,
    granularity: str,
    time_from: int,
    time_to: int,
) -> dict:
    """
    Fetch p99 latency at a specific traffic hop.

    hop: "inbound" (T4), "outbound" (T2/T6), "ingress" (T3)
    granularity: "deployment" or "pod"

    Primary source: actual panel PromQL from Grafana dashboard (uses $backend, $namespace
    variables exactly as the dashboard does, so it works for every service).
    Fallback: pre-written PromQL from config/queries.yaml.
    """
    panels, thresholds, queries = _cfg()
    dash_uid = panels["main_dashboard"]["uid"]

    if hop not in ("inbound", "outbound", "ingress"):
        return {"error": f"Unknown hop '{hop}'. Use: inbound, outbound, ingress"}

    # Variables matching the Grafana dashboard URL template vars
    # $backend = service/deployment name, $namespace = k8s namespace
    variables = {
        "backend": service,
        "namespace": namespace,
    }

    panel_map = {
        "inbound_deployment":  panels["main_dashboard"]["panels"]["linkerd_inbound_latency_deployment"],
        "inbound_pod":         panels["main_dashboard"]["panels"]["linkerd_inbound_latency_pod"],
        "outbound_deployment": panels["main_dashboard"]["panels"]["linkerd_outbound_latency_deployment"],
        "outbound_pod":        panels["main_dashboard"]["panels"]["linkerd_outbound_latency_pod"],
        # Panel 42 uses haproxy_backend_response_time_average_seconds (seconds gauge).
        # We re-enable the panel but multiply all raw values ×1000 after fetch to get ms.
        "ingress_deployment":  panels["main_dashboard"]["panels"]["ingress_latency_total"],
        "ingress_pod":         None,
    }
    panel_key = f"{hop}_{granularity}"
    panel_id = panel_map.get(panel_key)

    # Fallback queries use UPPER_CASE tokens; convert to Grafana $var style
    cluster = client.vars.get("cluster", "")
    if hop == "inbound":
        fallback = _render_fallback(
            queries["linkerd_inbound_latency_p99"],
            SERVICE=service, NAMESPACE=namespace, WINDOW="5m", GROUP_BY=granularity,
        )
    elif hop == "outbound":
        fallback = _render_fallback(
            queries["linkerd_outbound_latency_p99"],
            SERVICE=service, NAMESPACE=namespace, WINDOW="5m", GROUP_BY=granularity,
        )
    else:
        fallback = _render_fallback(
            queries["haproxy_ingress_latency_p99_ms"],
            NAMESPACE=namespace, CLUSTER=cluster, WINDOW="1m",
        )

    # ingress panel may have P99 targets; others use p99 from Linkerd histogram
    prefer_p99_flag = hop != "ingress"

    try:
        frames, _ = await _query_with_fallback(
            client, dash_uid, panel_id, variables, fallback,
            time_from, time_to, prefer_p99=prefer_p99_flag,
        )
        series_raw = _parse_frames(frames)

        # HAProxy ingress metrics at Myntra are in seconds (panel gauge and histogram fallback
        # both use seconds-scale values despite the "milliseconds" metric name).
        # Multiply by 1000 so all downstream comparisons and display are in ms.
        if hop == "ingress":
            series_raw = {k: [v * 1000 for v in vals] for k, vals in series_raw.items()}

        series = {k: _stats(v) for k, v in series_raw.items()}

        anomaly_pods: list[str] = []

        if hop == "ingress":
            spike_ms = thresholds["latency"].get("ingress_spike_ms", 300)
            anomaly_pods = [k for k, v in series.items() if (v.get("p99") or 0) > spike_ms]
        else:
            all_p99 = [v["p99"] for v in series.values() if v["p99"] is not None]
            if len(all_p99) > 1:
                median_p99 = sorted(all_p99)[(len(all_p99) - 1) // 2]
                mult = thresholds["latency"]["outlier_multiplier"]
                anomaly_pods = [k for k, v in series.items() if v["p99"] and v["p99"] > median_p99 * mult]

        # Active production pods: have p99 data in the window AND are not debug pods.
        # Terminated pods (rolling deployment history) have null p99 → excluded.
        # Debug pods are never counted toward fleet size or anomaly denominators.
        def _is_debug(label: str) -> bool:
            name = label
            for part in label.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    if k.strip() in ("pod", "kubernetes_pod_name"):
                        name = v.strip()
            return "debug" in name.lower()

        active_prod_series = {
            k: v for k, v in series.items()
            if v.get("p99") is not None and not _is_debug(k)
        }
        prod_anomaly_pods = [p for p in anomaly_pods if not _is_debug(p)]

        grafana_panel_id = panel_id if panel_id is not None else panels["main_dashboard"]["panels"]["ingress_latency_total"]
        return {
            "hop": hop,
            "granularity": granularity,
            "series": series,
            "active_pod_count": len(active_prod_series),
            "anomaly_pods": prod_anomaly_pods,
            "all_pods_elevated": len(active_prod_series) > 0 and len(prod_anomaly_pods) >= len(active_prod_series),
            "grafana_url": client.panel_url(dash_uid, grafana_panel_id, time_from, time_to),
        }
    except Exception as e:
        return {"error": str(e), "hop": hop}


async def fetch_cpu_throttling(
    client: GrafanaClient,
    service: str,
    namespace: str,
    time_from: int,
    time_to: int,
) -> dict:
    """
    Check CPU CFS throttling per pod.

    Returns per-pod throttle % stats and a dominant pattern:
      - "none":       no significant throttling
      - "spiky":      occasional bursts above limit → set limit = 2x request
      - "continuous": sustained throttling → increase request or add pods
    """
    panels, thresholds, queries = _cfg()
    dash_uid = panels["main_dashboard"]["uid"]
    panel_id = panels["main_dashboard"]["panels"].get("cpu_throttle")

    variables = {"backend": service, "namespace": namespace}
    fallback = _render_fallback(
        queries["cpu_throttle_pct"], SERVICE=service, NAMESPACE=namespace, WINDOW="5m"
    )

    try:
        frames, _ = await _query_with_fallback(
            client, dash_uid, panel_id, variables, fallback, time_from, time_to
        )
        series_raw = _parse_frames(frames)

        thr = thresholds["cpu_throttle"]
        pods: dict[str, Any] = {}
        pattern_votes = {"none": 0, "spiky": 0, "continuous": 0}

        for pod, vals in series_raw.items():
            if not vals:
                continue
            frac_continuous = sum(1 for v in vals if v > thr["continuous_threshold_pct"]) / len(vals)

            if frac_continuous < thr["none_fraction"]:
                pattern = "none"
            elif frac_continuous > thr["continuous_fraction"]:
                pattern = "continuous"
            else:
                pattern = "spiky"

            pattern_votes[pattern] += 1
            pods[pod] = {**_stats(vals), "pattern": pattern}

        if not pods:
            return {"pods": {}, "dominant_pattern": "none", "pattern_counts": pattern_votes,
                    "recommended_fix": None, "no_data": True}

        dominant = max(pattern_votes, key=lambda k: pattern_votes[k])
        fix_map = {
            "none": None,
            "spiky": "Set CPU limit = 2× CPU request (e.g., request=4 → limit=8). Gives burst headroom within CFS period.",
            "continuous": "Increase CPU request OR add more pods until throttling drops to near-zero.",
        }

        return {
            "pods": pods,
            "dominant_pattern": dominant,
            "pattern_counts": pattern_votes,
            "recommended_fix": fix_map[dominant],
            "no_data": False,
        }
    except Exception as e:
        return {"error": str(e)}


async def fetch_rpm_distribution(
    client: GrafanaClient,
    service: str,
    namespace: str,
    time_from: int,
    time_to: int,
) -> dict:
    """
    Check RPM distribution across pods.

    Uneven distribution (max > 2× min with ≥3 pods) indicates the upstream caller or
    load balancer is routing traffic unevenly. Common causes: HTTP keep-alive connections
    (long-lived upstream connections stick to one pod), sticky sessions, or connection
    pool imbalance at the caller.
    """
    _, thresholds, queries = _cfg()
    expr = _render_fallback(queries["inbound_rpm_per_pod"], SERVICE=service, NAMESPACE=namespace, WINDOW="5m")

    try:
        frames = await client.query_range(expr, time_from, time_to)
        series_raw = _parse_frames(frames)

        pod_avg: dict[str, float] = {}
        for pod, vals in series_raw.items():
            if vals:
                pod_avg[pod] = round(sum(vals) / len(vals), 1)

        if not pod_avg:
            return {"pods": {}, "uneven": False, "message": "No RPM data found"}

        rpm_cfg = thresholds["rpm"]
        rpms = list(pod_avg.values())
        max_rpm, min_rpm = max(rpms), max(min(rpms), 0.1)
        uneven = len(rpms) >= rpm_cfg["min_pod_count"] and max_rpm > min_rpm * rpm_cfg["imbalance_multiplier"]

        return {
            "pods": pod_avg,
            "max_rpm": round(max_rpm, 1),
            "min_rpm": round(min_rpm, 1),
            "mean_rpm": round(sum(rpms) / len(rpms), 1),
            "imbalance_ratio": round(max_rpm / min_rpm, 2),
            "uneven": uneven,
            "recommended_fix": (
                "Uneven RPM distribution detected. Common causes: HTTP keep-alive connections "
                "from the upstream service routing all requests on a persistent connection to "
                "one pod; sticky session misconfiguration; or connection pool asymmetry at the "
                "caller. Identify the upstream service and check its load balancing configuration."
            ) if uneven else None,
        }
    except Exception as e:
        return {"error": str(e)}


async def fetch_pod_restarts(
    client: GrafanaClient,
    service: str,
    namespace: str,
    time_from: int,
    time_to: int,
) -> dict:
    """
    Count pod restarts in the investigation window.

    Restarts cause brief latency spikes: the pod IP changes, kube-proxy/Linkerd
    endpoint propagation takes a few seconds, and new connections hit the old IP
    during that window.
    """
    _, thresholds, queries = _cfg()
    min_window = thresholds["pods"]["min_restart_window_s"]
    duration_s = max((time_to - time_from) // 1000, min_window)

    restart_expr = _render_fallback(
        queries["pod_restarts"],
        SERVICE=service, NAMESPACE=namespace, DURATION_S=str(duration_s),
    )
    oom_expr = _render_fallback(
        queries["oom_events"],
        SERVICE=service, NAMESPACE=namespace, DURATION_S=str(duration_s),
    )
    reason_expr = _render_fallback(
        queries["pod_restart_reasons"],
        SERVICE=service, NAMESPACE=namespace,
    )

    try:
        restart_frames, oom_frames, reason_frames = await asyncio.gather(
            client.query_range(restart_expr, time_from, time_to),
            client.query_range(oom_expr, time_from, time_to),
            client.query_range(reason_expr, time_from, time_to),
        )

        series_raw = _parse_frames(restart_frames)
        restarts = {pod: int(max(vals)) for pod, vals in series_raw.items() if vals}
        total = sum(restarts.values())

        # OOM events: count per pod (sum across containers)
        oom_raw = _parse_frames(oom_frames)
        oom_events: dict[str, int] = {}
        for label, vals in oom_raw.items():
            if vals and max(vals) > 0:
                parts = dict(p.split("=", 1) for p in label.split(",") if "=" in p)
                pod = parts.get("pod", label)
                oom_events[pod] = oom_events.get(pod, 0) + int(max(vals))
        total_oom = sum(oom_events.values())

        # Restart reasons: last termination reason per pod
        reason_raw = _parse_frames(reason_frames)
        restart_reasons: dict[str, str] = {}
        for label, vals in reason_raw.items():
            if vals and any(v > 0 for v in vals):
                parts = dict(p.split("=", 1) for p in label.split(",") if "=" in p)
                pod = parts.get("pod", "")
                reason = parts.get("reason", "")
                if pod and reason:
                    restart_reasons[pod] = reason

        return {
            "pods": restarts,
            "total_restarts": total,
            "has_restarts": total > 0,
            "has_data": bool(series_raw),
            "oom_events": oom_events,
            "total_oom_events": total_oom,
            "restart_reasons": restart_reasons,
            "recommended_fix": (
                "OOMKill detected — ask SRE to increase memory limit. "
                if total_oom > 0 else (
                    "Usually self-heals once new pod IP propagates (~seconds). "
                    "If restart frequency is high: increase pod count. "
                    "If >10 pods and still frequent: escalate to SRE."
                ) if total > 0 else None
            ),
        }
    except Exception as e:
        return {"error": str(e)}


async def fetch_worker_node_metrics(
    client: GrafanaClient,
    node_ip: str,
    time_from: int,
    time_to: int,
) -> dict:
    """
    Fetch VM-level metrics for a specific worker node.

    Use this after identifying which node a problematic pod is scheduled on
    (from the 'Pod IP and worker node mapping' Grafana panel).
    Detects noisy-neighbour situations where one container exhausts shared VM resources.
    """
    if not node_ip or not node_ip.strip():
        return {"error": "node_ip is required — get it from the 'Pod IP and worker node mapping' Grafana panel"}

    _, thresholds, queries = _cfg()
    wn = thresholds["worker_node"]
    warn_pct = thresholds["alert_threshold_pct"] / 100
    window = "5m"

    metric_queries = {
        "cpu_usage_pct":        _render_fallback(queries["node_cpu_usage_pct"], NODE_IP=node_ip, WINDOW=window),
        "memory_used_gb":       _render_fallback(queries["node_memory_used_gb"], NODE_IP=node_ip, WINDOW=window),
        "disk_iops":            _render_fallback(queries["node_disk_iops"], NODE_IP=node_ip, WINDOW=window),
        "disk_throughput_mbps": _render_fallback(queries["node_disk_throughput_mbps"], NODE_IP=node_ip, WINDOW=window),
        "network_mbps":         _render_fallback(queries["node_network_mbps"], NODE_IP=node_ip, WINDOW=window),
        "disk_wait_ms":         _render_fallback(queries["node_disk_wait_ms"], NODE_IP=node_ip, WINDOW=window),
    }

    hard_limits = {
        "disk_iops":            wn["disk_iops"],
        "disk_throughput_mbps": wn["disk_throughput_mbps"],
        "network_mbps":         wn["network_gbps"] * 1000,
    }

    metrics: dict[str, Any] = {}
    breaches: list[dict] = []

    for name, expr in metric_queries.items():
        try:
            frames = await client.query_range(expr, time_from, time_to)
            series_raw = _parse_frames(frames)
            all_vals: list[float] = [v for series in series_raw.values() for v in series]
            if all_vals:
                s = _stats(all_vals)
                metrics[name] = s
                limit = hard_limits.get(name)
                if limit and s["p90"] is not None and s["p90"] > limit * warn_pct:
                    breaches.append({
                        "metric": name,
                        "p90": s["p90"],
                        "limit": limit,
                        "pct_used": round(s["p90"] / limit * 100, 1),
                    })
            else:
                metrics[name] = {"error": "no data"}
        except Exception as e:
            metrics[name] = {"error": str(e)}

    return {
        "node_ip": node_ip,
        "metrics": metrics,
        "threshold_breaches": breaches,
        "is_noisy_neighbour": len(breaches) > 0,
        "recommended_fix": (
            "Noisy neighbour detected. "
            "1. Ask SRE to restart the offending pod to reschedule on a less-loaded node. "
            "2. If a specific service is consuming excess resources, adjust its CPU/disk quota. "
            "3. If overall node capacity is insufficient: escalate to SRE for node pool scaling."
        ) if breaches else None,
    }


async def fetch_dns_latency(
    client: GrafanaClient,
    time_from: int,
    time_to: int,
) -> dict:
    """
    Check CoreDNS p99 resolution latency.

    High DNS latency affects every service call that uses a DNS name.
    Primary: fetch from CoreDNS Grafana panel. Fallback: standard CoreDNS PromQL.
    """
    panels, thresholds, queries = _cfg()
    coredns = panels["coredns_dashboard"]
    panel_id = coredns["panels"]["dns_latency"]

    fallback = _render_fallback(queries["dns_latency_p99_ms"], WINDOW="5m")

    try:
        frames, _ = await _query_with_fallback(
            client, coredns["uid"], panel_id, {}, fallback,
            time_from, time_to, prefer_p99=True,
        )
        series_raw = _parse_frames(frames)

        all_vals: list[float] = [v for series in series_raw.values() for v in series]
        stats = _stats(all_vals)
        thr = thresholds["dns"]

        if stats["p99"] is None:
            status = "unknown"
        elif stats["p99"] > thr["critical_latency_ms"]:
            status = "critical"
        elif stats["p99"] > thr["warn_latency_ms"]:
            status = "warn"
        else:
            status = "ok"

        return {
            "latency_ms": stats,
            "status": status,
            "recommended_fix": (
                "DNS latency is elevated. "
                "Forward DNS timeout → escalate to SRE. "
                "Reverse DNS issues → check if your library resolves reverse DNS for connection pooling; "
                "if the load balancer in the path has TCP disabled, DNS over UDP is unreliable for large responses."
            ) if status in ("warn", "critical") else None,
            "grafana_url": client.panel_url(coredns["uid"], panel_id, time_from, time_to),
        }
    except Exception as e:
        return {"error": str(e)}


async def fetch_disk_iops(
    client: GrafanaClient,
    service: str,
    namespace: str,
    time_from: int,
    time_to: int,
) -> dict:
    """
    Check disk IOPS per pod and IOPS-per-CPU-core ratio.

    Worker nodes have a hard 2300 IOPS cap shared across all containers.
    Per-core limit is ~12 IOPS. Exceeding this causes disk-wait latency.
    """
    panels, thresholds, queries = _cfg()
    dash_uid = panels["main_dashboard"]["uid"]

    variables = {"backend": service, "namespace": namespace}
    iops_fallback  = _render_fallback(queries["disk_iops_per_pod"],  SERVICE=service, NAMESPACE=namespace, WINDOW="5m")
    cpu_fallback   = _render_fallback(queries["cpu_request_cores"],  SERVICE=service, NAMESPACE=namespace)

    try:
        (iops_frames, _), (cpu_frames, _) = await asyncio.gather(
            _query_with_fallback(
                client, dash_uid,
                panels["main_dashboard"]["panels"].get("max_storage_iops"),
                variables, iops_fallback, time_from, time_to,
            ),
            _query_with_fallback(
                client, dash_uid,
                panels["main_dashboard"]["panels"].get("cpu_request"),
                variables, cpu_fallback, time_from, time_to,
            ),
        )

        iops_raw = _parse_frames(iops_frames)
        cpu_raw  = _parse_frames(cpu_frames)

        total_cpu = sum(sum(v) / len(v) for v in cpu_raw.values() if v) or 1.0

        per_pod: dict[str, float] = {}
        peak_iops = 0.0
        for pod, vals in iops_raw.items():
            if vals:
                per_pod[pod] = round(sum(vals) / len(vals), 1)
                peak_iops = max(peak_iops, max(vals))

        if not per_pod:
            return {
                "pods_avg_iops": {}, "peak_iops": 0.0, "total_cpu_cores": 0.0,
                "iops_per_core": 0.0, "threshold_iops_per_core": thresholds["per_core"]["disk_iops"],
                "exceeds_threshold": False, "recommended_fix": None, "no_data": True,
            }

        iops_per_core = round(peak_iops / total_cpu, 2)
        limit_per_core = thresholds["per_core"]["disk_iops"]
        exceeds = iops_per_core > limit_per_core

        return {
            "pods_avg_iops": per_pod,
            "peak_iops": round(peak_iops, 1),
            "total_cpu_cores": round(total_cpu, 1),
            "iops_per_core": iops_per_core,
            "threshold_iops_per_core": limit_per_core,
            "exceeds_threshold": exceeds,
            "recommended_fix": (
                f"IOPS/core ({iops_per_core}) exceeds limit ({limit_per_core}). "
                "1. Switch to non-blocking (async) log writes with ≥50ms flush buffer. "
                "2. Increase pod count by ~5% to distribute I/O load across more nodes. "
                "3. Increase CPU request to get a proportionally larger IOPS share."
            ) if exceeds else None,
            "no_data": False,
            "grafana_url": client.panel_url(
                dash_uid,
                panels["main_dashboard"]["panels"].get("max_storage_iops"),
                time_from, time_to,
            ),
        }
    except Exception as e:
        return {"error": str(e)}


async def fetch_error_rate(
    client: GrafanaClient,
    service: str,
    namespace: str,
    time_from: int,
    time_to: int,
) -> dict:
    """
    Fetch Linkerd inbound success rate, error rate, and total RPS.

    Uses Linkerd's response_total metric with classification label.
    Returns success_rate_pct, error_rate_pct, total_rps, and per-status error breakdown.
    """
    _, _, queries = _cfg()

    success_expr = _render_fallback(
        queries["linkerd_success_rate_pct"], SERVICE=service, NAMESPACE=namespace, WINDOW="5m"
    )
    rps_expr = _render_fallback(
        queries["linkerd_total_rps"], SERVICE=service, NAMESPACE=namespace, WINDOW="5m"
    )
    error_expr = _render_fallback(
        queries["linkerd_error_rate_by_status"], SERVICE=service, NAMESPACE=namespace, WINDOW="5m"
    )

    try:
        success_frames, rps_frames, error_frames = await asyncio.gather(
            client.query_range(success_expr, time_from, time_to),
            client.query_range(rps_expr, time_from, time_to),
            client.query_range(error_expr, time_from, time_to),
        )

        success_raw = _parse_frames(success_frames)
        rps_raw = _parse_frames(rps_frames)
        error_raw = _parse_frames(error_frames)

        def _avg(series: dict) -> float | None:
            all_vals = [v for vals in series.values() for v in vals if v is not None]
            return round(sum(all_vals) / len(all_vals), 2) if all_vals else None

        success_rate = _avg(success_raw)
        total_rps = _avg(rps_raw)
        error_rate = round(100 - success_rate, 2) if success_rate is not None else None

        error_by_status: dict[str, float] = {}
        for label, vals in error_raw.items():
            if vals:
                error_by_status[label] = round(sum(vals) / len(vals), 4)

        thr_error = 1.0
        is_elevated = error_rate is not None and error_rate > thr_error

        return {
            "success_rate_pct": success_rate,
            "error_rate_pct": error_rate,
            "total_rps": total_rps,
            "error_by_status": error_by_status,
            "is_elevated_errors": is_elevated,
            "recommended_fix": (
                f"Error rate {error_rate}% is elevated. "
                "Check application logs for exceptions. "
                "If 5xx: review recent deployments. "
                "If 4xx: check client request format or auth config."
            ) if is_elevated else None,
        }
    except Exception as e:
        return {"error": str(e)}


async def fetch_memory_utilization(
    client: GrafanaClient,
    service: str,
    namespace: str,
    time_from: int,
    time_to: int,
) -> dict:
    """
    Check per-pod memory utilization as percentage of the memory limit.
    Flags pods at warn (≥80%) or critical (≥90%) thresholds — early warning before OOMKill.
    """
    _, thresholds, queries = _cfg()
    mem_cfg = thresholds.get("memory", {"warn_pct": 80, "critical_pct": 90})

    ws_expr  = _render_fallback(queries["memory_working_set_bytes"], SERVICE=service, NAMESPACE=namespace)
    lim_expr = _render_fallback(queries["memory_limit_bytes"],       SERVICE=service, NAMESPACE=namespace)

    try:
        ws_frames, lim_frames = await asyncio.gather(
            client.query_range(ws_expr,  time_from, time_to),
            client.query_range(lim_expr, time_from, time_to),
        )
        ws_raw  = _parse_frames(ws_frames)
        lim_raw = _parse_frames(lim_frames)

        def _avg_bytes(series: dict) -> dict[str, float]:
            return {k: sum(v) / len(v) for k, v in series.items() if v}

        ws_avg  = _avg_bytes(ws_raw)
        lim_avg = _avg_bytes(lim_raw)

        pods: dict[str, dict] = {}
        warn_count = 0
        critical_count = 0

        for pod_label, ws_bytes in ws_avg.items():
            limit = lim_avg.get(pod_label)
            pct = round(ws_bytes / limit * 100, 1) if limit and limit > 0 else None
            pods[pod_label] = {
                "working_set_mb": round(ws_bytes / 1e6, 1),
                "limit_mb": round(limit / 1e6, 1) if limit else None,
                "utilization_pct": pct,
            }
            if pct is not None:
                if pct >= mem_cfg["critical_pct"]:
                    critical_count += 1
                elif pct >= mem_cfg["warn_pct"]:
                    warn_count += 1

        max_pct = max(
            (p["utilization_pct"] for p in pods.values() if p.get("utilization_pct") is not None),
            default=None,
        )

        return {
            "pods": pods,
            "critical_count": critical_count,
            "warn_count": warn_count,
            "max_utilization_pct": max_pct,
            "warn_pct_threshold": mem_cfg["warn_pct"],
            "critical_pct_threshold": mem_cfg["critical_pct"],
        }
    except Exception as e:
        return {"error": str(e)}


async def fetch_replica_count(
    client: GrafanaClient,
    service: str,
    namespace: str,
    time_from: int,
    time_to: int,
) -> dict:
    """
    Check deployment replica health: available vs desired vs unavailable.
    Unavailable replicas indicate HPA scale-up in progress, scheduling failures, or CrashLoops.
    """
    _, _, queries = _cfg()

    avail_expr   = _render_fallback(queries["deployment_replicas_available"],   SERVICE=service, NAMESPACE=namespace)
    desired_expr = _render_fallback(queries["deployment_replicas_desired"],     SERVICE=service, NAMESPACE=namespace)
    unavail_expr = _render_fallback(queries["deployment_replicas_unavailable"], SERVICE=service, NAMESPACE=namespace)

    try:
        avail_frames, desired_frames, unavail_frames = await asyncio.gather(
            client.query_range(avail_expr,   time_from, time_to),
            client.query_range(desired_expr, time_from, time_to),
            client.query_range(unavail_expr, time_from, time_to),
        )

        def _last_val(frames: list) -> float | None:
            series = _parse_frames(frames)
            all_vals = [v for vals in series.values() for v in vals if v is not None]
            return all_vals[-1] if all_vals else None

        def _max_val(frames: list) -> float | None:
            series = _parse_frames(frames)
            all_vals = [v for vals in series.values() for v in vals if v is not None]
            return max(all_vals) if all_vals else None

        def _min_val(frames: list) -> float | None:
            series = _parse_frames(frames)
            all_vals = [v for vals in series.values() for v in vals if v is not None]
            return min(all_vals) if all_vals else None

        # Current state: last value in window (not average — gauge can only be read now)
        available   = _last_val(avail_frames)
        desired     = _last_val(desired_frames)
        # Worst case during window: max unavailable tells us if replicas were ever missing
        unavailable = _max_val(unavail_frames)
        # Did replicas dip during the spike? Min available reveals the worst moment
        min_available = _min_val(avail_frames)

        is_degraded = (
            (unavailable is not None and unavailable > 0)
            or (min_available is not None and desired is not None and min_available < desired)
        )

        return {
            "available":     int(available)     if available     is not None else None,
            "desired":       int(desired)       if desired       is not None else None,
            "unavailable":   int(unavailable)   if unavailable   is not None else None,
            "min_available": int(min_available) if min_available is not None else None,
            "is_degraded":   is_degraded,
        }
    except Exception as e:
        return {"error": str(e)}


async def fetch_hpa_status(
    client: GrafanaClient,
    service: str,
    namespace: str,
    time_from: int,
    time_to: int,
) -> dict:
    """
    Check HPA scaling behaviour during the investigation window.
    Detects whether HPA scaled up (new pods warming up can temporarily increase latency)
    and whether it hit the configured max limit (cannot scale further).
    """
    _, _, queries = _cfg()

    desired_expr   = _render_fallback(queries["hpa_desired_replicas"],   NAMESPACE=namespace)
    current_expr   = _render_fallback(queries["hpa_current_replicas"],   NAMESPACE=namespace)
    min_expr       = _render_fallback(queries["hpa_min_replicas"],       NAMESPACE=namespace)
    max_expr       = _render_fallback(queries["hpa_max_replicas"],       NAMESPACE=namespace)
    cpu_util_expr  = _render_fallback(queries["hpa_cpu_utilization_pct"], NAMESPACE=namespace)

    try:
        desired_frames, current_frames, min_frames, max_frames, cpu_frames = await asyncio.gather(
            client.query_range(desired_expr,  time_from, time_to),
            client.query_range(current_expr,  time_from, time_to),
            client.query_range(min_expr,      time_from, time_to),
            client.query_range(max_expr,      time_from, time_to),
            client.query_range(cpu_util_expr, time_from, time_to),
        )

        def _flat(frames: list) -> list[float]:
            s = _parse_frames(frames)
            return [v for vals in s.values() for v in vals if v is not None]

        def _last_scalar(frames: list) -> float | None:
            vals = _flat(frames)
            return vals[-1] if vals else None

        desired_vals = _flat(desired_frames)
        cpu_vals     = _flat(cpu_frames)

        hpa_min = _last_scalar(min_frames)
        hpa_max = _last_scalar(max_frames)
        current_desired = desired_vals[-1] if desired_vals else None
        peak_desired    = max(desired_vals)  if desired_vals else None
        min_desired     = min(desired_vals)  if desired_vals else None

        was_scaling = (
            peak_desired is not None and min_desired is not None
            and (peak_desired - min_desired) >= 2
        )
        hit_max_limit = (
            hpa_max is not None and peak_desired is not None
            and peak_desired >= hpa_max
        )
        avg_cpu_util = round(sum(cpu_vals) / len(cpu_vals), 1) if cpu_vals else None

        return {
            "hpa_min":          int(hpa_min)          if hpa_min         is not None else None,
            "hpa_max":          int(hpa_max)          if hpa_max         is not None else None,
            "current_desired":  int(current_desired)  if current_desired is not None else None,
            "peak_desired":     int(peak_desired)     if peak_desired    is not None else None,
            "was_scaling":      was_scaling,
            "hit_max_limit":    hit_max_limit,
            "cpu_utilization_pct": avg_cpu_util,
            "no_data": desired_vals == [],
        }
    except Exception as e:
        return {"error": str(e)}


async def fetch_ingress_health(
    client: GrafanaClient,
    service: str,
    namespace: str,
    time_from: int,
    time_to: int,
) -> dict:
    """
    Comprehensive HAProxy ingress analysis beyond total P99.

    Latency breakdown (Tq/Tw/Tc/Tr) pinpoints WHERE time is spent:
      Tq — request queued at HAProxy before connection attempt (maxconn / backend saturation)
      Tw — queued waiting for a free backend server slot (connection pool exhausted)
      Tc — TCP connection setup time (network / backend not accepting connections)
      Tr — backend application response time (mirrors T4 inbound latency)

    Also checks: HTTP 5xx count, client aborts, RPM, and TCP open connections (inbound/outbound).
    """
    _, thresholds, queries = _cfg()
    cluster = client.vars.get("cluster", "")
    ingress_thr = thresholds.get("ingress_health", {})

    def _render(key: str, window: str = "1m") -> str:
        return _render_fallback(queries[key], NAMESPACE=namespace, CLUSTER=cluster, WINDOW=window)

    try:
        (
            tq_frames, tw_frames, tc_frames, tr_frames,
            c5xx_frames, aborts_frames, rpm_frames,
            tcp_in_frames, tcp_out_frames,
        ) = await asyncio.gather(
            client.query_range(_render("ingress_latency_tq_ms"),    time_from, time_to),
            client.query_range(_render("ingress_latency_tw_ms"),    time_from, time_to),
            client.query_range(_render("ingress_latency_tc_ms"),    time_from, time_to),
            client.query_range(_render("ingress_latency_tr_ms"),    time_from, time_to),
            client.query_range(_render("ingress_5xx_count"),        time_from, time_to),
            client.query_range(_render("ingress_client_aborts", "5m"), time_from, time_to),
            client.query_range(_render("ingress_rpm"),              time_from, time_to),
            client.query_range(
                _render_fallback(queries["linkerd_tcp_open_inbound"],  NAMESPACE=namespace, WINDOW="1m"),
                time_from, time_to,
            ),
            client.query_range(
                _render_fallback(queries["linkerd_tcp_open_outbound"], NAMESPACE=namespace, WINDOW="1m"),
                time_from, time_to,
            ),
        )

        def _scalar_p99(frames: list) -> float | None:
            """Single-series histogram → P99 scalar (take max over window as spike indicator)."""
            raw = _parse_frames(frames)
            all_vals = [v for vals in raw.values() for v in vals if v is not None]
            return round(max(all_vals), 2) if all_vals else None

        def _scalar_sum_max(frames: list) -> float | None:
            """Counter/rate frames → take max value seen (spike indicator)."""
            raw = _parse_frames(frames)
            all_vals = [v for vals in raw.values() for v in vals if v is not None]
            return round(max(all_vals), 4) if all_vals else None

        def _per_pod_avg(frames: list) -> dict[str, float]:
            raw = _parse_frames(frames)
            return {k: round(sum(v) / len(v), 1) for k, v in raw.items() if v}

        tq_ms = _scalar_p99(tq_frames)
        tw_ms = _scalar_p99(tw_frames)
        tc_ms = _scalar_p99(tc_frames)
        tr_ms = _scalar_p99(tr_frames)

        # HAProxy histogram le boundaries at Myntra are in seconds — convert to ms
        tq_ms = round(tq_ms * 1000, 2) if tq_ms is not None else None
        tw_ms = round(tw_ms * 1000, 2) if tw_ms is not None else None
        tc_ms = round(tc_ms * 1000, 2) if tc_ms is not None else None
        tr_ms = round(tr_ms * 1000, 2) if tr_ms is not None else None

        c5xx_max  = _scalar_sum_max(c5xx_frames)    # max 5xx count in any 1-min window
        aborts_ps = _scalar_sum_max(aborts_frames)  # max abort rate (per second)
        rpm_max   = _scalar_sum_max(rpm_frames)     # max RPM in any 1-min window

        tcp_in_pods  = _per_pod_avg(tcp_in_frames)
        tcp_out_pods = _per_pod_avg(tcp_out_frames)

        # Dominant latency phase (what's eating the most time)
        phases = {
            "Tq": tq_ms, "Tw": tw_ms,
            "Tc": tc_ms, "Tr": tr_ms,
        }
        valid_phases = {k: v for k, v in phases.items() if v is not None}
        dominant_phase = max(valid_phases, key=lambda k: valid_phases[k]) if valid_phases else None

        thr = ingress_thr
        anomalies_found: list[str] = []
        if tq_ms is not None and tq_ms > thr.get("warn_tq_ms", 20):
            anomalies_found.append(f"Tq={tq_ms}ms (requests queuing at ingress — maxconn/saturation)")
        if tw_ms is not None and tw_ms > thr.get("warn_tw_ms", 20):
            anomalies_found.append(f"Tw={tw_ms}ms (backend server pool exhausted)")
        if tc_ms is not None and tc_ms > thr.get("warn_tc_ms", 5):
            anomalies_found.append(f"Tc={tc_ms}ms (slow TCP connect to backend — network issue)")
        if tr_ms is not None and tr_ms > thr.get("warn_tr_ms", 150):
            anomalies_found.append(f"Tr={tr_ms}ms (backend app response slow)")
        if c5xx_max is not None and c5xx_max > thr.get("warn_5xx_per_min", 5):
            anomalies_found.append(f"5xx_count={c5xx_max}/min")
        if aborts_ps is not None and aborts_ps > thr.get("warn_aborts_per_s", 0.5):
            anomalies_found.append(f"client_aborts={aborts_ps}/s (HAProxy dropping connections before response)")

        no_data = all(v is None for v in [tq_ms, tw_ms, tc_ms, tr_ms, c5xx_max, rpm_max])

        return {
            "latency_tq_ms": tq_ms,
            "latency_tw_ms": tw_ms,
            "latency_tc_ms": tc_ms,
            "latency_tr_ms": tr_ms,
            "dominant_phase": dominant_phase,
            "http_5xx_per_min": c5xx_max,
            "client_aborts_per_s": aborts_ps,
            "ingress_rpm": rpm_max,
            "tcp_open_inbound": tcp_in_pods,
            "tcp_open_outbound": tcp_out_pods,
            "anomalies": anomalies_found,
            "has_anomaly": bool(anomalies_found),
            "no_data": no_data,
        }
    except Exception as e:
        return {"error": str(e)}


# ─── Full dashboard snapshot ──────────────────────────────────────────────────

# Panel types that carry no PromQL data — skip them in the snapshot
_SKIP_PANEL_TYPES = {"row", "text", "news", "dashlist", "pluginlist", "alertlist", "logs"}

# Panels whose raw values are in seconds and need ×1000 → ms
_SECONDS_KEYWORDS = (
    "_seconds", "response_time_average", "request_duration_seconds",
    "duration_seconds",
)


async def fetch_all_panel_data(
    client: GrafanaClient,
    namespace: str,
    time_from: int,
    time_to: int,
) -> list[dict]:
    """
    Query every panel in the main service dashboard in parallel and return
    their raw series data. This gives the LLM a real-time, complete view of
    everything the dashboard shows — not just the metrics our specific tools cover.

    Each entry in the returned list is:
      {panel_id, title, section, series: {label: stats}, has_data, error?}
    """
    panels_cfg, _, _ = _cfg()
    dash_uid = panels_cfg["main_dashboard"]["uid"]

    # Ensure dashboard is cached
    try:
        await client.get_dashboard(dash_uid)
    except Exception as e:
        return [{"section": "ERROR", "title": "dashboard fetch failed", "error": str(e)}]

    flat_panels = client.iter_panels(dash_uid)

    # Variables to substitute in all panel expressions
    variables: dict[str, str] = {
        "namespace": namespace,
        "backend": namespace,
        **client.vars,
    }

    # Limit concurrent panel queries to avoid overwhelming Grafana
    semaphore = asyncio.Semaphore(15)

    async def _query_one(section: str, panel: dict) -> dict:
        panel_id = panel.get("id")
        title    = panel.get("title", f"Panel {panel_id}")
        ptype    = panel.get("type", "")

        if ptype in _SKIP_PANEL_TYPES:
            return {"panel_id": panel_id, "title": title, "section": section, "skip": True}

        targets = [
            t for t in panel.get("targets", [])
            if t.get("expr", "").strip() and not t.get("hide")
        ]
        if not targets:
            return {"panel_id": panel_id, "title": title, "section": section, "no_data": True}

        # For latency panels prefer p99 target; otherwise take first visible target
        def _pick(ts: list[dict]) -> dict:
            for t in ts:
                e = t.get("expr", "").lower()
                if "0.99" in e or '"p99"' in e:
                    return t
            return ts[0]

        target = _pick(targets)
        raw_expr = target.get("expr", "")
        expr = client.apply_variables(raw_expr, {**client.vars, **variables})
        ds_uid = client.panel_datasource_uid(panel)

        async with semaphore:
            try:
                frames = await client.query_range(expr, time_from, time_to, datasource_uid=ds_uid)
            except Exception:
                # If panel's datasource fails, retry with default
                try:
                    frames = await client.query_range(expr, time_from, time_to)
                except Exception as e2:
                    return {"panel_id": panel_id, "title": title, "section": section,
                            "error": str(e2)[:120]}

        series_raw = _parse_frames(frames)

        # Seconds → ms conversion for HAProxy gauge/histogram panels
        if any(kw in raw_expr for kw in _SECONDS_KEYWORDS):
            series_raw = {k: [v * 1000 for v in vals] for k, vals in series_raw.items()}

        # Only keep pods with actual values; limit to top 15 by max value to keep output compact
        series_stats: dict[str, dict] = {}
        for label, vals in series_raw.items():
            if vals:
                series_stats[label] = _stats(vals)

        top_series = dict(
            sorted(series_stats.items(),
                   key=lambda x: x[1].get("max") or 0, reverse=True)[:15]
        )

        return {
            "panel_id": panel_id,
            "title": title,
            "section": section,
            "panel_type": ptype,
            "series": top_series,
            "total_series": len(series_stats),
            "has_data": bool(top_series),
        }

    tasks = [_query_one(sec, p) for sec, p in flat_panels]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    output: list[dict] = []
    for (sec, panel), res in zip(flat_panels, raw_results):
        if isinstance(res, Exception):
            output.append({
                "panel_id": panel.get("id"),
                "title": panel.get("title", ""),
                "section": sec,
                "error": str(res)[:120],
            })
        else:
            output.append(res)

    return output


# ─── Tool schemas for the LLM ─────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "fetch_latency_at_hop",
        "description": (
            "Fetch p99 latency at a specific traffic hop. Use this first to determine "
            "scope (few pods vs all pods) and to localize which hop introduces latency. "
            "Compare T4 inbound vs T3 ingress: if T3 is fine but T4 is high, the problem "
            "is server-side. If T6 outbound is high, check downstream services/DBs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Deployment name (e.g. 'catalog-service')"},
                "namespace": {"type": "string", "description": "Kubernetes namespace"},
                "hop": {
                    "type": "string",
                    "enum": ["inbound", "outbound", "ingress"],
                    "description": "inbound=T4 server-linkerd→server, outbound=T2/T6, ingress=T3",
                },
                "granularity": {
                    "type": "string",
                    "enum": ["deployment", "pod"],
                    "description": "Use 'pod' to check if a subset of pods is affected",
                },
                "time_from": {"type": "integer", "description": "Start time in Unix milliseconds"},
                "time_to": {"type": "integer", "description": "End time in Unix milliseconds"},
            },
            "required": ["service", "namespace", "hop", "granularity", "time_from", "time_to"],
        },
    },
    {
        "name": "fetch_cpu_throttling",
        "description": (
            "Check CPU CFS throttling per pod. Returns throttle % per pod and a pattern "
            "(spiky/continuous/none). Spiky throttling → set limit=2x request. "
            "Continuous throttling → increase CPU request or add pods."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "namespace": {"type": "string"},
                "time_from": {"type": "integer"},
                "time_to": {"type": "integer"},
            },
            "required": ["service", "namespace", "time_from", "time_to"],
        },
    },
    {
        "name": "fetch_rpm_distribution",
        "description": (
            "Check requests-per-minute distribution across pods. "
            "Uneven distribution (max > 2x min) with ≥3 pods indicates the upstream caller "
            "or load balancer is routing traffic unevenly — e.g. HTTP keep-alive connections "
            "routing all requests on a persistent connection to one pod."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "namespace": {"type": "string"},
                "time_from": {"type": "integer"},
                "time_to": {"type": "integer"},
            },
            "required": ["service", "namespace", "time_from", "time_to"],
        },
    },
    {
        "name": "fetch_pod_restarts",
        "description": (
            "Count pod restarts in the time window. Restarts cause transient latency "
            "spikes as the pod IP changes and endpoint propagation takes a few seconds. "
            "Usually self-heals; persistent restarts may need pod count increase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "namespace": {"type": "string"},
                "time_from": {"type": "integer"},
                "time_to": {"type": "integer"},
            },
            "required": ["service", "namespace", "time_from", "time_to"],
        },
    },
    {
        "name": "fetch_worker_node_metrics",
        "description": (
            "Fetch VM-level metrics for a specific worker node. "
            "Use after identifying the node from the 'Pod IP and worker node mapping' panel. "
            "Detects noisy-neighbour: one container consuming excess disk/CPU/network "
            "that throttles all other containers on the same VM."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_ip": {"type": "string", "description": "Worker node IP (from Pod IP and worker node mapping panel)"},
                "time_from": {"type": "integer"},
                "time_to": {"type": "integer"},
            },
            "required": ["node_ip", "time_from", "time_to"],
        },
    },
    {
        "name": "fetch_dns_latency",
        "description": (
            "Check CoreDNS p99 latency. High DNS latency affects all service calls "
            "that use DNS names. Forward DNS timeout or reverse-DNS with TCP-disabled "
            "load balancer are the two main causes. Both require SRE escalation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "time_from": {"type": "integer"},
                "time_to": {"type": "integer"},
            },
            "required": ["time_from", "time_to"],
        },
    },
    {
        "name": "fetch_disk_iops",
        "description": (
            "Check disk IOPS per pod and IOPS-per-CPU-core ratio. "
            "Worker nodes have a hard 2300 IOPS cap shared across all containers. "
            "Per-core limit is ~12 IOPS. Exceeding this causes disk-wait latency, "
            "especially for apps doing synchronous log writes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "namespace": {"type": "string"},
                "time_from": {"type": "integer"},
                "time_to": {"type": "integer"},
            },
            "required": ["service", "namespace", "time_from", "time_to"],
        },
    },
    {
        "name": "fetch_error_rate",
        "description": (
            "Fetch Linkerd inbound success rate, error rate, and total RPS for the service. "
            "Returns success_rate_pct, error_rate_pct, total_rps, and error breakdown by status code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "namespace": {"type": "string"},
                "time_from": {"type": "integer"},
                "time_to": {"type": "integer"},
            },
            "required": ["service", "namespace", "time_from", "time_to"],
        },
    },
    {
        "name": "fetch_memory_utilization",
        "description": (
            "Check per-pod memory utilization as % of memory limit. "
            "Flags pods at warn (≥80%) or critical (≥90%) to catch OOM risk before a pod restart occurs. "
            "Returns per-pod working_set_mb, limit_mb, utilization_pct."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "namespace": {"type": "string"},
                "time_from": {"type": "integer"},
                "time_to": {"type": "integer"},
            },
            "required": ["service", "namespace", "time_from", "time_to"],
        },
    },
    {
        "name": "fetch_replica_count",
        "description": (
            "Check deployment replica health: available vs desired vs unavailable. "
            "Unavailable replicas indicate HPA scale-up, scheduling failures, or CrashLoops. "
            "Returns available, desired, unavailable counts and is_degraded flag."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "namespace": {"type": "string"},
                "time_from": {"type": "integer"},
                "time_to": {"type": "integer"},
            },
            "required": ["service", "namespace", "time_from", "time_to"],
        },
    },
    {
        "name": "fetch_hpa_status",
        "description": (
            "Check HPA scaling behaviour during the window. "
            "Detects scale-up events (new pods warming up raise transient latency) "
            "and whether HPA hit its configured max limit (cannot scale further). "
            "Returns was_scaling, hit_max_limit, peak_desired, hpa_max, cpu_utilization_pct."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service":    {"type": "string"},
                "namespace":  {"type": "string"},
                "time_from":  {"type": "integer"},
                "time_to":    {"type": "integer"},
            },
            "required": ["service", "namespace", "time_from", "time_to"],
        },
    },
    {
        "name": "fetch_ingress_health",
        "description": (
            "Comprehensive HAProxy ingress analysis: latency breakdown (Tq/Tw/Tc/Tr), "
            "HTTP 5xx count, client aborts, RPM, and TCP open connections. "
            "Tq=request queue wait (maxconn/saturation), Tw=backend pool queue, "
            "Tc=TCP connect time, Tr=backend app response. "
            "Use alongside fetch_latency_at_hop(ingress) for complete ingress picture."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service":   {"type": "string"},
                "namespace": {"type": "string"},
                "time_from": {"type": "integer"},
                "time_to":   {"type": "integer"},
            },
            "required": ["service", "namespace", "time_from", "time_to"],
        },
    },
]
