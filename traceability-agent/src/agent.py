"""
ReAct-free diagnostic agent for K8s service health.

Strategy: run all diagnostic tools in parallel upfront, then make a single
LLM call to interpret the data and write the report. No iterative tool-calling.
This is far more reliable with local models (Ollama) and faster with cloud models.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

from .grafana_client import GrafanaClient
from .llm_client import AnthropicLLM, create_llm
from .prompts import SYSTEM_PROMPT
from .tools import (
    fetch_all_panel_data,
    fetch_cpu_throttling,
    fetch_disk_iops,
    fetch_dns_latency,
    fetch_error_rate,
    fetch_hpa_status,
    fetch_ingress_health,
    fetch_latency_at_hop,
    fetch_memory_utilization,
    fetch_pod_restarts,
    fetch_replica_count,
    fetch_rpm_distribution,
    fetch_worker_node_metrics,
)


_TOOL_FNS = {
    "fetch_latency_at_hop": fetch_latency_at_hop,
    "fetch_cpu_throttling": fetch_cpu_throttling,
    "fetch_rpm_distribution": fetch_rpm_distribution,
    "fetch_pod_restarts": fetch_pod_restarts,
    "fetch_worker_node_metrics": fetch_worker_node_metrics,
    "fetch_dns_latency": fetch_dns_latency,
    "fetch_disk_iops": fetch_disk_iops,
    "fetch_error_rate": fetch_error_rate,
    "fetch_memory_utilization": fetch_memory_utilization,
    "fetch_replica_count": fetch_replica_count,
    "fetch_hpa_status": fetch_hpa_status,
    "fetch_ingress_health": fetch_ingress_health,
}


def _val(v: Any, unit: str = "") -> str:
    if v is None:
        return "NO DATA"
    return f"{v}{unit}"


def _extract_result(tool_calls: list[dict], results: list[Any], name: str, hop: str = "") -> dict | None:
    for tc, result in zip(tool_calls, results):
        if tc["name"] == name and tc["input"].get("hop", "") == hop:
            return result if isinstance(result, dict) and "error" not in result else None
    return None


def _build_diagnosis(
    tool_calls: list[dict],
    results: list[Any],
    anomalies: list[str],
    service: str,
    namespace: str,
) -> dict:
    """
    Derive root cause and action plan from the anomalies that were actually detected.
    Only diagnoses what is flagged — does not invent a problem when nothing is wrong.
    """
    inbound  = _extract_result(tool_calls, results, "fetch_latency_at_hop", "inbound")
    outbound = _extract_result(tool_calls, results, "fetch_latency_at_hop", "outbound")
    ingress  = _extract_result(tool_calls, results, "fetch_latency_at_hop", "ingress")
    cpu      = _extract_result(tool_calls, results, "fetch_cpu_throttling")
    rpm      = _extract_result(tool_calls, results, "fetch_rpm_distribution")
    restarts = _extract_result(tool_calls, results, "fetch_pod_restarts")
    dns      = _extract_result(tool_calls, results, "fetch_dns_latency")
    iops     = _extract_result(tool_calls, results, "fetch_disk_iops")
    hpa      = _extract_result(tool_calls, results, "fetch_hpa_status")
    ingress_health = _extract_result(tool_calls, results, "fetch_ingress_health")

    root_cause = ""
    steps: list[str] = []
    secondary_steps: list[str] = []
    escalate = False
    escalate_reason = "N/A"

    # Which anomaly types are actually present
    has_rpm      = any("RPM IMBALANCE"   in a for a in anomalies)
    has_inbound  = any("T4" in a for a in anomalies)
    has_outbound = any("T6" in a for a in anomalies)
    has_ingress  = any("T3" in a for a in anomalies)
    has_cpu      = any("CPU THROTTLE"    in a for a in anomalies)
    has_restart  = any("POD RESTARTS"    in a for a in anomalies)
    has_dns      = any("T7" in a or "DNS" in a for a in anomalies)
    has_iops     = any("DISK IOPS"       in a for a in anomalies)
    has_errors   = any("ERROR RATE"      in a for a in anomalies)
    has_hpa_max  = any("HPA MAX LIMIT"   in a for a in anomalies)
    has_hpa_scale = any("HPA SCALING"    in a for a in anomalies)
    has_oom      = any("OOM EVENTS"      in a for a in anomalies)
    has_ingress_detail = any("INGRESS DETAIL" in a for a in anomalies)

    # ── No anomalies → service is healthy ────────────────────────────────────
    if not anomalies:
        note = ""
        if rpm and rpm.get("uneven"):
            ratio = rpm.get("imbalance_ratio", 0)
            note = (
                f" Note: RPM distribution shows a {ratio}x skew across pods — not causing "
                f"visible latency issues now, but could worsen under heavier traffic."
            )
        return {
            "root_cause": (
                f"No active performance issue detected. All signals are within normal ranges.{note}"
            ),
            "steps": [],
            "secondary_steps": [],
            "escalate": False,
            "escalate_reason": "N/A",
        }

    # Helper: build secondary-finding steps for anomalies NOT chosen as root cause
    def _secondary(skip_rule: str) -> list[str]:
        extra: list[str] = []
        if "IOPS" not in skip_rule and has_iops and iops:
            ipc = iops.get("iops_per_core")
            extra.append(
                f"[DISK IOPS — {ipc} IOPS/core, limit 12]\n"
                f"  The hot pod is also doing synchronous I/O in the request path.\n"
                f"  Switch to async logging:\n"
                f"    Python: logging.handlers.QueueHandler with a background flush thread\n"
                f"    Java:   Log4j AsyncAppender or Logback AsyncAppender\n"
                f"  After deploying, IOPS/core should drop below 12 in the Grafana panel."
            )
        if "CPU" not in skip_rule and has_cpu and cpu:
            pattern = cpu.get("dominant_pattern", "")
            extra.append(
                f"[CPU THROTTLE — {pattern}]\n"
                f"  kubectl set resources deployment/{service} -n {namespace} "
                f"--limits=cpu=<2×_current_cpu_request>"
            )
        if "RESTART" not in skip_rule and has_restart and restarts:
            total = restarts.get("total_restarts", 0)
            extra.append(
                f"[POD RESTARTS — {total} restart(s)]\n"
                f"  kubectl describe pods -n {namespace} -l app={service} | grep -A 8 'Last State'"
            )
        if "DNS" not in skip_rule and has_dns and dns:
            p99 = dns.get("latency_ms", {}).get("p99")
            extra.append(
                f"[DNS LATENCY — p99={p99}ms]\n"
                f"  Use fully-qualified names (svc.namespace.svc.cluster.local). "
                f"If still critical, escalate to SRE."
            )
        if "ERROR" not in skip_rule and has_errors:
            extra.append(
                f"[ERROR RATE elevated]\n"
                f"  kubectl logs -n {namespace} -l app={service} --since=30m "
                f"| grep -iE '5[0-9][0-9]|error|exception'"
            )
        return extra

    # ── Rule 1: RPM imbalance ─────────────────────────────────────────────────
    if has_rpm:
        pod_rpms = rpm.get("pods", {}) if rpm else {}
        # Work only with prod pods
        prod_rpms = {k: v for k, v in pod_rpms.items() if not _is_debug_pod(k)}
        hot_pod_label = max(prod_rpms, key=lambda k: prod_rpms[k]) if prod_rpms else None
        hot_pod       = _pod_name(hot_pod_label) if hot_pod_label else None
        max_rpm       = prod_rpms[hot_pod_label] if hot_pod_label else rpm.get("max_rpm")
        min_rpm_label = min(prod_rpms, key=lambda k: prod_rpms[k]) if prod_rpms else None
        min_rpm       = prod_rpms[min_rpm_label] if min_rpm_label else rpm.get("min_rpm")
        ratio         = round(max_rpm / min_rpm, 1) if (max_rpm and min_rpm and min_rpm > 0) else rpm.get("imbalance_ratio", 0)

        slow_pods = [_pod_name(p) for p in (inbound.get("anomaly_pods", []) if inbound else []) if not _is_debug_pod(p)]
        hot_is_also_slow = hot_pod and any(
            hot_pod.endswith(sp.split("-")[-1]) or sp.endswith(hot_pod.split("-")[-1])
            for sp in slow_pods
        )
        cold_but_slow = [p for p in slow_pods if not (
            hot_pod and (hot_pod.endswith(p.split("-")[-1]) or p.endswith(hot_pod.split("-")[-1]))
        )]

        hot_note = f"Pod {hot_pod} is receiving {max_rpm}rpm. " if hot_pod else ""
        confirm_note = (
            f"Its inbound latency is also elevated — confirming it is overloaded. "
            if hot_is_also_slow else ""
        )
        secondary_note = (
            f"Pod(s) {', '.join(cold_but_slow)} are also slow despite low RPM — "
            f"see ALSO ADDRESS below for contributing factors. "
            if cold_but_slow else ""
        )

        root_cause = (
            f"Extreme load imbalance across {len(prod_rpms)} prod pods (ratio={ratio}x, "
            f"max={max_rpm}rpm vs min={min_rpm}rpm). "
            f"{hot_note}{confirm_note}"
            f"Traffic is not being distributed evenly — the upstream caller or load balancer "
            f"is routing a disproportionate share of requests to specific pods. "
            f"{secondary_note}"
        ).strip()

        hot_pod_check = (
            f"\n    Confirm the overloaded pod (read-only, DS team can run):\n"
            f"    kubectl top pod {hot_pod} -n {namespace}"
            if hot_pod else ""
        )
        steps = [
            f"DS team: identify which upstream service or client is sending traffic to {service}.\n"
            f"    Check the Grafana / Linkerd Viz service map or ask the SRE team.{hot_pod_check}",
            f"Ask SRE to investigate the load balancing configuration for the upstream caller\n"
            f"    and apply the appropriate fix (connection pooling, Linkerd injection, or\n"
            f"    session affinity settings depending on the caller's protocol).",
            f"Verify: watch RPM distribution panel — ratio should drop to ≤ 2x within minutes.",
        ]
        escalate = True
        escalate_reason = "SRE must diagnose the upstream caller's load balancing configuration and apply the fix."
        secondary_steps = _secondary("RPM")

    # ── Rule 2: Outbound ~5s → Linkerd protocol detection timeout ────────────
    elif has_outbound:
        series = outbound.get("series", {}) if outbound else {}
        linkerd_timeout_pod = None
        for pod, stats in series.items():
            p99 = stats.get("p99") or 0
            if 4000 <= p99 <= 6000:
                linkerd_timeout_pod = (pod, p99)
                break

        if linkerd_timeout_pod:
            _, p99 = linkerd_timeout_pod
            root_cause = (
                f"Linkerd protocol detection timeout on outbound (~{p99}ms). "
                f"The port is not in opaque-ports, so Linkerd waits up to 5s on every new "
                f"TCP connection to detect HTTP/1 vs HTTP/2 vs opaque."
            )
            escalate = True
            escalate_reason = "SRE must annotate the service with opaque-ports to fix the Linkerd protocol detection timeout."
            steps = [
                f"DS team: find the port number (read-only):\n"
                f"    kubectl get svc {service} -n {namespace}",
                f"Ask SRE to annotate the service with the port found above (no restart needed):\n"
                f"    kubectl annotate service {service} -n {namespace} \\\n"
                f"      linkerd.io/opaque-ports='<port_number>' --overwrite",
                f"Watch T6 outbound p99 — should drop from ~5000ms to <100ms within 60 seconds.",
                f"Permanent fix: add to Helm values.yaml under service.annotations:\n"
                f"  linkerd.io/opaque-ports: \"<port_number>\"",
            ]
        else:
            top_p99 = max((s.get("p99") or 0 for s in series.values()), default=0)
            root_cause = (
                f"Outbound (T6) latency elevated ({top_p99}ms) — the downstream service, "
                f"database, or cache that {service} calls is responding slowly. "
                f"The problem is NOT inside {service} itself."
            )
            steps = [
                f"Identify which downstream {service} is calling (check the service map or outbound Linkerd panel).",
                "Run the K8s debug agent against the downstream service's dashboard to diagnose it.",
                "Common causes: DB connection pool exhausted, cache cold start, downstream deployment rollout.",
            ]
        secondary_steps = _secondary("OUTBOUND")

    # ── Rule 3: Ingress elevated + all pods elevated → ingress saturation ─────
    elif has_ingress and has_inbound and inbound and inbound.get("all_pods_elevated"):
        ingress_p99 = max(
            (s.get("p99") or 0 for s in (ingress or {}).get("series", {}).values()), default=0
        )
        dom_phase = (ingress_health or {}).get("dominant_phase")
        dom_note = ""
        if dom_phase == "Tq":
            dom_note = (
                f" HAProxy latency breakdown shows Tq={ingress_health.get('latency_tq_ms')}ms "
                f"— requests are queuing at the ingress (maxconn limit hit or all backends saturated). "
                f"This is an ingress-layer bottleneck, NOT an application bug."
            )
        elif dom_phase == "Tw":
            dom_note = (
                f" Breakdown shows Tw={ingress_health.get('latency_tw_ms')}ms "
                f"— HAProxy is queuing requests waiting for a free backend server slot. "
                f"The pod pool is exhausted. Scale up pods."
            )
        elif dom_phase == "Tc":
            dom_note = (
                f" Breakdown shows Tc={ingress_health.get('latency_tc_ms')}ms "
                f"— TCP connection setup to backend is slow. Possible network issue or pod not ready."
            )
        elif dom_phase == "Tr":
            dom_note = (
                f" Breakdown shows Tr={ingress_health.get('latency_tr_ms')}ms "
                f"— backend application response is slow. Problem is inside {service} pods."
            )
        root_cause = (
            f"Ingress is elevated (T3 p99={ingress_p99}ms) and ALL pods show elevated inbound.{dom_note}"
        )
        escalate = True
        escalate_reason = "Ingress controller scaling or backend pool increase requires SRE access."
        steps = [
            "ESCALATE TO SRE — provide this report, the time window, and whether Tq or Tw is the dominant phase.",
        ]
        secondary_steps = _secondary("INGRESS")

    # ── Rule 4: CPU throttle ──────────────────────────────────────────────────
    elif has_cpu:
        pattern = cpu["dominant_pattern"]
        pods = cpu.get("pods", {})
        affected = sum(1 for s in pods.values() if s.get("pattern") == pattern)
        escalate = True
        if pattern == "spiky":
            root_cause = (
                f"CPU CFS throttle bursts on {affected}/{len(pods)} pods. "
                f"CPU limit is too close to CPU request — no burst headroom within the CFS period. "
                f"GC pauses or bursty request handling momentarily hit the limit and stall the container."
            )
            escalate_reason = f"SRE must update CPU limits on deployment/{service} in namespace {namespace}."
            steps = [
                f"DS team: check current CPU resources (read-only):\n"
                f"    kubectl get deploy {service} -n {namespace} "
                f"-o jsonpath='{{.spec.template.spec.containers[0].resources}}'",
                f"Ask SRE to increase the CPU limit to 2× the current CPU request:\n"
                f"    kubectl set resources deployment/{service} -n {namespace} "
                f"--limits=cpu=<2×_current_cpu_request>",
                "Throttle % should drop near-zero in Grafana CPU throttle panel within 5 minutes.",
                f"Permanent fix: update resources.limits.cpu in Helm values.yaml.",
            ]
        else:
            root_cause = (
                f"Continuous CPU throttling on {affected}/{len(pods)} pods. "
                f"Steady-state load exceeds the CPU request — the pod is under-provisioned."
            )
            escalate_reason = f"SRE must update CPU requests on deployment/{service} in namespace {namespace}."
            steps = [
                f"DS team: check current CPU resources (read-only):\n"
                f"    kubectl get deploy {service} -n {namespace} "
                f"-o jsonpath='{{.spec.template.spec.containers[0].resources}}'",
                f"Ask SRE to increase the CPU request:\n"
                f"    kubectl set resources deployment/{service} -n {namespace} "
                f"--requests=cpu=<higher_value>",
                "Average throttle % should drop below 5%. Monitor for 10 minutes after the change.",
                f"Permanent fix: update resources.requests.cpu in Helm values.yaml.",
            ]
        secondary_steps = _secondary("CPU")

    # ── Rule 5: Pod restarts ──────────────────────────────────────────────────
    elif has_restart:
        total = restarts.get("total_restarts", 0) if restarts else 0
        oom_count = restarts.get("total_oom_events", 0) if restarts else 0
        oom_note = f" OOM kills: {oom_count}." if oom_count else ""
        root_cause = (
            f"{total} pod restart(s) in the window.{oom_note} Each restart causes 2–5s of request failures "
            f"while kube-proxy/Linkerd endpoint update propagates the new pod IP."
        )
        escalate = True
        escalate_reason = f"SRE must investigate restart reason and update resources or probe thresholds on deployment/{service}."
        steps = [
            f"DS team: check restart reason (read-only):\n"
            f"    kubectl describe pods -n {namespace} -l app={service} | grep -A 8 'Last State'",
            f"If OOMKill — ask SRE to increase memory limit:\n"
            f"    kubectl set resources deployment/{service} -n {namespace} --limits=memory=<higher>",
            f"If liveness probe fail — ask SRE to review probe initialDelaySeconds / failureThreshold.",
            "Permanent fix: update the relevant resource or probe setting in Helm values.yaml.",
        ]
        secondary_steps = _secondary("RESTART")

    # ── Rule 6: DNS critical ──────────────────────────────────────────────────
    elif has_dns:
        p99 = dns.get("latency_ms", {}).get("p99") if dns else None
        root_cause = (
            f"CoreDNS p99 is critical ({p99}ms, threshold=50ms). "
            f"DNS resolution delays affect every outbound service call that uses a DNS name."
        )
        escalate = True
        escalate_reason = "CoreDNS capacity issue requires cluster-level access."
        steps = [
            "ESCALATE TO SRE — provide this metric data and the time window.",
            "Workaround: ensure services use fully-qualified DNS names "
            "(e.g. service.namespace.svc.cluster.local) to avoid ndots=5 multi-step resolution.",
        ]
        secondary_steps = _secondary("DNS")

    # ── Rule 7: Disk IOPS ─────────────────────────────────────────────────────
    elif has_iops:
        ipc = iops.get("iops_per_core") if iops else None
        slow_pods_list = [_pod_name(p) for p in (inbound.get("anomaly_pods", []) if inbound else [])]
        root_cause = (
            f"Disk IOPS/core ({ipc}) exceeds the limit (12 IOPS/core on Azure P20). "
            f"Synchronous I/O in the request path (likely log writes) is saturating "
            f"the node's shared IOPS budget."
            + (f" Affected pods: {', '.join(slow_pods_list)}." if slow_pods_list else "")
        )
        escalate = True
        escalate_reason = f"SRE must restart deployment/{service} after the DS team applies the async logging config change."
        steps = [
            "DS team: switch application logging to async mode (code/config change — no kubectl needed):\n"
            "  Java: Log4j AsyncAppender or logback AsyncAppender\n"
            "  Python: logging.handlers.QueueHandler with a background flush thread",
            f"Ask SRE to restart the deployment to pick up the new logging config:\n"
            f"    kubectl rollout restart deployment/{service} -n {namespace}",
            "Watch IOPS/core panel in Grafana — should drop below 12 within 5 minutes of restart.",
        ]
        secondary_steps = _secondary("IOPS")

    # ── Rule HPA: HPA hit max limit ──────────────────────────────────────────
    elif has_hpa_max or has_hpa_scale:
        hpa_max_val = hpa.get("hpa_max") if hpa else None
        peak = hpa.get("peak_desired") if hpa else None
        if has_hpa_max:
            root_cause = (
                f"HPA hit its configured max limit of {hpa_max_val} replicas. "
                f"CPU/memory utilization triggered scale-up but no more pods can be added. "
                f"The cluster cannot absorb the current traffic volume — latency rises as pods stay overloaded."
            )
            escalate = True
            escalate_reason = f"SRE must raise hpa_max in values.yaml and redeploy, or add cluster capacity."
            steps = [
                f"DS team: confirm HPA limits (read-only):\n"
                f"    kubectl get hpa -n {namespace}",
                f"Ask SRE to increase maxReplicas in Helm values.yaml for {service} and redeploy.",
                "Watch Grafana HPA panel — desired should drop below new max within minutes.",
            ]
        else:
            root_cause = (
                f"HPA scaled up to {peak} pods during the window. "
                f"New pods warming up (JVM JIT, cache cold start) temporarily reduce effective capacity "
                f"and can explain transient latency spikes."
            )
            steps = [
                "Monitor: latency should self-resolve as pods finish warming up (typically 1–3 minutes).",
                f"If spikes are frequent, ask SRE to increase HPA minReplicas for {service} "
                f"so the fleet pre-scales before traffic ramps.",
            ]
        secondary_steps = _secondary("HPA")

    # ── Rule 8: Inbound latency elevated, no corroborating signal found ───────
    elif has_inbound or has_ingress or has_errors:
        slow_pods_list = [_pod_name(p) for p in (inbound.get("anomaly_pods", []) if inbound else [])]
        pods_str = ", ".join(slow_pods_list) if slow_pods_list else "<elevated pods>"
        root_cause = (
            f"Latency anomaly on pod(s) {pods_str} with no corroborating signal "
            f"(CPU, RPM, restarts, DNS, IOPS all within normal range). "
            f"Likely a transient spike, a dependency not covered by these metrics, or a recent deployment."
        )
        steps = [
            f"Check logs on the affected pod(s):\n"
            + "\n".join(
                f"    kubectl logs -n {namespace} {p} --since=30m "
                f"| grep -iE 'error|slow|timeout|warn|exception'"
                for p in (slow_pods_list or [f"-l app={service}"])
            ),
            f"Check for recent deployments:\n"
            f"    kubectl rollout history deployment/{service} -n {namespace}",
            "If the spike has already cleared: treat as transient and set up an alert. "
            "If it persists: open an incident with this report and the Grafana links.",
        ]
        secondary_steps = _secondary("INBOUND")

    # ── Fallback ──────────────────────────────────────────────────────────────
    else:
        root_cause = (
            "Anomalies detected but no specific root cause matched by the playbook. "
            "Review the WHAT'S WRONG section and check application logs."
        )
        steps = [
            f"kubectl logs -n {namespace} -l app={service} --since=10m "
            f"| grep -iE 'error|slow|timeout|warn|exception'",
        ]
        secondary_steps = _secondary("")

    return {
        "root_cause": root_cause,
        "steps": steps,
        "secondary_steps": secondary_steps,
        "escalate": escalate,
        "escalate_reason": escalate_reason,
    }


def _pod_name(label: str) -> str:
    """Extract pod name from a Prometheus label string like 'pod=full-name,...'"""
    for part in label.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip() in ("pod", "kubernetes_pod_name"):
                return v.strip()
    return label


def _is_debug_pod(label: str) -> bool:
    """Return True if the pod name contains 'debug' — these pods are excluded from analysis."""
    return "debug" in _pod_name(label).lower()


def _detect_anomalies(tool_calls: list[dict], results: list[Any]) -> list[str]:
    """
    Pre-compute which signals are anomalous so the LLM doesn't have to discover them.
    Returns a list of plain-text anomaly descriptions, each referencing exact values.
    """
    anomalies: list[str] = []

    # Get authoritative desired pod count up front.
    # Rolling deployments produce one series per pod ever seen in the window — the raw
    # series count grossly overstates fleet size (e.g. 102 series for a 40-pod service).
    desired_replicas: int | None = None
    for tc, result in zip(tool_calls, results):
        if tc["name"] == "fetch_replica_count" and isinstance(result, dict) and "error" not in result:
            d = result.get("desired")
            if d is not None and int(d) > 0:
                desired_replicas = int(d)
            break

    for tc, result in zip(tool_calls, results):
        if not isinstance(result, dict) or "error" in result:
            continue
        name = tc["name"]
        hop  = tc["input"].get("hop", "")

        if name == "fetch_latency_at_hop":
            series = result.get("series", {})
            # Exclude debug pods from latency analysis
            prod_series = {k: v for k, v in series.items() if not _is_debug_pod(k)}
            anomaly_pods = [p for p in result.get("anomaly_pods", []) if not _is_debug_pod(p)]
            all_elevated = result.get("all_pods_elevated", False)

            # Active prod pods: have p99 data in the window AND are not debug pods.
            # Compute directly here — never trust active_pod_count from the tool since
            # it may have been computed before debug filtering was applied.
            active_prod = {k: v for k, v in prod_series.items() if v.get("p99") is not None}
            active_count = len(active_prod)

            # Display denominator: prefer authoritative desired replicas; fall back to
            # active series count (pods with data in window); last resort: all series.
            display_total = (
                desired_replicas
                if desired_replicas and desired_replicas >= len(anomaly_pods)
                else active_count or len(prod_series)
            )

            _HOP_LABEL = {
                "inbound":  "T4 (server-linkerd → server)",
                "outbound": "T6 (server-linkerd → other-services/DB)",
                "ingress":  "T3 (ingress → server-linkerd)",
            }
            hop_label = _HOP_LABEL.get(hop, hop.upper())

            if all_elevated:
                top_val = max((v.get("p99") or 0 for v in active_prod.values()), default=0)
                anomalies.append(
                    f"[{hop_label}] ALL {display_total} backends elevated — "
                    f"peak p99={top_val}ms  (all_pods_elevated=True)"
                )
            elif anomaly_pods:
                top = sorted(
                    [(pod, prod_series[pod].get("p99") or 0) for pod in anomaly_pods if pod in prod_series],
                    key=lambda x: x[1], reverse=True
                )
                detail = ", ".join(f"{_pod_name(p)}={v}ms" for p, v in top[:4])
                anomalies.append(
                    f"[{hop_label}] {len(anomaly_pods)}/{display_total} backends with p99 spike — "
                    f"{detail}"
                )

        elif name == "fetch_error_rate":
            err = result.get("error_rate_pct")
            if err is not None and err > 1.0:
                sr = result.get("success_rate_pct")
                anomalies.append(
                    f"[ERROR RATE] Elevated — error={err}%  success={sr}%"
                )

        elif name == "fetch_cpu_throttling":
            pattern = result.get("dominant_pattern", "none")
            if pattern in ("spiky", "continuous"):
                pods = result.get("pods", {})
                affected = sum(1 for s in pods.values() if s.get("pattern") == pattern)
                anomalies.append(
                    f"[CPU THROTTLE] Pattern={pattern} on {affected}/{len(pods)} pods"
                )

        elif name == "fetch_rpm_distribution":
            # Recalculate ratio using only prod pods — debug pods naturally sit idle
            # and would grossly inflate the imbalance ratio.
            all_pods = result.get("pods", {})
            prod_pods = {k: v for k, v in all_pods.items() if not _is_debug_pod(k)}
            if len(prod_pods) >= 2:
                prod_max = max(prod_pods.values())
                prod_min = min(prod_pods.values())
                ratio = round(prod_max / prod_min, 1) if prod_min > 0 else prod_max
                if ratio >= 10:
                    anomalies.append(
                        f"[RPM IMBALANCE] Extreme load skew across prod pods — ratio={ratio}x  "
                        f"(max={prod_max}rpm vs min={prod_min}rpm)"
                    )

        elif name == "fetch_pod_restarts":
            total = result.get("total_restarts", 0)
            if total and total > 0:
                anomalies.append(f"[POD RESTARTS] {total} restart(s) in window")
            total_oom = result.get("total_oom_events", 0)
            if total_oom and total_oom > 0:
                reasons = result.get("restart_reasons", {})
                reason_str = ", ".join(set(reasons.values())) if reasons else "unknown"
                anomalies.append(
                    f"[OOM EVENTS] {total_oom} OOM kill(s) in window — reason(s): {reason_str}"
                )

        elif name == "fetch_dns_latency":
            status = result.get("status", "unknown")
            if status in ("warn", "critical"):
                p99 = result.get("latency_ms", {}).get("p99")
                anomalies.append(f"[T7 — DNS resolution] {status.upper()} — p99={p99}ms (warn>10ms, critical>50ms)")

        elif name == "fetch_disk_iops":
            if result.get("exceeds_threshold"):
                ipc = result.get("iops_per_core")
                anomalies.append(
                    f"[DISK IOPS] Exceeds limit — {ipc} IOPS/core (limit=12 IOPS/core)"
                )

        elif name == "fetch_worker_node_metrics":
            for b in result.get("threshold_breaches", []):
                anomalies.append(
                    f"[NODE {b['metric'].upper()}] {b['pct_used']}% of limit "
                    f"({b['p90']} vs limit {b['limit']})"
                )

        elif name == "fetch_memory_utilization":
            crit = result.get("critical_count", 0)
            warn = result.get("warn_count", 0)
            max_pct = result.get("max_utilization_pct")
            if crit > 0:
                anomalies.append(
                    f"[MEMORY] {crit} pod(s) at critical utilization — "
                    f"max={max_pct}% (threshold≥{result.get('critical_pct_threshold', 90)}%)"
                )
            elif warn > 0:
                anomalies.append(
                    f"[MEMORY] {warn} pod(s) at elevated utilization — "
                    f"max={max_pct}% (warn threshold≥{result.get('warn_pct_threshold', 80)}%)"
                )

        elif name == "fetch_hpa_status":
            if result.get("hit_max_limit"):
                hpa_max = result.get("hpa_max")
                peak = result.get("peak_desired")
                anomalies.append(
                    f"[HPA MAX LIMIT] HPA hit configured max={hpa_max} replicas "
                    f"(peak_desired={peak}) — cannot scale further"
                )
            elif result.get("was_scaling"):
                peak = result.get("peak_desired")
                curr = result.get("current_desired")
                anomalies.append(
                    f"[HPA SCALING] Scale-up event during window — "
                    f"desired grew to peak={peak}, now={curr} "
                    f"(new pods warming up may explain transient latency)"
                )

        elif name == "fetch_replica_count":
            if result.get("is_degraded"):
                avail     = result.get("available")
                desired   = result.get("desired")
                unavail   = result.get("unavailable")
                min_avail = result.get("min_available")
                dip_note  = (
                    f", dipped to min={min_avail} during window"
                    if min_avail is not None and min_avail != avail else ""
                )
                anomalies.append(
                    f"[REPLICAS] Degraded — available={avail} of desired={desired}, "
                    f"unavailable_max={unavail}{dip_note}"
                )

        elif name == "fetch_ingress_health":
            ing_anomalies = result.get("anomalies", [])
            if ing_anomalies:
                dom = result.get("dominant_phase", "?")
                detail = "; ".join(ing_anomalies[:3])
                anomalies.append(
                    f"[INGRESS DETAIL] HAProxy ingress anomalies (dominant={dom}): {detail}"
                )

    # Count how many non-node tools returned actual data vs empty
    # (node metrics require a specific IP so skip them in this census)
    _no_data_count = 0
    _data_count = 0
    for tc, result in zip(tool_calls, results):
        if not isinstance(result, dict) or "error" in result:
            continue
        n = tc["name"]
        if n == "fetch_worker_node_metrics":
            continue
        has_data = (
            (n == "fetch_latency_at_hop"       and bool(result.get("series")))
            or (n == "fetch_error_rate"         and result.get("success_rate_pct") is not None)
            or (n == "fetch_cpu_throttling"     and not result.get("no_data"))
            or (n == "fetch_rpm_distribution"   and bool(result.get("pods")))
            or (n == "fetch_pod_restarts"       and result.get("has_data", False))
            or (n == "fetch_dns_latency"        and result.get("status", "unknown") != "unknown")
            or (n == "fetch_disk_iops"          and not result.get("no_data"))
            or (n == "fetch_memory_utilization" and bool(result.get("pods")))
            or (n == "fetch_replica_count"      and result.get("available") is not None)
            or (n == "fetch_hpa_status"         and not result.get("no_data"))
            or (n == "fetch_ingress_health"     and not result.get("no_data"))
        )
        if has_data:
            _data_count += 1
        else:
            _no_data_count += 1

    # If most signals are empty and no anomaly found, surface a warning so the user
    # knows to check their service name / namespace — not silently claim "healthy"
    if _no_data_count >= 5 and not anomalies:
        anomalies.append(
            "[NO DATA] Most metrics returned empty — verify the service name matches the exact "
            "Kubernetes namespace/deployment name (check the Namespace dropdown in Grafana)"
        )

    return anomalies


def _format_all_results(tool_calls: list[dict], results: list[Any]) -> str:
    """
    Format all tool results into a structured text block for the LLM.
    Each tool section starts with STATUS: ANOMALY | OK | ERROR | NO DATA.
    Every number is explicitly labelled.
    """
    lines: list[str] = []

    for tc, result in zip(tool_calls, results):
        name = tc["name"]
        hop  = tc["input"].get("hop", "")
        gran = tc["input"].get("granularity", "")
        tag  = name + (f"(hop={hop}, granularity={gran})" if hop else "")
        lines.append(f"\n{'─'*60}")
        lines.append(f"TOOL: {tag}")

        if not isinstance(result, dict):
            lines.append(f"  STATUS: ERROR")
            lines.append(f"  RESULT: {result}")
            continue
        if "error" in result:
            lines.append(f"  STATUS: ERROR — {result['error']}")
            continue

        if name == "fetch_latency_at_hop":
            series = result.get("series", {})
            anomaly_pods = result.get("anomaly_pods", [])
            elevated = result.get("all_pods_elevated", False)
            is_anomaly = bool(anomaly_pods) or elevated
            lines.append(f"  STATUS: {'ANOMALY' if is_anomaly else 'OK'}")
            prod_anomaly = [p for p in anomaly_pods if not _is_debug_pod(p)]
            prod_series  = {k: v for k, v in series.items() if not _is_debug_pod(k)}
            active_prod  = {k: v for k, v in prod_series.items() if v.get("p99") is not None}
            lines.append(f"  all_pods_elevated : {elevated}")
            lines.append(f"  active_pods_in_window : {len(active_prod)}  (total series including terminated: {len(prod_series)})")
            lines.append(f"  anomaly_pods ({len(prod_anomaly)}/{len(active_prod)} active): "
                         f"{[_pod_name(p) for p in prod_anomaly] if prod_anomaly else 'none'}")
            for pod, s in sorted(prod_series.items()):
                marker = " ← ANOMALY" if pod in prod_anomaly else ""
                lines.append(
                    f"  {_pod_name(pod):50s}  "
                    f"p50={_val(s.get('p50'),'ms')}  "
                    f"p90={_val(s.get('p90'),'ms')}  "
                    f"p99={_val(s.get('p99'),'ms')}{marker}"
                )
            if result.get("grafana_url"):
                lines.append(f"  grafana_url: {result['grafana_url']}")

        elif name == "fetch_error_rate":
            err = result.get("error_rate_pct")
            is_anomaly = err is not None and err > 1.0
            has_data = result.get("success_rate_pct") is not None
            lines.append(f"  STATUS: {'ANOMALY' if is_anomaly else ('OK' if has_data else 'NO DATA')}")
            lines.append(f"  success_rate_pct  : {_val(result.get('success_rate_pct'),'%')}")
            lines.append(f"  error_rate_pct    : {_val(result.get('error_rate_pct'),'%')}")
            lines.append(f"  total_rps         : {_val(result.get('total_rps'),' req/s')}")
            for status, pct in result.get("error_by_status", {}).items():
                lines.append(f"  error breakdown   : {status} = {pct}%")

        elif name == "fetch_cpu_throttling":
            pattern = result.get("dominant_pattern", "none")
            is_anomaly = pattern in ("spiky", "continuous")
            status = "ANOMALY" if is_anomaly else ("NO DATA" if result.get("no_data") else "OK")
            lines.append(f"  STATUS: {status}")
            lines.append(f"  dominant_pattern  : {pattern}")
            lines.append(f"  pattern_counts    : {result.get('pattern_counts', {})}")
            for pod, s in result.get("pods", {}).items():
                if _is_debug_pod(pod):
                    continue
                lines.append(f"  {_pod_name(pod):50s}: pattern={s.get('pattern')}  "
                             f"p99_throttle={_val(s.get('p99'),'%')}")

        elif name == "fetch_rpm_distribution":
            is_anomaly = bool(result.get("uneven"))
            ratio = result.get("imbalance_ratio", 1)
            no_data = not result.get("pods")
            status = "ANOMALY — extreme load imbalance" if is_anomaly else ("NO DATA" if no_data else "OK")
            lines.append(f"  STATUS: {status}")
            lines.append(f"  uneven            : {result.get('uneven')}")
            lines.append(f"  imbalance_ratio   : {_val(ratio,'x')}  (threshold: 2x)")
            lines.append(f"  max_rpm           : {_val(result.get('max_rpm'),'rpm')}")
            lines.append(f"  min_rpm           : {_val(result.get('min_rpm'),'rpm')}")
            prod_pod_rpms = {k: v for k, v in result.get("pods", {}).items() if not _is_debug_pod(k)}
            for pod, rpm in sorted(prod_pod_rpms.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {_pod_name(pod):50s}: {rpm}rpm")

        elif name == "fetch_pod_restarts":
            total = result.get("total_restarts", 0)
            is_anomaly = bool(total and total > 0)
            no_data = not result.get("has_data", True)
            status = "ANOMALY" if is_anomaly else ("NO DATA" if no_data else "OK")
            lines.append(f"  STATUS: {status}")
            lines.append(f"  total_restarts    : {_val(total)}")
            for pod, count in result.get("pods", {}).items():
                if _is_debug_pod(pod):
                    continue
                lines.append(f"  {_pod_name(pod):50s}: {count} restart(s)")
            total_oom = result.get("total_oom_events", 0)
            lines.append(f"  total_oom_events  : {_val(total_oom)}")
            for pod, reason in result.get("restart_reasons", {}).items():
                if not _is_debug_pod(pod):
                    lines.append(f"  {_pod_name(pod):50s}: reason={reason}")

        elif name == "fetch_dns_latency":
            status = result.get("status", "unknown")
            is_anomaly = status in ("warn", "critical")
            has_data = status != "unknown"
            s = result.get("latency_ms", {})
            lines.append(f"  STATUS: {'ANOMALY' if is_anomaly else ('OK' if has_data else 'NO DATA')}")
            lines.append(f"  dns_status        : {status}")
            lines.append(f"  p50={_val(s.get('p50'),'ms')}  p90={_val(s.get('p90'),'ms')}  p99={_val(s.get('p99'),'ms')}")
            if result.get("grafana_url"):
                lines.append(f"  grafana_url: {result['grafana_url']}")

        elif name == "fetch_disk_iops":
            is_anomaly = bool(result.get("exceeds_threshold"))
            status = "ANOMALY" if is_anomaly else ("NO DATA" if result.get("no_data") else "OK")
            lines.append(f"  STATUS: {status}")
            lines.append(f"  iops_per_core     : {_val(result.get('iops_per_core'),' IOPS/core')}  (limit=12)")
            lines.append(f"  peak_iops         : {_val(result.get('peak_iops'),' IOPS')}")
            lines.append(f"  total_cpu_cores   : {_val(result.get('total_cpu_cores'))}")
            if result.get("grafana_url"):
                lines.append(f"  grafana_url: {result['grafana_url']}")

        elif name == "fetch_worker_node_metrics":
            breaches = result.get("threshold_breaches", [])
            is_anomaly = bool(breaches)
            lines.append(f"  STATUS: {'ANOMALY' if is_anomaly else 'OK'}")
            lines.append(f"  is_noisy_neighbour: {result.get('is_noisy_neighbour')}")
            for b in breaches:
                lines.append(f"  BREACH: {b['metric']} = {b['p90']} ({b['pct_used']}% of limit {b['limit']})")
            for metric, s in result.get("metrics", {}).items():
                if isinstance(s, dict) and "p90" in s:
                    lines.append(f"  {metric}: p90={_val(s.get('p90'))}")

        elif name == "fetch_memory_utilization":
            crit = result.get("critical_count", 0)
            warn = result.get("warn_count", 0)
            max_pct = result.get("max_utilization_pct")
            is_anomaly = bool(crit or warn)
            lines.append(f"  STATUS: {'ANOMALY' if is_anomaly else ('OK' if result.get('pods') else 'NO DATA')}")
            lines.append(f"  max_utilization_pct  : {_val(max_pct, '%')}")
            lines.append(f"  critical_pods_count  : {crit}  (>= {result.get('critical_pct_threshold', 90)}%)")
            lines.append(f"  warn_pods_count      : {warn}  (>= {result.get('warn_pct_threshold', 80)}%)")
            for pod_label, p in result.get("pods", {}).items():
                if _is_debug_pod(pod_label):
                    continue
                pct = p.get("utilization_pct")
                marker = " ← CRITICAL" if pct and pct >= result.get("critical_pct_threshold", 90) else (
                         " ← WARN"     if pct and pct >= result.get("warn_pct_threshold", 80)     else "")
                lines.append(
                    f"  {_pod_name(pod_label):50s}  "
                    f"used={_val(p.get('working_set_mb'), 'MB')}  "
                    f"limit={_val(p.get('limit_mb'), 'MB')}  "
                    f"util={_val(pct, '%')}{marker}"
                )

        elif name == "fetch_hpa_status":
            is_anomaly = result.get("hit_max_limit") or result.get("was_scaling")
            no_data = result.get("no_data", False)
            status = "ANOMALY" if is_anomaly else ("NO DATA" if no_data else "OK")
            lines.append(f"  STATUS: {status}")
            lines.append(f"  hpa_min              : {_val(result.get('hpa_min'))}")
            lines.append(f"  hpa_max              : {_val(result.get('hpa_max'))}")
            lines.append(f"  current_desired      : {_val(result.get('current_desired'))}")
            lines.append(f"  peak_desired         : {_val(result.get('peak_desired'))}")
            lines.append(f"  was_scaling          : {result.get('was_scaling')}")
            lines.append(f"  hit_max_limit        : {result.get('hit_max_limit')}")
            lines.append(f"  cpu_utilization_pct  : {_val(result.get('cpu_utilization_pct'), '%')}")

        elif name == "fetch_replica_count":
            is_degraded = result.get("is_degraded", False)
            lines.append(f"  STATUS: {'ANOMALY' if is_degraded else 'OK'}")
            lines.append(f"  available (current)    : {_val(result.get('available'))}")
            lines.append(f"  desired   (current)    : {_val(result.get('desired'))}")
            lines.append(f"  unavailable (max/window): {_val(result.get('unavailable'))}")
            lines.append(f"  min_available (dip)    : {_val(result.get('min_available'))}")

        elif name == "fetch_ingress_health":
            is_anomaly = result.get("has_anomaly", False)
            no_data = result.get("no_data", False)
            status = "ANOMALY" if is_anomaly else ("NO DATA" if no_data else "OK")
            lines.append(f"  STATUS: {status}")
            lines.append(f"  latency_tq_ms        : {_val(result.get('latency_tq_ms'), 'ms')}  (request queue at HAProxy)")
            lines.append(f"  latency_tw_ms        : {_val(result.get('latency_tw_ms'), 'ms')}  (backend pool queue)")
            lines.append(f"  latency_tc_ms        : {_val(result.get('latency_tc_ms'), 'ms')}  (TCP connect to backend)")
            lines.append(f"  latency_tr_ms        : {_val(result.get('latency_tr_ms'), 'ms')}  (backend app response)")
            lines.append(f"  dominant_phase       : {result.get('dominant_phase', 'N/A')}")
            lines.append(f"  http_5xx_per_min     : {_val(result.get('http_5xx_per_min'))}")
            lines.append(f"  client_aborts_per_s  : {_val(result.get('client_aborts_per_s'))}")
            lines.append(f"  ingress_rpm          : {_val(result.get('ingress_rpm'))}")
            tcp_in = result.get("tcp_open_inbound", {})
            tcp_out = result.get("tcp_open_outbound", {})
            if tcp_in:
                max_in = max(tcp_in.values())
                lines.append(f"  tcp_open_inbound_max : {max_in} connections")
            if tcp_out:
                max_out = max(tcp_out.values())
                lines.append(f"  tcp_open_outbound_max: {max_out} connections")
            for a in result.get("anomalies", []):
                lines.append(f"  ANOMALY: {a}")

    lines.append(f"\n{'─'*60}")
    return "\n".join(lines)


def _one_line(tool_name: str, result: Any) -> str:
    """Compact progress line printed as each tool finishes."""
    if not isinstance(result, dict):
        return str(result)[:120]
    if "error" in result:
        return f"ERROR — {result['error']}"
    try:
        if tool_name == "fetch_latency_at_hop":
            series = {k: v for k, v in result.get("series", {}).items() if not _is_debug_pod(k)}
            top = sorted(
                [(k, v["p99"]) for k, v in series.items() if v.get("p99") is not None],
                key=lambda x: x[1], reverse=True
            )[:3]
            top_str = ", ".join(f"{_pod_name(k).split('-')[-1]}={v}ms" for k, v in top)
            prod_anomalies = [p for p in result.get("anomaly_pods", []) if not _is_debug_pod(p)]
            flags = f" | OUTLIERS: {len(prod_anomalies)} pods" if prod_anomalies else ""
            return f"{top_str or 'no data'}{flags}"
        if tool_name == "fetch_error_rate":
            sr = result.get("success_rate_pct")
            rps = result.get("total_rps")
            err = result.get("error_rate_pct")
            return f"success={sr}%  error={err}%  rps={rps}"
        if tool_name == "fetch_cpu_throttling":
            if result.get("no_data"):
                return "NO DATA"
            return f"pattern={result.get('dominant_pattern')}"
        if tool_name == "fetch_rpm_distribution":
            prod = {k: v for k, v in result.get("pods", {}).items() if not _is_debug_pod(k)}
            if not prod:
                return "NO DATA"
            if len(prod) >= 2:
                mx, mn = max(prod.values()), min(prod.values())
                ratio = round(mx / mn, 1) if mn > 0 else mx
                return f"prod pods only — ratio={ratio}x  (max={mx}rpm vs min={mn}rpm)"
            return f"ratio={result.get('imbalance_ratio')}x"
        if tool_name == "fetch_pod_restarts":
            if not result.get("has_data", True):
                return "NO DATA"
            return f"restarts={result.get('total_restarts')}"
        if tool_name == "fetch_worker_node_metrics":
            breaches = result.get("threshold_breaches", [])
            return f"noisy_neighbour={result.get('is_noisy_neighbour')}  breaches={len(breaches)}"
        if tool_name == "fetch_dns_latency":
            return f"status={result.get('status')}  p99={result.get('latency_ms', {}).get('p99')}ms"
        if tool_name == "fetch_disk_iops":
            if result.get("no_data"):
                return "NO DATA"
            return f"exceeds={result.get('exceeds_threshold')}  iops_per_core={result.get('iops_per_core')}"
        if tool_name == "fetch_memory_utilization":
            max_pct = result.get("max_utilization_pct")
            crit = result.get("critical_count", 0)
            warn = result.get("warn_count", 0)
            return f"max_util={max_pct}%  critical={crit}  warn={warn}"
        if tool_name == "fetch_replica_count":
            avail     = result.get("available")
            desired   = result.get("desired")
            unavail   = result.get("unavailable")
            min_avail = result.get("min_available")
            dip = f" (min={min_avail})" if min_avail is not None and min_avail != avail else ""
            return f"available={avail}/{desired}{dip}  unavailable_max={unavail}"
        if tool_name == "fetch_hpa_status":
            if result.get("no_data"):
                return "NO DATA"
            scaling = "SCALING" if result.get("was_scaling") else "stable"
            max_hit = " HIT_MAX" if result.get("hit_max_limit") else ""
            return f"{scaling}{max_hit}  peak={result.get('peak_desired')}  max={result.get('hpa_max')}  cpu_util={result.get('cpu_utilization_pct')}%"
        if tool_name == "fetch_ingress_health":
            if result.get("no_data"):
                return "NO DATA"
            parts = []
            if result.get("latency_tq_ms") is not None:
                parts.append(f"Tq={result['latency_tq_ms']}ms")
            if result.get("latency_tw_ms") is not None:
                parts.append(f"Tw={result['latency_tw_ms']}ms")
            if result.get("latency_tr_ms") is not None:
                parts.append(f"Tr={result['latency_tr_ms']}ms")
            if result.get("http_5xx_per_min") is not None:
                parts.append(f"5xx={result['http_5xx_per_min']}/min")
            if result.get("client_aborts_per_s") is not None:
                parts.append(f"aborts={result['client_aborts_per_s']}/s")
            return "  ".join(parts) if parts else "no anomalies"
    except Exception:
        pass
    return json.dumps(result, default=str)[:200]


def _format_panel_snapshot(panels: list[dict]) -> str:
    """
    Compact, section-grouped text of every dashboard panel's real-time data.
    Skipped/empty panels are omitted. Per-panel series are capped at 15 to keep size manageable.
    """
    lines: list[str] = []
    current_section: str | None = None

    for p in panels:
        if p.get("skip"):
            continue

        section = p.get("section", "")
        if section != current_section:
            current_section = section
            lines.append(f"\n  ── {section} ──")

        panel_id = p.get("panel_id", "?")
        title    = p.get("title", "")

        if p.get("error"):
            lines.append(f"  [{panel_id}] {title}: ERROR — {p['error']}")
            continue

        if not p.get("has_data"):
            continue  # silently skip no-data panels — keeps output readable

        series: dict = p.get("series", {})
        total  = p.get("total_series", len(series))

        lines.append(f"  [{panel_id}] {title}  ({total} series)")

        for label, st in series.items():
            pod_name = _pod_name(label) if ("pod=" in label or "pod" in label) else label[:60]
            p99  = st.get("p99")
            p50  = st.get("p50")
            mx   = st.get("max")
            avg  = st.get("avg")

            parts: list[str] = []
            if p99 is not None:
                parts.append(f"p99={p99}")
            if p50 is not None:
                parts.append(f"p50={p50}")
            if mx is not None and mx != p99:
                parts.append(f"max={mx}")
            if not parts and avg is not None:
                parts.append(f"avg={avg}")

            lines.append(f"    {pod_name}: {' '.join(parts)}")

        if total > 15:
            lines.append(f"    ... and {total - 15} more series (showing top 15 by max value)")

    return "\n".join(lines) if lines else "  (no panel data fetched)"


class DebugAgent:
    def __init__(self, grafana: GrafanaClient, llm_backend: str | None = None):
        self.grafana = grafana
        self.llm = create_llm(llm_backend)

    async def _call_tool(self, name: str, inputs: dict) -> Any:
        fn = _TOOL_FNS.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return await fn(self.grafana, **inputs)
        except TypeError as e:
            return {"error": f"Tool argument error for '{name}': {e}"}
        except Exception as e:
            return {"error": f"Tool '{name}' failed: {e}"}

    def _llm_full_diagnosis(
        self,
        tool_calls: list[dict],
        results: list[Any],
        anomalies: list[str],
        service: str,
        namespace: str,
        cluster: str,
        from_iso: str,
        to_iso: str,
        symptom: str,
        panel_snapshot: list[dict] | None = None,
    ) -> dict:
        """
        Single Anthropic LLM call: give Claude all raw metric data and anomaly list,
        get back a fully specific diagnosis (narrative + root_cause + steps + secondary_steps).
        Falls back to code-based _build_diagnosis if the call fails or returns invalid JSON.
        """
        raw_data = _format_all_results(tool_calls, results)

        # Build an explicit list of pod names that actually appear in the data,
        # so the model cannot invent names that don't exist.
        known_pods: set[str] = set()
        for tc, result in zip(tool_calls, results):
            if not isinstance(result, dict) or "error" in result:
                continue
            for key in ("series", "pods"):
                for pod_label in result.get(key, {}):
                    name = _pod_name(pod_label)
                    if name and name != pod_label:
                        known_pods.add(name)

        # Extract current pod count from replica data — used to anchor the LLM's
        # pod count claim and prevent "102 pods" from rolling-deployment series inflation
        replica_result = None
        for tc, result in zip(tool_calls, results):
            if tc["name"] == "fetch_replica_count" and isinstance(result, dict) and "error" not in result:
                replica_result = result
                break

        # Mark which signals have NO DATA so the model knows not to cite them
        no_data_signals: list[str] = []
        for tc, result in zip(tool_calls, results):
            if not isinstance(result, dict) or "error" in result:
                continue
            name = tc["name"]
            hop = tc["input"].get("hop", "")
            if name == "fetch_dns_latency" and result.get("status", "unknown") == "unknown":
                no_data_signals.append("DNS latency")
            elif name == "fetch_error_rate" and result.get("success_rate_pct") is None:
                no_data_signals.append("Error rate")
            elif name == "fetch_latency_at_hop" and not result.get("series"):
                no_data_signals.append(f"{hop} latency")
            elif name == "fetch_memory_utilization" and not result.get("pods"):
                no_data_signals.append("Memory utilization")
            elif name == "fetch_replica_count" and result.get("available") is None:
                no_data_signals.append("Replica count")

        replica_context = ""
        if replica_result and replica_result.get("desired") is not None:
            r_desired   = replica_result.get("desired")
            r_available = replica_result.get("available")
            replica_context = (
                f"\n\nCURRENT REPLICA STATE (authoritative pod count — use this, not series count):\n"
                f"  Desired replicas  : {r_desired}\n"
                f"  Available replicas: {r_available}\n"
                f"  IMPORTANT: Rolling deployments create one series per pod ever deployed in the window. "
                f"The metric data may show 100+ series for a service that only runs {r_desired} pods. "
                f"The REAL fleet size is {r_desired} desired / {r_available} available. "
                f"The anomaly strings above already use the correct denominator — do not recalculate."
            )

        prompt = (
            f"You are diagnosing a live Kubernetes incident. Analyze the metric data below.\n\n"
            f"Context:\n"
            f"  Service:   {service}\n"
            f"  Namespace: {namespace}\n"
            f"  Cluster:   {cluster}\n"
            f"  Window:    {from_iso} → {to_iso}\n"
            + (f"  Symptom reported: {symptom}\n" if symptom else "") +
            f"\nANOMALIES DETECTED (these are the ONLY problems — ordered most severe first):\n"
            + "\n".join(f"  {i+1}. {a}" for i, a in enumerate(anomalies)) +
            (
                f"\n\nSIGNALS WITH NO DATA (do NOT cite these as causes — data was unavailable):\n"
                + "\n".join(f"  - {s}" for s in no_data_signals)
                if no_data_signals else ""
            ) +
            (
                f"\n\nVALID POD NAMES IN THIS DATA (use ONLY these — do not invent others):\n"
                + "\n".join(f"  - {p}" for p in sorted(known_pods))
                if known_pods else ""
            ) +
            replica_context +
            f"\n\nCOMPLETE METRIC DATA:\n{raw_data}\n\n"
            + (
                f"FULL GRAFANA DASHBOARD SNAPSHOT (real-time data from ALL dashboard panels):\n"
                + _format_panel_snapshot(panel_snapshot)
                + "\n\n"
                if panel_snapshot else ""
            )
            + f"SERVICE CONTEXT — read carefully before diagnosing:\n"
            f"  - This is an MLP service. It does NOT use gRPC. Never cite gRPC as a cause.\n"
            f"  - Debug pods (name contains 'debug') are already excluded from the data — do not mention them.\n\n"
            f"METRIC DEFINITIONS (do not confuse these):\n"
            f"  RPM = requests per minute (traffic volume per pod)\n"
            f"  IOPS/core = disk I/O operations per second per CPU core (storage pressure)\n"
            f"  p99 latency = 99th percentile response time in milliseconds\n\n"
            f"CAUSE → FIX PLAYBOOK (use the entry that matches anomaly #1; never apply a different entry's fix):\n"
            f"\n"
            f"  [RPM IMBALANCE]\n"
            f"    Fix: DS team identifies upstream caller via Linkerd service map or Grafana.\n"
            f"         Ask SRE to fix load balancing config on the upstream caller (HTTP keep-alive tuning,\n"
            f"         connection pool settings, or session affinity). No restart needed.\n"
            f"    Do NOT suggest: rollout restart, CPU limit changes, or scaling.\n"
            f"\n"
            f"  [INBOUND LATENCY] subset of pods, T6 normal, T3 normal\n"
            f"    Fix: check logs on affected pods for slow queries / GC pauses / blocking calls.\n"
            f"         kubectl logs -n {namespace} <pod> --since=30m | grep -iE 'slow|timeout|error'\n"
            f"         Check for recent deployment: kubectl rollout history deployment/{service} -n {namespace}\n"
            f"    Do NOT suggest: rollout restart as a first step.\n"
            f"\n"
            f"  [INGRESS LATENCY] T3 elevated, all pods elevated (Tq dominant)\n"
            f"    Fix: HAProxy maxconn or backend pool exhausted. Ask SRE to increase maxconn or pod count.\n"
            f"    Do NOT suggest: application code changes or restart.\n"
            f"\n"
            f"  [CPU THROTTLE] spiky\n"
            f"    Fix: Ask SRE to increase CPU limit to 2× current request on deployment/{service}.\n"
            f"         kubectl set resources deployment/{service} -n {namespace} --limits=cpu=<2x_current>\n"
            f"    Do NOT suggest: rollout restart or scaling.\n"
            f"\n"
            f"  [CPU THROTTLE] continuous\n"
            f"    Fix: Ask SRE to increase CPU request (pod is under-provisioned for steady-state load).\n"
            f"    Do NOT suggest: rollout restart.\n"
            f"\n"
            f"  [DISK IOPS] > 12 IOPS/core\n"
            f"    Cause: synchronous logging in the request path saturating the Azure P20 disk IOPS budget.\n"
            f"    Fix step 1 (DS team, no kubectl): switch application logging to async mode.\n"
            f"      Python → logging.handlers.QueueHandler with a background flush thread\n"
            f"      Java   → Log4j AsyncAppender or Logback AsyncAppender\n"
            f"    Fix step 2: ONLY AFTER the code change, Ask SRE to rollout restart to pick up new config.\n"
            f"    Do NOT suggest rollout restart without the logging code change first.\n"
            f"\n"
            f"  [POD RESTARTS] OOMKill\n"
            f"    Fix: Ask SRE to increase memory limit on deployment/{service}.\n"
            f"    Do NOT suggest: rollout restart (the pod is already restarting).\n"
            f"\n"
            f"  [POD RESTARTS] liveness probe\n"
            f"    Fix: Ask SRE to increase initialDelaySeconds or failureThreshold on the liveness probe.\n"
            f"\n"
            f"  [DNS] critical latency\n"
            f"    Fix: Use fully-qualified DNS names. Escalate to SRE for CoreDNS capacity.\n"
            f"\n"
            f"  [HPA MAX LIMIT]\n"
            f"    Fix: Ask SRE to increase maxReplicas in Helm values.yaml and redeploy.\n"
            f"\n"
            f"  [T6 OUTBOUND] ~5000ms\n"
            f"    Fix: Ask SRE to annotate service with linkerd.io/opaque-ports for the relevant port.\n"
            f"\n"
            f"CRITICAL: Do NOT suggest 'kubectl rollout restart' unless the cause is [DISK IOPS] AND the\n"
            f"  async logging code change has been made. For all other causes use the specific fix above.\n\n"
            f"Respond with ONLY valid JSON — no markdown fences, no extra text:\n"
            f"{{\n"
            f'  "narrative": "2–4 sentences explaining the cause-effect chain using measured values (ms, %, rpm). Copy counts like N/M backends directly from the ANOMALIES section — do not recount. Do NOT name individual pods.",\n'
            f'  "root_cause": "One sentence stating the cause with the key measured value (e.g. latency, throttle %, ratio). Do NOT list individual pod names — they are already in the anomalies section.",\n'
            f'  "steps": [\n'
            f'    "Step 1: specific kubectl command using namespace={namespace} and the actual pod/deployment names",\n'
            f'    "Step 2: ...",\n'
            f'    "Step 3: how to verify the fix in Grafana"\n'
            f'  ],\n'
            f'  "secondary_steps": ["Fix for anomaly #2 if present, empty list if only one anomaly"],\n'
            f'  "escalate": false,\n'
            f'  "escalate_reason": "N/A"\n'
            f"}}\n\n"
            f"ABSOLUTE RULES — violating any of these makes the diagnosis wrong:\n"
            f"1. In steps/secondary_steps, use ONLY pod names from the VALID POD NAMES list — never modify or invent pod names\n"
            f"2. Use ONLY metric values that appear in the COMPLETE METRIC DATA section\n"
            f"3. Never cite a NO DATA signal as a cause or contributing factor\n"
            f"4. The primary root_cause must match anomaly #1 (most severe) from the list above\n"
            f"5. secondary_steps must address anomaly #2+ only — not the primary root cause\n"
            f"6. All kubectl commands must use --namespace {namespace}\n"
            f"7. ESCALATION (mandatory): The DS team has NO kubectl write access to production.\n"
            f"   - Set escalate=true if ANY step requires kubectl annotate/rollout/set/delete\n"
            f"   - Prefix every such step with 'Ask SRE to run:'\n"
            f"   - escalate_reason must summarize what the SRE team needs to do\n"
            f"   - Read-only steps (kubectl logs, kubectl get) do NOT require escalation\n"
            f"8. TOTAL POD COUNT: Use the CURRENT REPLICA STATE section for fleet size.\n"
            f"   - DO NOT count metric series — rolling deployments create multiple pod hashes over time.\n"
            f"   - When describing fleet size, cite desired/available from CURRENT REPLICA STATE, not series count.\n"
            f"9. DO NOT repeat individual pod names in narrative or root_cause — they are already listed in the\n"
            f"   ANOMALIES section. Use aggregate descriptions instead (e.g. 'affected pods', 'N/M backends').\n"
            f"10. Use the EXACT counts from the ANOMALIES section (e.g. '4/27 backends') — never recount from\n"
            f"    the metric data. The anomaly string is authoritative; recounting produces wrong numbers."
        )

        try:
            kwargs: dict = {}
            if not isinstance(self.llm, AnthropicLLM):
                kwargs["json_mode"] = True
            response = self.llm.complete(
                [{"role": "user", "content": prompt}],
                SYSTEM_PROMPT,
                [],
                **kwargs,
            )
            text = (response.content or "").strip()

            # Strip markdown code fences if the model wraps with them anyway
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)
            return {
                "narrative":       str(parsed.get("narrative", "")),
                "root_cause":      str(parsed.get("root_cause", "")),
                "steps":           list(parsed.get("steps", [])),
                "secondary_steps": list(parsed.get("secondary_steps", [])),
                "escalate":        bool(parsed.get("escalate", False)),
                "escalate_reason": str(parsed.get("escalate_reason", "N/A")),
            }
        except Exception as e:
            fallback = _build_diagnosis(tool_calls, results, anomalies, service, namespace)
            fallback["narrative"] = f"(LLM diagnosis unavailable: {e}. Code-based analysis below.)"
            return fallback

    async def investigate(
        self,
        service: str,
        namespace: str,
        time_from: int,
        time_to: int,
        grafana_vars: dict[str, str] | None = None,
        symptom: str = "",
        verbose: bool = True,
        return_raw: bool = False,
    ) -> "str | dict":
        """
        Run a full multi-signal diagnostic investigation.

        Runs all tools in parallel, then asks the LLM to interpret and report.
        No iterative tool-calling — reliable with local and cloud models.
        """
        def _log(msg: str) -> None:
            if verbose:
                print(msg, flush=True)

        self.grafana.vars = grafana_vars or {}

        from_iso = datetime.fromtimestamp(time_from / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        to_iso   = datetime.fromtimestamp(time_to   / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        duration_s = (time_to - time_from) // 1000
        cluster = self.grafana.vars.get("cluster", "unknown")

        # ── Step 1: Collect all metrics in parallel ─────────────────────────────
        tool_calls = [
            {"name": "fetch_latency_at_hop",
             "input": {"service": service, "namespace": namespace, "hop": "inbound",
                       "granularity": "pod", "time_from": time_from, "time_to": time_to}},
            {"name": "fetch_latency_at_hop",
             "input": {"service": service, "namespace": namespace, "hop": "outbound",
                       "granularity": "deployment", "time_from": time_from, "time_to": time_to}},
            {"name": "fetch_latency_at_hop",
             "input": {"service": service, "namespace": namespace, "hop": "ingress",
                       "granularity": "deployment", "time_from": time_from, "time_to": time_to}},
            {"name": "fetch_ingress_health",
             "input": {"service": service, "namespace": namespace,
                       "time_from": time_from, "time_to": time_to}},
            {"name": "fetch_error_rate",
             "input": {"service": service, "namespace": namespace,
                       "time_from": time_from, "time_to": time_to}},
            {"name": "fetch_cpu_throttling",
             "input": {"service": service, "namespace": namespace,
                       "time_from": time_from, "time_to": time_to}},
            {"name": "fetch_rpm_distribution",
             "input": {"service": service, "namespace": namespace,
                       "time_from": time_from, "time_to": time_to}},
            {"name": "fetch_pod_restarts",
             "input": {"service": service, "namespace": namespace,
                       "time_from": time_from, "time_to": time_to}},
            {"name": "fetch_dns_latency",
             "input": {"time_from": time_from, "time_to": time_to}},
            {"name": "fetch_disk_iops",
             "input": {"service": service, "namespace": namespace,
                       "time_from": time_from, "time_to": time_to}},
            {"name": "fetch_memory_utilization",
             "input": {"service": service, "namespace": namespace,
                       "time_from": time_from, "time_to": time_to}},
            {"name": "fetch_replica_count",
             "input": {"service": service, "namespace": namespace,
                       "time_from": time_from, "time_to": time_to}},
            {"name": "fetch_hpa_status",
             "input": {"service": service, "namespace": namespace,
                       "time_from": time_from, "time_to": time_to}},
        ]

        _log("\nCollecting metrics and full dashboard snapshot in parallel...")
        results_list, panel_snapshot = await asyncio.gather(
            asyncio.gather(*[self._call_tool(tc["name"], tc["input"]) for tc in tool_calls]),
            fetch_all_panel_data(self.grafana, namespace, time_from, time_to),
        )
        results = list(results_list)

        panels_with_data = sum(1 for p in panel_snapshot if p.get("has_data"))
        _log(f"  ✓ Dashboard snapshot: {panels_with_data} panels with data")

        for tc, result in zip(tool_calls, results):
            hop   = tc["input"].get("hop", "")
            label = f"({hop})" if hop else ""
            status = "✓" if not (isinstance(result, dict) and "error" in result) else "✗"
            _log(f"  {status} {tc['name']}{label}: {_one_line(tc['name'], result)}")

        # ── Step 2: Detect anomalies ─────────────────────────────────────────────
        anomalies = _detect_anomalies(tool_calls, results)

        # ── Step 3: Generate diagnosis ────────────────────────────────────────────
        # Anthropic path: single LLM call with ALL raw data → fully specific diagnosis
        # Ollama / no-anomaly path: code-based rule engine (fast, deterministic)
        if anomalies:
            _log("\nGenerating LLM diagnosis (data-driven)...")
            llm_result = self._llm_full_diagnosis(
                tool_calls, results, anomalies, service, namespace,
                cluster, from_iso, to_iso, symptom,
                panel_snapshot=panel_snapshot,
            )
            narrative = llm_result.pop("narrative", "")
            diagnosis = llm_result
        else:
            diagnosis = _build_diagnosis(tool_calls, results, anomalies, service, namespace)
            _log("\nGenerating narrative...")
            if not anomalies:
                narrative = "No anomalies detected in this time window. All collected signals are within normal ranges."
            else:
                narrative_prompt = (
                    f"Service: {service}/{namespace}  Cluster: {cluster}\n"
                    f"Window: {from_iso} → {to_iso}\n\n"
                    f"ANOMALIES FOUND (use ONLY these values — do not invent numbers):\n"
                    + "\n".join(f"  {a}" for a in anomalies) +
                    f"\n\nROOT CAUSE DETERMINED BY PLAYBOOK:\n  {diagnosis['root_cause']}\n\n"
                    "Write 2–4 sentences explaining what is happening in this specific service. "
                    "Requirements:\n"
                    "- Name the specific pods/values from the anomaly list above\n"
                    "- Explain the cause → effect chain between the anomalies\n"
                    "- Do NOT repeat the root cause verbatim — add context and interpretation\n"
                    "- Do NOT reference any metric value not listed in ANOMALIES FOUND\n"
                    "Write ONLY the paragraph — no headers, no bullet points."
                )
                narrative = ""
                try:
                    response = self.llm.complete(
                        [{"role": "user", "content": narrative_prompt}], SYSTEM_PROMPT, []
                    )
                    narrative = (response.content or "").strip()
                except Exception as e:
                    if "Timeout" in type(e).__name__ or "timeout" in str(e).lower():
                        narrative = "(LLM timed out — see root cause and steps below for the full analysis)"
                    else:
                        raise

        # Healthy signals = tools that returned OK with data
        healthy: list[str] = []
        for tc, result in zip(tool_calls, results):
            if not isinstance(result, dict) or "error" in result:
                continue
            name = tc["name"]
            hop  = tc["input"].get("hop", "")
            label_map = {
                ("fetch_latency_at_hop", "inbound"):  "T4 — server-linkerd → server",
                ("fetch_latency_at_hop", "outbound"): "T6 — server-linkerd → other-services/DB",
                ("fetch_latency_at_hop", "ingress"):  "T3 — ingress → server-linkerd",
                ("fetch_cpu_throttling", ""):         "CPU throttle",
                ("fetch_rpm_distribution", ""):       "RPM distribution",
                ("fetch_pod_restarts", ""):            "Pod restarts",
                ("fetch_dns_latency", ""):             "T7 — DNS resolution",
                ("fetch_disk_iops", ""):               "Disk IOPS",
                ("fetch_error_rate", ""):              "Error rate",
                ("fetch_memory_utilization", ""):      "Memory utilization",
                ("fetch_replica_count", ""):           "Replica count",
                ("fetch_hpa_status", ""):              "HPA status",
                ("fetch_ingress_health", ""):          "Ingress details",
            }
            label = label_map.get((name, hop))
            if not label:
                continue

            # Determine if this tool's result is healthy (not in anomaly list)
            anomaly_keys = [name.split("(")[0] for name in anomalies]
            is_anomaly = any(
                (name == "fetch_latency_at_hop" and hop == "inbound"  and "T4" in a)
                or (name == "fetch_latency_at_hop" and hop == "outbound" and "T6" in a)
                or (name == "fetch_latency_at_hop" and hop == "ingress"  and "T3" in a)
                or (name == "fetch_rpm_distribution" and "RPM" in a)
                or (name == "fetch_cpu_throttling" and "CPU" in a)
                or (name == "fetch_pod_restarts" and "RESTART" in a)
                or (name == "fetch_dns_latency" and ("T7" in a or "DNS" in a))
                or (name == "fetch_disk_iops" and "DISK" in a)
                or (name == "fetch_error_rate" and "ERROR" in a)
                or (name == "fetch_memory_utilization" and "MEMORY" in a)
                or (name == "fetch_replica_count" and "REPLICAS" in a)
                for a in anomalies
            )
            if is_anomaly:
                continue

            # Build concise healthy summary
            summary = ""
            if name == "fetch_latency_at_hop":
                series = result.get("series", {})
                p99s = [v.get("p99") for v in series.values() if v.get("p99") is not None]
                if p99s:
                    summary = f"p99={max(p99s)}ms"
                else:
                    summary = "NO DATA"
            elif name == "fetch_cpu_throttling":
                summary = "NO DATA" if result.get("no_data") else f"pattern={result.get('dominant_pattern', 'none')}"
            elif name == "fetch_rpm_distribution":
                prod = {k: v for k, v in result.get("pods", {}).items() if not _is_debug_pod(k)}
                if not prod:
                    summary = "NO DATA"
                elif len(prod) >= 2:
                    mx, mn = max(prod.values()), min(prod.values())
                    ratio = round(mx / mn, 1) if mn > 0 else mx
                    summary = f"ratio={ratio}x (prod pods)"
                else:
                    summary = f"ratio={result.get('imbalance_ratio', 1)}x (prod pods)"
            elif name == "fetch_pod_restarts":
                if not result.get("has_data", True):
                    summary = "NO DATA"
                else:
                    summary = f"{result.get('total_restarts', 0)} restarts"
            elif name == "fetch_dns_latency":
                p99 = result.get("latency_ms", {}).get("p99")
                summary = f"p99={p99}ms" if p99 is not None else "NO DATA"
            elif name == "fetch_disk_iops":
                summary = "NO DATA" if result.get("no_data") else f"{result.get('iops_per_core')} IOPS/core"
            elif name == "fetch_error_rate":
                sr = result.get("success_rate_pct")
                summary = f"success={sr}%" if sr is not None else "NO DATA"
            elif name == "fetch_memory_utilization":
                max_pct = result.get("max_utilization_pct")
                summary = f"max={max_pct}%" if max_pct is not None else "NO DATA"
            elif name == "fetch_replica_count":
                avail     = result.get("available")
                desired   = result.get("desired")
                min_avail = result.get("min_available")
                if avail is not None:
                    dip = f" (min={min_avail})" if min_avail is not None and min_avail != avail else ""
                    summary = f"{avail}/{desired} available{dip}"
                else:
                    summary = "NO DATA"
            elif name == "fetch_hpa_status":
                if result.get("no_data"):
                    summary = "NO DATA"
                else:
                    curr = result.get("current_desired")
                    hpa_max = result.get("hpa_max")
                    summary = f"desired={curr}/{hpa_max}  scaling={'yes' if result.get('was_scaling') else 'no'}"
            elif name == "fetch_ingress_health":
                if result.get("no_data"):
                    summary = "NO DATA"
                else:
                    dom = result.get("dominant_phase")
                    tr = result.get("latency_tr_ms")
                    c5xx = result.get("http_5xx_per_min", 0) or 0
                    summary = f"dominant={dom}  Tr={tr}ms  5xx={c5xx}/min"

            healthy.append(f"  {label}: {summary}")

        # Collect Grafana links
        grafana_links: list[str] = []
        for result in results:
            if isinstance(result, dict) and result.get("grafana_url"):
                grafana_links.append(f"  {result['grafana_url']}")

        # ── Step 4: Assemble the final report ────────────────────────────────────
        sep = "━" * 62

        what_wrong_lines = [f"  {a}" for a in anomalies] if anomalies else ["  (none detected)"]
        step_lines = "\n".join(
            f"  Step {i+1} — {s}" for i, s in enumerate(diagnosis["steps"])
        )

        secondary = diagnosis.get("secondary_steps", [])
        also_address = (
            f"\nALSO ADDRESS\n{'━'*14}\n"
            + "\n\n".join(f"  {s}" for s in secondary)
            + "\n"
        ) if secondary else ""

        report = (
            f"INVESTIGATION REPORT\n{sep}\n"
            f"Service:      {service} / {namespace}\n"
            f"Cluster:      {cluster}\n"
            f"Window:       {from_iso} → {to_iso}\n"
            f"\n"
            f"WHAT'S WRONG\n{'━'*14}\n"
            + "\n".join(what_wrong_lines) +
            f"\n\n"
            f"HEALTHY SIGNALS\n{'━'*17}\n"
            + ("\n".join(healthy) if healthy else "  (no healthy data — check Grafana connectivity)") +
            f"\n\n"
            f"WHAT IS HAPPENING\n{'━'*19}\n"
            f"  {narrative}\n"
            f"\n"
            f"MOST LIKELY CAUSE\n{'━'*19}\n"
            f"  {diagnosis['root_cause']}\n"
            f"\n"
            f"WHAT TO DO\n{'━'*12}\n"
            + step_lines +
            f"\n\n"
            + also_address
            + f"ESCALATE TO SRE: {'YES' if diagnosis['escalate'] else 'NO'}\n"
            f"  Reason: {diagnosis['escalate_reason']}\n"
            + (f"\nGRAFANA LINKS\n{'━'*15}\n" + "\n".join(grafana_links) + "\n" if grafana_links else "")
        )

        if return_raw:
            return {
                "service": service,
                "namespace": namespace,
                "cluster": cluster,
                "from_iso": from_iso,
                "to_iso": to_iso,
                "anomalies": anomalies,
                "healthy": [h.strip() for h in healthy],
                "narrative": narrative,
                "root_cause": diagnosis["root_cause"],
                "steps": diagnosis["steps"],
                "secondary_steps": diagnosis.get("secondary_steps", []),
                "escalate": diagnosis["escalate"],
                "escalate_reason": diagnosis["escalate_reason"],
                "grafana_links": [lnk.strip() for lnk in grafana_links],
                "tool_summary": [
                    {
                        "name": tc["name"],
                        "hop": tc["input"].get("hop", ""),
                        "ok": isinstance(r, dict) and "error" not in r,
                        "one_line": _one_line(tc["name"], r),
                    }
                    for tc, r in zip(tool_calls, results)
                ],
            }

        return report


def create_agent_from_env(llm_backend: str | None = None) -> DebugAgent:
    """Construct a DebugAgent from environment variables."""
    grafana = GrafanaClient(
        base_url=os.getenv("GRAFANA_URL", ""),
        token=os.getenv("GRAFANA_TOKEN", ""),
        datasource_uid=os.getenv("GRAFANA_DATASOURCE_UID", ""),
    )
    return DebugAgent(grafana=grafana, llm_backend=llm_backend)
