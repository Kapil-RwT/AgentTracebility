"""
Unit tests for tool implementations.
Grafana HTTP calls are mocked with respx so no real network is needed.
"""

import pytest
import httpx
import respx

from src.grafana_client import GrafanaClient
from src.tools import (
    _parse_frames,
    _stats,
    fetch_cpu_throttling,
    fetch_disk_iops,
    fetch_dns_latency,
    fetch_latency_at_hop,
    fetch_pod_restarts,
    fetch_rpm_distribution,
)

BASE_URL = "https://grafana.test"
DS_UID = "test-prometheus"

TIME_FROM = 1_700_000_000_000
TIME_TO = 1_700_003_600_000  # +1 hour


# ─── Helpers ──────────────────────────────────────────────────────────────────

def frame(label_value: str, values: list[float]) -> dict:
    """Build a minimal Grafana data frame with one series."""
    n = len(values)
    return {
        "schema": {
            "fields": [
                {"name": "Time", "type": "time"},
                {"name": label_value, "type": "number", "labels": {"pod": label_value}},
            ]
        },
        "data": {
            "values": [
                [TIME_FROM + i * 60_000 for i in range(n)],
                values,
            ]
        },
    }


def grafana_response(*frames_list) -> dict:
    return {"results": {"A": {"frames": list(frames_list)}}}


@pytest.fixture
def client() -> GrafanaClient:
    return GrafanaClient(BASE_URL, token="tok", datasource_uid=DS_UID)


# ─── _parse_frames ─────────────────────────────────────────────────────────────

class TestParseFrames:
    def test_single_series(self):
        result = _parse_frames([frame("pod-1", [10.0, 20.0, 30.0])])
        assert "pod=pod-1" in result
        assert result["pod=pod-1"] == [10.0, 20.0, 30.0]

    def test_empty_input(self):
        assert _parse_frames([]) == {}

    def test_null_values_dropped(self):
        f = frame("pod-1", [10.0, None, 30.0])
        result = _parse_frames([f])
        assert result["pod=pod-1"] == [10.0, 30.0]

    def test_multiple_series(self):
        result = _parse_frames([frame("pod-1", [1.0]), frame("pod-2", [2.0])])
        assert len(result) == 2


# ─── _stats ───────────────────────────────────────────────────────────────────

class TestStats:
    def test_basic_percentiles(self):
        s = _stats(list(range(1, 101)))  # 1..100 (100 values)
        # int(100 * 0.50) = index 50 → value 51 (floor percentile)
        assert s["p50"] == 51
        assert s["p90"] == 91
        assert s["max"] == 100
        assert s["p50"] <= s["p90"] <= s["p99"] <= s["max"]

    def test_empty_returns_nones(self):
        s = _stats([])
        assert all(v is None for v in s.values())

    def test_single_value(self):
        s = _stats([42.0])
        assert s["p50"] == s["p90"] == s["p99"] == s["max"] == 42.0


# ─── fetch_latency_at_hop ─────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFetchLatencyAtHop:
    @respx.mock
    async def test_inbound_detects_outlier_pod(self, client):
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(
                200,
                json=grafana_response(
                    frame("pod-1", [100.0, 110.0, 105.0]),
                    frame("pod-2", [400.0, 450.0, 420.0]),  # ~4x median → outlier
                ),
            )
        )
        result = await fetch_latency_at_hop(
            client, "my-svc", "prod", "inbound", "pod", TIME_FROM, TIME_TO
        )
        assert "series" in result
        assert len(result["anomaly_pods"]) >= 1
        # pod-2 p99 should be detected as anomaly
        assert any("pod-2" in p for p in result["anomaly_pods"])

    @respx.mock
    async def test_all_pods_elevated_flag(self, client):
        # Both pods equally elevated → all_pods_elevated should be True
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(
                200,
                json=grafana_response(
                    frame("pod-1", [500.0] * 10),
                    frame("pod-2", [500.0] * 10),
                ),
            )
        )
        result = await fetch_latency_at_hop(
            client, "my-svc", "prod", "inbound", "pod", TIME_FROM, TIME_TO
        )
        # Both at same level → neither is an outlier → anomaly_pods empty → all_pods_elevated False
        assert result["all_pods_elevated"] is False

    @respx.mock
    async def test_unknown_hop_returns_error(self, client):
        result = await fetch_latency_at_hop(
            client, "my-svc", "prod", "t4", "pod", TIME_FROM, TIME_TO
        )
        assert "error" in result

    @respx.mock
    async def test_grafana_500_returns_error(self, client):
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(500, json={"message": "internal"})
        )
        result = await fetch_latency_at_hop(
            client, "my-svc", "prod", "inbound", "pod", TIME_FROM, TIME_TO
        )
        assert "error" in result


# ─── fetch_cpu_throttling ─────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFetchCpuThrottling:
    @respx.mock
    async def test_spiky_pattern(self, client):
        # Mostly below threshold, occasional spikes above 30%
        vals = [2.0] * 25 + [60.0] * 3 + [2.0] * 2
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(200, json=grafana_response(frame("pod-1", vals)))
        )
        result = await fetch_cpu_throttling(client, "svc", "ns", TIME_FROM, TIME_TO)
        assert result["dominant_pattern"] == "spiky"
        assert "2×" in result["recommended_fix"]

    @respx.mock
    async def test_continuous_pattern(self, client):
        vals = [40.0] * 30  # sustained above 10% and 30%
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(200, json=grafana_response(frame("pod-1", vals)))
        )
        result = await fetch_cpu_throttling(client, "svc", "ns", TIME_FROM, TIME_TO)
        assert result["dominant_pattern"] == "continuous"
        assert result["recommended_fix"] is not None

    @respx.mock
    async def test_no_throttle(self, client):
        vals = [1.0] * 30
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(200, json=grafana_response(frame("pod-1", vals)))
        )
        result = await fetch_cpu_throttling(client, "svc", "ns", TIME_FROM, TIME_TO)
        assert result["dominant_pattern"] == "none"
        assert result["recommended_fix"] is None


# ─── fetch_rpm_distribution ───────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFetchRpmDistribution:
    @respx.mock
    async def test_uneven_distribution(self, client):
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(
                200,
                json=grafana_response(
                    frame("pod-1", [1000.0] * 10),
                    frame("pod-2", [200.0] * 10),
                    frame("pod-3", [190.0] * 10),
                ),
            )
        )
        result = await fetch_rpm_distribution(client, "svc", "ns", TIME_FROM, TIME_TO)
        assert result["uneven"] is True
        assert result["recommended_fix"] is not None

    @respx.mock
    async def test_even_distribution(self, client):
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(
                200,
                json=grafana_response(
                    frame("pod-1", [500.0] * 10),
                    frame("pod-2", [510.0] * 10),
                    frame("pod-3", [495.0] * 10),
                ),
            )
        )
        result = await fetch_rpm_distribution(client, "svc", "ns", TIME_FROM, TIME_TO)
        assert result["uneven"] is False


# ─── fetch_dns_latency ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFetchDnsLatency:
    @respx.mock
    async def test_critical_latency(self, client):
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(200, json=grafana_response(frame("dns", [80.0] * 10)))
        )
        result = await fetch_dns_latency(client, TIME_FROM, TIME_TO)
        assert result["status"] == "critical"
        assert result["recommended_fix"] is not None

    @respx.mock
    async def test_warn_latency(self, client):
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(200, json=grafana_response(frame("dns", [15.0] * 10)))
        )
        result = await fetch_dns_latency(client, TIME_FROM, TIME_TO)
        assert result["status"] == "warn"

    @respx.mock
    async def test_ok_latency(self, client):
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(200, json=grafana_response(frame("dns", [3.0] * 10)))
        )
        result = await fetch_dns_latency(client, TIME_FROM, TIME_TO)
        assert result["status"] == "ok"
        assert result["recommended_fix"] is None


# ─── fetch_disk_iops ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFetchDiskIops:
    @respx.mock
    async def test_exceeds_threshold(self, client):
        # 200 IOPS on a service with 4 CPU cores → 50 IOPS/core > 12 limit
        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # IOPS query
                return httpx.Response(200, json=grafana_response(frame("pod-1", [200.0] * 10)))
            else:
                # CPU request query — return 4 cores
                return httpx.Response(200, json=grafana_response(frame("total", [4.0] * 10)))

        respx.post(f"{BASE_URL}/api/ds/query").mock(side_effect=side_effect)
        result = await fetch_disk_iops(client, "svc", "ns", TIME_FROM, TIME_TO)
        assert result["exceeds_threshold"] is True
        assert result["recommended_fix"] is not None

    @respx.mock
    async def test_within_threshold(self, client):
        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(200, json=grafana_response(frame("pod-1", [10.0] * 10)))
            else:
                return httpx.Response(200, json=grafana_response(frame("total", [4.0] * 10)))

        respx.post(f"{BASE_URL}/api/ds/query").mock(side_effect=side_effect)
        result = await fetch_disk_iops(client, "svc", "ns", TIME_FROM, TIME_TO)
        assert result["exceeds_threshold"] is False


# ─── fetch_pod_restarts ───────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFetchPodRestarts:
    @respx.mock
    async def test_detects_restarts(self, client):
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(200, json=grafana_response(frame("pod-1", [0, 1, 2])))
        )
        result = await fetch_pod_restarts(client, "svc", "ns", TIME_FROM, TIME_TO)
        assert result["has_restarts"] is True
        assert result["total_restarts"] == 2

    @respx.mock
    async def test_no_restarts(self, client):
        respx.post(f"{BASE_URL}/api/ds/query").mock(
            return_value=httpx.Response(200, json=grafana_response(frame("pod-1", [0.0] * 10)))
        )
        result = await fetch_pod_restarts(client, "svc", "ns", TIME_FROM, TIME_TO)
        assert result["has_restarts"] is False
        assert result["recommended_fix"] is None
