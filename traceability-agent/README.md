# K8s Debug Agent

An LLM-powered agent that diagnoses latency spikes in Kubernetes services by querying Grafana metrics and reasoning over them using a T1→T7 hop model.

## Quick start

```bash
cp .env.example .env
# Fill in GRAFANA_URL, GRAFANA_TOKEN, GRAFANA_DATASOURCE_UID, ANTHROPIC_API_KEY
pip install -r requirements.txt

# Investigate a service
python -m src.interfaces.cli --service catalog-service --namespace production

# Custom time window + symptom
python -m src.interfaces.cli \
  --service catalog-service --namespace production \
  --from 2h --symptom "p99 spiked to 2s, all pods affected"

# Use local Ollama instead of Claude (free, needs GPU)
LLM_BACKEND=ollama python -m src.interfaces.cli --service catalog-service --namespace production
```

## Local testing with Ollama

```bash
# Install Ollama: https://ollama.com
ollama serve
ollama pull llama3.1:8b   # ~5GB download; mistral-nemo or qwen2.5:7b also work
export LLM_BACKEND=ollama
python -m src.interfaces.cli --service my-service --namespace default
```

## Run tests

```bash
pytest tests/ -v
```

## Architecture

```
CLI
    │
    ▼
Agent (agent.py)          ← ReAct loop: LLM → tools → reason → repeat
    │
    ├── LLM Client         ← AnthropicLLM (Claude) or OllamaLLM (local)
    │
    └── Tools (tools.py)   ← Query Grafana, apply thresholds, return structured data
            │
            ▼
        Grafana HTTP API   ← POST /api/ds/query (PromQL)
```

## Traffic hop model

Every request traverses T1→T7. The agent binary-searches which hop introduces latency:

| Hop | Path | Grafana metric |
|-----|------|----------------|
| T1 | Client → Client-Linkerd | No metrics |
| T2 | Client-Linkerd → Ingress | Linkerd outbound latency |
| T3 | Ingress → Server-Linkerd | nginx_ingress_controller_request_duration |
| T4 | Server-Linkerd → Server | Linkerd inbound latency |
| T5 | Server → Server-Linkerd | No metrics |
| T6 | Server-Linkerd → Downstream | Linkerd outbound latency |
| T7 | DNS resolution | coredns_dns_request_duration |

## Configuration

| File | Purpose |
|------|---------|
| `config/panels.yaml` | Grafana dashboard UIDs and panel IDs |
| `config/thresholds.yaml` | All numeric thresholds (IOPS limits, CPU throttle %, DNS latency) |
| `config/queries.yaml` | PromQL templates — edit here if your metric names differ |

## Finding your Grafana datasource UID

1. Open Grafana → Configuration → Data Sources
2. Click your Prometheus datasource
3. The UID is in the URL: `/datasources/edit/<UID>`

## Adding a new tool

1. Add the async function to `src/tools.py` following the existing pattern
2. Add its schema to `TOOL_SCHEMAS` at the bottom of `src/tools.py`
3. Register it in `_TOOL_FNS` in `src/agent.py`
4. Mention it in the system prompt in `src/prompts.py`
