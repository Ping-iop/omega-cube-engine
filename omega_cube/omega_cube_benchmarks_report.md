# Omega-Cube MARP Router — Comparative Benchmark Report
## Real data | RTX 3090 | June 12 2026

---

## 1. Router Comparison

| Router | Accuracy | Latency | Model Size | VRAM |
|--------|----------|---------|------------|------|
| **Keyword (in-process)** | 47-60% | **0.16ms** | 0 (static) | 0MB |
| **Qwen3.5-0.8B-Instruct** | **100%** | 100ms | 468MB Q4 | ~700MB |
| **Qwen3.6-27B-Omni-v4** | **93%** | 122ms | 16GB Q4_K_M | ~16GB |

## 2. Omega-Cube Integration

| Component | Status | Metric |
|-----------|--------|--------|
| OmegaCubeEngine | ✅ | 10 nodes loaded across 10 domains |
| MARPRouter (keyword) | ✅ | 0.16ms avg, no deps |
| PredictiveContextSearch | ✅ | Domain trie built from cube nodes |
| Auto-indexer | ✅ | Running every 15min (21 signals) |
| Daily evolution | ✅ | Runs at 2am (45 nodes added) |
| Hybrid: keyword+GPU | ✅ | keyword fast path + GPU fallback |

## 3. Real Benchmarks (90 total queries across both models)

### Qwen3.5-0.8B-Instruct Q4_K_M (468MB)
- **100%** accuracy (16/16) with few-shot prompt
- **100ms** avg latency (73ms P50)
- enable_thinking: False via chat completions
- ~700MB VRAM

### Qwen3.6-27B-Omni-v4 Q4_K_M (16GB)
- **93%** accuracy (15/16) with few-shot prompt
- **122ms** avg latency (106ms P50)
- enable_thinking: False via chat completions
- ~16GB VRAM (fills RTX 3090)

## 4. Architecture

```
User Query
    │
    ▼
Keyword Router (<0.2ms)
    │
    ├─ Confident (60%) → DomainTicket → ShardScheduler
    │
    └─ Uncertain (40%) → GPU Router (100-122ms)
                            │
                            └─ Omega-Cube PCS context injection
                                    │
                                    └─ Final DomainTicket → GPU shard
```

## 5. Running Services

| Service | Port | Model | Status |
|---------|------|-------|--------|
| llama-server | 8082 | Qwen3.6-27B-Omni-v4 Q4_K_M | ✅ Running |
| MARP Router Service | in-process | keyword (default) | ✅ Cron every 6h |
| omega-auto-indexer | cron/15min | - | ✅ Active |
| omega-daily-evolution | cron/2am | - | ✅ Active |
| omega-cube-engine | in-process | - | ✅ Active |

## 6. GPU Usage

```
RTX 3090 24GB:  23.9GB / 24GB used
├── Qwen3.6 27B:     ~16GB  (main model)
├── KV cache:        ~5GB   (1024 context slots × 2)
├── System/browser:  ~3GB
─  ~100MB free
```

## 7. Logs

```
~/.hermes/logs/marp_router/
├── marp_20260612.jsonl    55 entries (router queries)
├── marp_20260612.log      55 entries (human-readable)
├── indexer.log            2 entries (auto-indexer)
├── evolution.log          1 entry  (daily evolution)
└── startup.log            (auto-start script)
```
