# Axion-Cube Engine v2.0

**14-Component Hierarchical Memory + Model Routing System**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-green)](https://python.org)

> *"Like magnetic metallic spheres forming cubes — mini-universes that rotate and adjust simultaneously, connected by hierarchy and topic, forming patterns to answer complex questions."*

---

## What is Axion-Cube?

Axion-Cube is the **knowledge engine** of the AXION protocol (AXiomatic ONtological engine).
A multi-dimensional hierarchical graph memory system that stores, searches, verifies,
and routes knowledge. 14 innovations in one engine (10 original + 4 from arXiv 2026).

| # | Component | Innovation |
|---|-----------|------------|
| 1 | **TensorNode** | N-dimensional simultaneous hierarchies |
| 2 | **HolographicEncoder** | O(1) circular convolution search (256D) |
| 3 | **QuantumAnnealer** | Dynamic topology optimization |
| 4 | **DiffusionSampler** | Parallel non-autoregressive retrieval |
| 5 | **GrayScaleValidator** | Multi-bit truth (6 dimensions, H-Bit) |
| 6 | **AutoResearchLoop** | Self-optimizing weekly pipeline |
| 7 | **PredictiveContextSearch** | Domain-aware prefix trie |
| 8 | **CollectiveHierarchy** | Session-driven graph evolution |
| 9 | **ProbabilisticHierarchy** | 4-layer Bayesian anchoring |
| 10 | **MARPRouter** | Model-Agnostic Routing Protocol |
| 11 | **HierarchicalSummarizer** | H²MT coarse-to-fine O(log n) routing |
| 12 | **TypedSchema** | VirtualSet pre-execution validation |
| 13 | **BoundaryController** | PAGE-RAG grounding filter |
| 14 | **HallucinationDetector** | Inference misalignment bias detection |

## MARP v2: Model-Agnostic Routing Protocol

**6 improvements applied (2026-07-26):**

1. **Hierarchical routing** — O(log n) via HierarchicalSummarizer (H²MT)
2. **Grounded context** — BoundaryController filters ungrounded nodes (PAGE-RAG)
3. **Bias detection** — HallucinationDetector in domain classification
4. **Holographic context nodes** — 256D embeddings in every ContextNode
5. **AdaptiveScheduler** — Session-based domain frequency learning
6. **Evolving keywords** — Graph-driven keyword extraction (CORTEX)

### Benchmarks (real data, 2026-07-26)

| Metric | v1 | v2 | Delta |
|--------|-----|-----|-------|
| Routing accuracy | 52.5% | **60.0%** | +7.5% |
| Context nodes/query | 0.6 | **7.3** | 12x |
| Hierarchical routing | 0% | **100%** | New |
| Holographic context | 0% | **100%** | New |
| Adaptive prefetch | 0 hits | **8 hits** | New |
| Prediction accuracy | N/A | **62.5%** | New |

---

## Quick Start

```python
from omega_cube import OmegaCubeEngine
from omega_cube.marp import MARPRouter, ShardScheduler
from omega_cube.marp.protocol import ShardConfig, MARPMode

# Memory engine
engine = OmegaCubeEngine()
engine.add_node("Calculus fundamentals", ["math.calculus"], [0.8, 0.5])

# MARP Router (the "clerk")
router = MARPRouter()
shards = [
    ShardConfig(name="math_v1", domains=["math"],
                mode=MARPMode.WRAPPER, base_model="gemma-4-31b",
                adapter_type="lora", gpu_memory_mb=8000),
]

# Route query to domain shard
decision = router.route("What is the derivative of x^2?", shards)
print(f"Active shards: {decision.active_shards}")  # ['math_v1']
print(f"Token savings: {decision.token_savings_estimate:.0%}")  # 62%
print(f"Routing time: {decision.routing_time_ms:.3f}ms")  # 0.079ms
```

---

## Modelo Router Local (Qwen3.5-0.8B Q6 — real, funcionando)

| Métrica | Valor |
|---------|-------|
| Modelo | Qwen3.5-0.8B-Q6_K.gguf |
| Tamaño | 639MB (Q6_K cuantizado) |
| Accuracy | **100%** (16/16 domain classification, few-shot prompt) |
| Latencia GPU | **100ms avg, 73ms P50** (via llama-server HTTP) |
| Latencia keyword | **0.079ms** (pre-filtro, 64% queries) |
| Ubicación | `J:/modelos_ia/Qwen3.5-0.8B-Q6_K.gguf` |
| Servicio | `marp/router_service.py` |
| Logs | `~/.hermes/logs/marp_router/` (JSONL diario) |

---

## Benchmarks

### PredictiveContextSearch
```
160 trials, 8 domains, 240 nodes
Predictive: 160/160 = 100% | Flat: 80/160 = 50%
Latency: 0.057ms (O(k))
```

### Retrieval Modes
```
Mode          P@5    Time
holographic   17%    3.7ms  (108x faster than diffusion)
diffusion     22%    400ms
```

### MARP Router (local execution)
```
100 queries, 10 domain shards
Avg latency:   0.079ms
P99 latency:   1.237ms
Throughput:    12,626 q/sec
Token savings: 62.2%
```

### H-Bit Spectrum (local execution)
```
512x512 PNG, crop robustness
100% image → 331/331 tiles, 98.3% confidence
 25% image →  82/82 tiles, 98.3% confidence
  3% image →   9/9  tiles, 98.3% confidence ← solo 15 filas!
```

---

## Comparison with State-of-the-Art

| Feature | GAM | All-Mem | MemVerse | **Omega-Cube v1.5** |
|---------|-----|---------|----------|---------------------|
| Hierarchical Graph | ✅ | ✅ | ✅ | ✅ |
| Dynamic Topology | ❌ | ✅ | ❌ | ✅ Annealing |
| Multi-Dimensional | ❌ | ❌ | ❌ | ✅ Tensor |
| Holographic Encoding | ❌ | ❌ | ❌ | ✅ |
| Diffusion Retrieval | ❌ | ❌ | ❌ | ✅ |
| Multi-Bit Verification | ❌ | ❌ | ❌ | ✅ H-Bit |
| Auto-Optimization | ❌ | ❌ | ❌ | ✅ AutoResearch |
| Predictive Context | ❌ | ❌ | ❌ | ✅ 100% |
| Collective Evolution | ❌ | ❌ | ❌ | ✅ |
| Probabilistic Hierarchy | ❌ | ❌ | ❌ | ✅ 4-layer |
| **Model Routing** | ❌ | ❌ | ❌ | **✅ MARP** |
| Model-Agnostic | ✅ | ✅ | ✅ | ✅ |

---

## Installation

```bash
git clone https://github.com/Ping-iop/omega-cube-engine.git
cd omega-cube-engine
# Zero dependencies (Python stdlib only!)
python -c "from omega_cube.marp import MARPRouter; print('Ready')"
```

---

## Repository

```
omega-cube-engine/
├── omega_cube/
│   ├── engine.py              # OmegaCubeEngine (10 components)
│   ├── marp/                  # MARP Router (Component #10)
│   │   ├── router.py          #   Query → domain classification
│   │   ├── protocol.py        #   DomainTicket, ShardConfig
│   │   └── scheduler.py       #   GPU-native shard activation
│   ├── tensor_node.py         # TensorNode + TensorIndex
│   ├── holographic.py         # HolographicEncoder
│   ├── annealer.py            # QuantumInspiredAnnealer
│   ├── diffusion_sampler.py   # DiffusionGraphSampler
│   ├── grayscale.py           # GrayScaleValidator
│   ├── predictive_search.py   # PredictiveContextSearch
│   ├── collective_evolution.py
│   ├── probabilistic_hierarchy.py
│   └── autoresearch.py
├── benchmarks/
│   ├── final_benchmark_data.json  # Raw local execution data
│   ├── comparative_benchmarks.json # External citations
│   └── comparative_external.py
├── omega_cube_paper.pdf        # Academic paper
├── omega_cube_benchmarks_v15.pdf # Charts + sources
├── PAPER.md
└── README.md
```

---

## Citation

```bibtex
@article{omega-cube-2026,
  title={Omega-Cube v1.5: Multi-Dimensional Memory + Model-Agnostic Routing},
  author={Omega-Cube Research},
  year={2026},
  url={https://github.com/Ping-iop/omega-cube-engine}
}
```

---

*Built with passion by Omega-Cube Research — June 2026*
