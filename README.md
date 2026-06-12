# Omega-Cube Engine 🧲

**Multi-Dimensional Hierarchical Graph Memory System**

[![arXiv](https://img.shields.io/badge/arXiv-pending-red)](https://arxiv.org)
[![Zenodo](https://img.shields.io/badge/Zenodo-DOI-blue)](https://zenodo.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)

> *"Like magnetic metallic spheres forming cubes — mini-universes that rotate and adjust simultaneously, connected by hierarchy and topic, forming patterns to answer complex questions."*

## 📄 Paper

Read the full paper: [omega_cube_paper.pdf](omega_cube_paper.pdf) | [LaTeX source](paper.tex)

**Omega-Cube: A Multi-Dimensional Hierarchical Graph Memory System**  
*Omega-Cube Research, June 2026*

---

## 🚀 What is Omega-Cube?

Omega-Cube is the next evolution of hierarchical graph memory for AI agents. It solves the **context decay problem** in long LLM conversations by representing knowledge as **multi-dimensional tensor nodes** — like magnetic Rubik's cubes that can be accessed from multiple perspectives simultaneously.

### 5 Innovations (All in One Engine)

| # | Innovation | Description |
|---|---|---|
| 1 | **Tensor Hierarchies** | Nodes exist in N simultaneous hierarchy dimensions |
| 2 | **Holographic Encoding** | O(1) approximate retrieval via circular convolution |
| 3 | **Quantum-Inspired Annealing** | Dynamic topology optimization ("rotating cubes") |
| 4 | **Diffusion Graph Sampling** | Parallel non-autoregressive retrieval (inspired by DiffusionGemma) |
| 5 | **Gray-Scale Validation** | Multi-bit truth assessment (H-Bit inspired) |

### Why This Matters

```
Traditional RAG:  Query → Flat vector search → Noisy results
Omega-Cube:       Query → Diffusion over N-dim tensor space → Ranked + clustered + verified
```

- **108× faster** holographic mode vs full diffusion (3.7ms vs 400ms)
- **Multi-topic queries** return results organized by domain
- **Emergent patterns** across domains without explicit programming
- **Model-agnostic**: plugs into any LLM agent via MCP

---

## 🏗️ Architecture

```
                    ┌──────────────────────────┐
                    │    Omega-Cube Engine      │
                    └──────────┬───────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼─────┐    ┌──────────▼──────────┐    ┌──────▼──────┐
   │  Tensor  │    │    Holographic      │    │  Quantum    │
   │  Nodes   │    │    Encoder          │    │  Annealer   │
   │  N-dim   │    │  Circular conv.     │    │  Topology   │
   └──────────┘    └─────────────────────┘    └─────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Diffusion Sampler   │
                    │  (Parallel, not      │
                    │   autoregressive)    │
                    └─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Gray-Scale Valid.   │
                    │  (Multi-bit truth)   │
                    └─────────────────────┘
```

---

## 📦 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/omega-cube/omega-cube-engine.git
cd omega-cube-engine

# No dependencies needed (Python stdlib only!)
python -c "from omega_cube import OmegaCubeEngine; print('✅ Ready')"
```

### Basic Usage

```python
from omega_cube import OmegaCubeEngine

# Create engine
engine = OmegaCubeEngine()

# Add knowledge with multi-dimensional hierarchies
engine.add_node(
    content="SDXL produces higher quality images than SD1.5 but requires more VRAM",
    hierarchies=[
        "COMFYUI.WORKFLOWS.GENERATION.SDXL",
        "QUALITY.IMAGE_RESOLUTION.HIGH",
        "MODELS.DIFFUSION.COMPARISON"
    ],
    tensor_position=[0.72, 0.85, 0.63],
    node_type="CONCEPT",
    confidence=0.92
)

# Query in different modes
results = engine.query("best image generation model", mode="diffusion")
for r in results:
    print(f"[{r['score']:.3f}] {r['content'][:100]}")

# Multi-topic query
multi = engine.query_multi_topic(
    "optimization techniques",
    topics=["COMFYUI", "ML", "HERMES"],
)

# Find emergent patterns
patterns = engine.find_patterns("cross-domain integration")
for p in patterns:
    print(f"Pattern: {p['cube_topic']} (strength: {p['pattern_strength']:.3f})")
```

### Retrieval Modes

| Mode | Speed | Precision | Best For |
|---|---|---|---|
| `holographic` | ⚡ 3.7ms | Moderate | Real-time, first-pass filtering |
| `tensor` | ⚡ 3.8ms | Moderate | Spatial proximity queries |
| `diffusion` | 🐢 400ms | Higher | Precision-critical retrieval |
| `combined` | 🐢 422ms | Higher | Best overall accuracy |
| `annealing` | 🐢 Varies | Variable | Pattern discovery |

---

## 🧪 Benchmarks

### Predictive Context Search (NEW — June 2026)

```
Test: 160 context-switching trials, 8 domains, 240 nodes

PredictiveContextSearch:  160/160 = 100.0% accuracy
Flat search (no context):  80/160 =  50.0% accuracy

Δ: +50.0 pts (2.0x advantage)
Latency: 0.057ms avg (O(k), independent of corpus size)
```

**Why flat fails:** Without context, the same prefix always returns the same first alphabetical match. "Ma" in an Evony conversation should return "Marcian" — but flat search returns "Macrophage" if MEDICAL is alphabetically first. Predictive re-ranks based on the active conversation domain.

### Retrieval Modes (synthetic benchmark)

```bash
python omega_cube/benchmark.py
```

The benchmark creates a 5-domain, 55-node graph and evaluates 20 queries across 4 modes. Results on synthetic data:

```
Mode                 P@5     P@10   Avg Time
─────────────────  ──────  ──────  ────────
diffusion          22.0%   13.0%    400.6ms
holographic        17.0%   14.5%      3.7ms
tensor             17.0%   14.5%      3.8ms
combined           22.0%   14.5%    421.9ms

Pattern Emergence: 13 patterns found (95.6% avg alignment)
```

---

## 🔬 AutoResearch Self-Optimization

Omega-Cube includes an autonomous experimentation loop:

```python
from omega_cube import AutoResearchLoop

loop = AutoResearchLoop(
    engine=engine,
    benchmark_fn=benchmark_retrieval,
    save_fn=engine.save,
)

# Run 50 experiments overnight
results = loop.run(num_experiments=50, max_hours=8.0)
print(f"Best score: {results['best_score']}")
print(f"Improvements: {results['improvements']}/{results['total_experiments']}")
```

---

## 🔌 MCP Server Integration

Omega-Cube exposes all functionality via MCP for plug-and-play use with any MCP-compatible agent:

```bash
# Register with Hermes
hermes mcp add --command python --args "omega_cube_mcp_server.py" omega-cube
```

Tools exposed:
- `omega_cube_query` — Multi-mode retrieval
- `omega_cube_multi_topic` — Per-topic parallel retrieval  
- `omega_cube_patterns` — Cross-domain pattern detection
- `omega_cube_learn` — Multi-dimensional knowledge ingestion
- `omega_cube_stats` — Engine statistics

---

## 📊 Comparison with Related Work

| Feature | GAM (Apr '26) | All-Mem (Mar '26) | MemVerse (Jun '26) | Omega-Cube |
|---|---:|---:|---:|---:|
| Hierarchical Graph | ✅ | ✅ | ✅ | ✅ |
| Dynamic Topology | ❌ | ✅ SPLIT/MERGE | ❌ | ✅ Annealing |
| Multi-Dimensional | ❌ | ❌ | ❌ | ✅ Tensor |
| Holographic Encoding | ❌ | ❌ | ❌ | ✅ |
| Diffusion Retrieval | ❌ | ❌ | ❌ | ✅ |
| Multi-Bit Verification | ❌ | ❌ | ❌ | ✅ H-Bit |
| Auto-Optimization | ❌ | ❌ | ❌ | ✅ AutoResearch |
| Predictive Context Search | ❌ | ❌ | ❌ | ✅ 100% vs 50% flat |
| Collective Hierarchy Evolution | ❌ | ❌ | ❌ | ✅ 1,064 signals |
| Probabilistic Hierarchy | ❌ | ❌ | ❌ | ✅ 4-layer Bayesian |
| Model-Agnostic | ✅ | ✅ | ✅ | ✅ |

---

## 🗺️ Roadmap

### v1.1 — Performance
- [ ] Hierarchical pruning for diffusion (O(n) → O(log n))
- [ ] GPU-accelerated holographic encoding
- [ ] Streaming ingestion for real-time conversations

### v1.2 — Integration
- [ ] LangChain / LlamaIndex plugins
- [ ] Web UI for graph visualization
- [ ] REST API endpoint

### v2.0 — Neuro-Symbolic
- [ ] Parametric distillation (MemVerse-style)
- [ ] Pre-training with hierarchical objective (1-3B model)
- [ ] H-Bit deep integration (bit-level gray-scale)

### v3.0 — Distributed
- [ ] Cube swarms (each cube = independent agent)
- [ ] Emergent consensus (replaces centralized annealing)
- [ ] Cross-instance knowledge sharing

---

## 📖 Citation

```bibtex
@article{omega-cube-2026,
  title={Omega-Cube: A Multi-Dimensional Hierarchical Graph Memory System},
  author={Omega-Cube Research},
  year={2026},
  url={https://github.com/Ping-iop/omega-cube-engine}
}
```

---

## 🙏 Acknowledgments

This project builds upon the shoulders of giants. Special gratitude to:

**Andrej Karpathy** — for [AutoResearch](https://github.com/karpathy/autoresearch), [llm.c](https://github.com/karpathy/llm.c), [nanoGPT](https://github.com/karpathy/nanoGPT), [micrograd](https://github.com/karpathy/micrograd), and a decade of relentless public-domain AI education. Your tools, videos, and open-source ethos have democratized AI research and inspired an entire generation of builders who would otherwise never have had access to this field. AutoResearch's autonomous experimentation loop directly inspired Omega-Cube's self-optimization engine. The magnetic cube metaphor — knowledge domains rotating and aligning like Rubik's cubes — was born from the same spirit of playful exploration you embody.

**Google DeepMind** — for [DiffusionGemma](https://deepmind.google/models/gemma/diffusiongemma/), proving that diffusion models can generate coherent text non-autoregressively, inspiring Omega-Cube's parallel graph sampling.

**Nous Research** — for Hermes Agent, the platform that runs this entire ecosystem.

**The open-source AI community** — every paper on arXiv, every public dataset, every shared model weight, and every `pip install` that makes independent research possible.

---

*"Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. This repo is the story of how it all began."*  
— @karpathy, AutoResearch README, March 2026

---

## 📁 Repository Structure

```
omega-cube-engine/
├── omega_cube/
│   ├── __init__.py              # Package entry
│   ├── engine.py                # OmegaCubeEngine (unified)
│   ├── tensor_node.py           # TensorNode + TensorIndex
│   ├── holographic.py           # HolographicEncoder
│   ├── annealer.py              # QuantumInspiredAnnealer
│   ├── diffusion_sampler.py     # DiffusionGraphSampler
│   ├── grayscale.py             # GrayScaleValidator
│   ├── autoresearch.py          # AutoResearchLoop
│   ├── benchmark.py             # Benchmark suite
│   └── paper.tex                # LaTeX paper source
├── paper/
│   └── omega_cube_paper.pdf     # Compiled paper
├── README.md
└── LICENSE
```

---

**Built with ❤️ by Omega-Cube Research**  
*"From magnetic cubes to multi-dimensional memory."*
