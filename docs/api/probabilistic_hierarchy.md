# ProbabilisticHierarchyEngine — Four-Layer Bayesian Knowledge Evolution

**Module:** `omega_cube.probabilistic_hierarchy` (569 lines, 23,827 bytes)  
**Purpose:** Hierarchical truth management using Bayesian updating across four layers: IMMUTABLE (axioms), PROBABILISTIC (scientific facts), EMERGENT (behavioral patterns), FLUID (session data). Anchoring protection prevents low-trust evidence from shifting high-layer nodes.

## Architecture

Four hierarchy layers with different stability characteristics:

| Layer | Variance | Example | Behavior |
|-------|----------|---------|----------|
| **IMMUTABLE** | σ²≈0 | "Fire requires oxygen" | Infinite stability; cannot shift |
| **PROBABILISTIC** | σ²~100 | "Solar temperature ~5,500°C" | Can be refined by evidence |
| **EMERGENT** | σ²~400 | "Users connect ComfyUI+Evony" | Behavioral pattern, moderate stability |
| **FLUID** | σ²~1000 | "Current session: debugging SDXL" | Temporary, rapid adaptation |

## Class: `ProbabilisticNode`

### Structure

```python
class ProbabilisticNode:
    content: str                                    # Node description
    hierarchy: str                                  # Hierarchical path key
    layer: str                                      # IMMUTABLE/PROBABILISTIC/EMERGENT/FLUID
    mean: float                                     # Current truth estimate [0,1]
    variance: float                                 # Uncertainty (σ²)
    confidence: float                               # Derived from mean and stability
    is_immutable: bool                              # Layer-0 protection flag
    evidence_count: int                             # Number of updates applied
    source_signals: list[str]                       # Last 10 evidence sources
```

### Key Methods

| Method | Description |
|--------|-------------|
| `bayesian_update(evidence, evidence_variance, source)` | Apply Bayes' theorem to update mean/variance with new evidence; returns shift magnitude |
| `stability` (property) | IMMUTABLE → ∞; otherwise 1/variance |

### Bayesian Update Algorithm

```python
precision_prior = 1.0 / variance
precision_evidence = 1.0 / evidence_variance
total_precision = precision_prior + precision_evidence

# Precision-weighted average
self.mean = (mean * precision_prior + evidence * precision_evidence) / total_precision
self.variance = 1.0 / total_precision  # Posterior variance always lower than prior
```

- **Anchoring protection**: If `is_immutable=True`, shift returns 0.0 immediately — axiom cannot be moved by any evidence
- **Source tracking**: Last 10 sources stored; older ones dropped (FIFO)

## Class: `ProbabilisticHierarchyEngine`

### Initialization

```python
class ProbabilisticHierarchyEngine:
    nodes: dict[str, ProbabilisticNode]              # hierarchy → node
    updates_log: list[dict]                          # Per-update history
    total_updates: int                               # Total evidence applications
    total_shift: float                               # Cumulative mean shifts
```

### Source Variance Table (`SOURCE_VARIANCES`)

| Source Type | Variance | Trust Level | Example |
|-------------|----------|-------------|---------|
| AXIOM | 1e-10 | Absolute | Published theorem |
| PEER_REVIEWED | 50 | High | Published paper |
| BENCHMARK | 30 | Very high | Experimental result |
| NEW_PAPER | 100 | Medium-high | Recent publication |
| EXPERT_OPINION | 80 | Medium-high | Domain expert |
| COLLECTIVE_USAGE | 200 | Medium | Aggregated user behavior |
| USER_SESSION | 500 | Low | Single session observation |
| REAL_TIME_QUERY | 1000 | Very low | Current conversation |

### Node Creation Methods

| Method | Layer | Initial Mean | Variance | Use Case |
|--------|-------|-------------|----------|----------|
| `add_axiom()` | IMMUTABLE | 1.0 | 1e-10 | Absolute truths (fire needs oxygen) |
| `add_probabilistic(conf=0.9)` | PROBABILISTIC | confidence | (1-conf)*10 | Scientific facts |
| `add_emergent()` | EMERGENT | 0.5 | 0.8 | Behavioral patterns |
| `add_fluid()` | FLUID | 0.5 | 2.0 | Temporary session data |

### Core Method: `update_from_evidence(hierarchy, evidence, source_type, source_detail)`

```python
# Anchoring protection
layer_idx = LAYER_ORDER.index(node.layer)
source_trust = 1.0 / (1.0 + evidence_variance / 100)

if layer_idx <= 1 and source_trust < 0.5:
    # High-layer node + weak evidence → dampen impact
    evidence_variance *= 5.0

shift = node.bayesian_update(evidence, evidence_variance, source_detail)
```

- **Anchoring**: If updating IMMUTABLE/PROBABILISTIC nodes with low-trust evidence (USER_SESSION, REAL_TIME_QUERY), variance is multiplied by 5x to reduce impact
- **Auto-creation**: If hierarchy key doesn't exist, auto-creates as FLUID node
- **Logging**: Records per-update metadata (hierarchy, layer, mean_after, variance_after, shift, source)

### Collective Integration: `update_from_collective_engine(collective_engine)`

Feeds signals from `CollectiveHierarchyEngine`:

1. **Transitions → EMERGENT layer** — Each domain→domain transition becomes an EMERGENT node updated with COLLECTIVE_USAGE evidence
2. **Co-occurrences → PROBABILISTIC or EMERGENT** — Strong co-occurrences (normalized >0.3) become PROBABILISTIC nodes; weak ones stay EMERGENT
3. **Domain activity → FLUID layer** — High-frequency domain access creates FLUID nodes

### Research Paper Ingestion: `ingest_paper(title, findings, confidence=0.7)`

```python
for key, value in findings.items():
    hierarchy = f"PAPER.{title[:30].replace(' ', '_')}.{key}"
    self.add_probabilistic(f"From {title}: {key} = {value}", hierarchy, confidence)
    self.update_from_evidence(hierarchy, float(value), "NEW_PAPER", ...)
```

### Statistics

| Method | Returns |
|--------|---------|
| `get_layer_stats()` | Per-layer counts, avg confidence, avg variance across 4 layers |
| `get_hierarchy_snapshot()` | Full snapshot: all nodes with metadata + layer stats + totals |

### Persistence

```python
engine.save("probabilistic.json")   # JSON persistence of all nodes
engine.load("probabilistic.json")   # Restore from disk
```

## Demo: Layer Behavior Under Evidence Pressure

The `demo_probabilistic_hierarchy()` function demonstrates:

1. **IMMUTABLE resistance**: Axiom "Fire requires oxygen" attacked 1000 times with USER_SESSION evidence (shift < 0.001) — protected
2. **PROBABILISTIC refinement**: Solar temperature updated from paper ("5,772K") — mean shifts slightly, variance decreases
3. **EMERGENT promotion**: ComfyUI+Evony connection strengthened by 100 user sessions (confidence rises toward PROBABILISTIC level)
4. **FLUID adaptation**: Session-specific facts adapt rapidly but with high variance

## Benchmark: Immutable vs Flexible Protection

`benchmark_immutable_vs_flexible()` tests whether user sessions can shift an axiom:

```
Axiom attacked 1000 times: total shift = ~0 (protected ✅)
Probabilistic updated 1000 times: total shift > 0.01 (adaptable ✅)
```

This validates the anchoring protection mechanism works correctly — high-layer nodes resist low-trust evidence while remaining nodes adapt appropriately.

## Usage Example

```python
from omega_cube.probabilistic_hierarchy import ProbabilisticHierarchyEngine

engine = ProbabilisticHierarchyEngine()

# Add immutable axiom
engine.add_axiom("Fire requires oxygen", "SCIENCE.FIRE.REQUIREMENTS")

# Add probabilistic fact
engine.add_probabilistic(
    "SDXL produces higher quality than SD1.5",
    "AI.MODELS.SDXL.QUALITY", confidence=0.85
)

# Update with evidence from different sources
engine.update_from_evidence("SCIENCE.FIRE.REQUIREMENTS", 0.1, "USER_SESSION", 
                            "someone claims fire doesn't need oxygen")
# → IMMUTABLE: shift = 0 (protected)

engine.update_from_evidence("AI.MODELS.SDXL.QUALITY", 0.92, "BENCHMARK",
                            "LoCoMo benchmark 2026")
# → PROBABILISTIC: mean shifts toward 0.92, variance decreases
```
