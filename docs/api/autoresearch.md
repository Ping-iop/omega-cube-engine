# AutoResearchLoop — Autonomous Graph Optimization

**Module:** `omega_cube.autoresearch` (280 lines, 10,566 bytes)  
**Purpose:** Self-optimization loop that autonomously experiments with graph topology and encoding parameters.  
**Author:** Omega-Cube Research | **Date:** 2026-06-11

## Overview

Integrates Karpathy's AutoResearch pattern with the Omega-Cube engine: an agent loop that runs overnight, evaluating modifications against benchmarks and keeping only improvements. Pattern: modify → evaluate → compare → keep or rollback. In a single night can run 50+ experiments across topology, encoding dimensions, annealing parameters, diffusion steps, and gray-scale weights.

## Class: `AutoResearchLoop`

### Initialization

```python
class AutoResearchLoop:
    def __init__(
        self,
        engine: OmegaCubeEngine,           # Engine to optimize
        benchmark_fn: Callable[[], dict],  # Returns composite score dict
        save_fn: Callable[[], None],       # Persists best config
        experiment_dir: str = None,        # Defaults to ~/.hermes/axioma-omega-protocol/experiments/
    )
```

### Configuration Space (`param_space`)

| Parameter | Values |
|-----------|--------|
| `holographic_dim` | [64, 128, 256, 512] |
| `anneal_temp` | [0.5, 1.0, 2.0, 5.0] |
| `cooling_rate` | [0.90, 0.93, 0.95, 0.97, 0.99] |
| `diffusion_steps` | [10, 15, 20, 30, 50] |
| `guidance_scale` | [1.0, 2.0, 3.0, 5.0] |
| `gray_weights_factuality` | [0.25, 0.30, 0.35, 0.40, 0.50] |
| `gray_weights_relevance` | [0.15, 0.20, 0.25, 0.30] |
| `tensor_grid_size` | [5, 8, 10, 15, 20] |

### Main Method: `run()`

```python
def run(
    self,
    num_experiments: int = 50,         # Max experiments to run
    max_hours: float = 8.0,            # Wall-clock time budget
    early_stop_patience: int = 15,     # Stop if no improvement for N experiments
) -> dict
```

Returns summary of best configuration found:
```python
{
    "best_score": 0.8734,
    "best_config": {"holographic_dim": 256, "anneal_temp": 1.0, ...},
    "total_experiments": 47,
    "improvements": 8,
    "elapsed_hours": 6.3,
}
```

### Modification Types

| Type | Method | Description |
|------|--------|-------------|
| `topology_split` / `topology_merge` | `_mod_topology()` | Split or merge graph nodes |
| `holographic_dim_{n}` | `_mod_holographic()` | Change encoding dimension size |
| `annealer_params` | `_mod_annealer()` | Adjust initial temp + cooling rate |
| `diffusion_params` | `_mod_diffusion()` | Change diffusion steps + guidance scale |
| `grayscale_weights` | `_mod_grayscale()` | Adjust factuality vs relevance weights |

### Experiment Lifecycle

```
1. Snapshot current state for rollback safety
2. Evaluate baseline score via benchmark_fn()
3. For each experiment:
   a. Generate random modification (uniform choice from 5 types)
   b. Apply modification to engine attributes
   c. Re-evaluate with benchmark_fn()
   d. If improved → keep, save(), reset patience counter
   e. If not improved → rollback to snapshot, increment patience
   f. Log experiment metadata (id, name, score, delta, timestamp)
4. After loop: restore best config, persist, save log
5. Return summary dict with scores and elapsed time
```

### Experiment Logging

Each experiment is logged as a dict:
```python
{
    "exp_id": 12,
    "modification": "holographic_dim_512",
    "score": 0.8912,
    "best_score": 0.8734,
    "improved": True,
    "timestamp": 1686432000.0,
}
```

Saved periodically (every 10 experiments) to `experiments/autoresearch_log.json`.

## Usage Example

```python
from omega_cube.autoresearch import AutoResearchLoop

# Define a benchmark function that returns composite scores
def my_benchmark():
    return {"composite_score": engine.evaluate_quality()}

# Create the auto-research loop
loop = AutoResearchLoop(
    engine=engine,
    benchmark_fn=my_benchmark,
    save_fn=lambda: engine.save(),
)

# Run overnight optimization
results = loop.run(num_experiments=100, max_hours=8.0, early_stop_patience=20)
print(f"Best score: {results['best_score']:.4f} in {results['elapsed_hours']:.1f}h")
```
