# QuantumInspiredAnnealer — Topology Optimization via Simulated Annealing

**Source:** `omega_cube/annealer.py` (250 lines, 9,385 bytes)  
**Author:** Omega-Cube Research  
**Date:** 2026-06-11

## Overview

Implements topology optimization using simulated annealing with quantum-inspired tunneling. Each "cube" (topic domain subgraph) rotates through its dimensions simultaneously; the system converges to a minimum-energy configuration where all cubes are aligned for optimal query response.

**Key properties:**
- **Non-deterministic**: Same query can find novel patterns across runs
- **Parallel**: All cubes anneal simultaneously, not sequentially
- **Emergent**: Patterns arise from local interactions, not central planning

## Classes

### `QuantumInspiredAnnealer` — Main Annealing Engine

#### Initialization

```python
from omega_cube.annealer import QuantumInspiredAnnealer

annealer = QuantumInspiredAnnealer(
    initial_temp=1.0,        # Starting temperature (default: 1.0)
    cooling_rate=0.95,       # Temperature decay factor per step (0 < rate < 1)
    min_temp=0.01,           # Stop condition when temp drops below this
    steps_per_temp=5,        # Neighbor evaluations per temperature level
    tunneling_prob=0.1,      # Probability of quantum tunneling jump (default: 0.1)
    seed=42,                 # Deterministic RNG seed for reproducibility
)
```

#### `anneal(cubes, energy_fn, neighbor_fn, max_iterations=500)` → `(best_config, best_energy, energy_history)`

Runs simulated annealing to find optimal cube configuration. Uses Metropolis criterion with quantum tunneling for escaping local minima.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cubes` | `list[dict]` | Yes | Initial cube states (each dict has config fields like `topic`, `alignment`, `position`) |
| `energy_fn` | `Callable[[list[dict]], float]` | Yes | Scoring function: lower = better configuration |
| `neighbor_fn` | `Callable[[list[dict]], list[dict]]` | Yes | Generates a neighboring configuration from current state |
| `max_iterations` | `int` | No | Maximum annealing steps (default: 500) |

**Returns:** Tuple of `(best_config, best_energy, energy_history)` where `energy_history` tracks energy across all iterations for convergence analysis.

```python
# Define cube states representing topic domains
cubes = [
    {"topic": "COMFYUI", "alignment": 0.75, "position": [0.3, 0.6]},
    {"topic": "EVONY", "alignment": 0.82, "position": [0.5, 0.4]},
    {"topic": "HBIT", "alignment": 0.68, "position": [0.7, 0.8]},
]

# Energy: minimize deviation from optimal alignment + clustering penalty
def energy_fn(config):
    base = sum((c["alignment"] - 0.85)**2 for c in config)
    if len(config) > 1:
        positions = [c["position"] for c in config]
        avg = [sum(p[i] for p in positions) / len(positions) for i in range(2)]
        spread = sum((p[0]-avg[0])**2 + (p[1]-avg[1])**2 for p in positions)
        return base + 0.1 * spread
    return base

# Neighbor: perturb one cube's position randomly
def neighbor_fn(current):
    import random
    new = [dict(c) for c in current]
    idx = random.randint(0, len(new)-1)
    new[idx]["position"] = [random.uniform(0,1), random.uniform(0,1)]
    return new

best_config, best_energy, history = annealer.anneal(cubes, energy_fn, neighbor_fn)
```

#### `multi_objective_anneal(cubes, energy_fns, neighbor_fn, max_iterations=500)` → `(best_config, energy_history)`

Multi-objective annealing with weighted combination of multiple energy functions. Each function captures a different quality criterion (relevance, coherence, novelty, efficiency).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cubes` | `list[dict]` | Yes | Initial cube states |
| `energy_fns` | `list[tuple[Callable, float]]` | Yes | List of `(function, weight)` pairs for multi-objective optimization |
| `neighbor_fn` | `Callable` | Yes | Neighbor generator |

```python
# Optimize for both alignment and diversity simultaneously
energy_fns = [
    (lambda c: -sum(c["alignment"] for c in c), 0.7),   # Maximize alignment (weight 0.7)
    (lambda c: sum(c.get("diversity_bonus", 0) for c in c), 0.3),  # Maximize diversity (weight 0.3)
]

best_config, history = annealer.multi_objective_anneal(cubes, energy_fns, neighbor_fn)
```

#### Quantum Tunneling Mechanism

The `_tunnel` method implements quantum-inspired tunneling: when `random.random() < tunneling_prob`, instead of evaluating a single neighbor, the algorithm makes 2–5 consecutive random moves. This simulates quantum tunneling through energy barriers, escaping local minima that classical simulated annealing would get stuck in.

```python
# Tunnel = multiple random steps to escape local minimum
def _tunnel(self, cubes, neighbor_fn):
    result = [dict(c) for c in cubes]
    for _ in range(random.randint(2, 5)):
        result = neighbor_fn(result)
    return result
```

---

### `CubeRotator` — Individual Cube Rotation Manager

Manages rotation of individual cubes through their hierarchy dimensions. A "rotation" changes which dimension is primary, which subtopic is exposed, and which associations are active.

#### `random_rotation(cube)` → `dict`

Randomly rotates a cube by selecting random active dimension, exposed subtopic, and association state.

```python
from omega_cube.annealer import CubeRotator

cube = {
    "dimensions": ["COMFYUI", "EVONY", "HBIT"],
    "subtopics": ["SDXL", "Marcian", "GrayScale"],
    "associations": {"related_1": True, "related_2": False},
}

rotated = CubeRotator.random_rotation(cube)
# → Randomly selects one dimension as active, one subtopic as exposed, 
#   randomly toggles associations
```

#### `query_aligned_rotation(cube, query_vector)` → `dict`

Deterministic rotation that aligns the cube's primary dimension with the query vector. Scores each dimension against the query using cosine similarity and selects the best match.

```python
cube = {
    "dimension_vectors": [[1,0,0], [0,1,0], [0,0,1]],  # One basis vector per dimension
}

query_vec = [0.8, 0.3, 0.2]  # Query leaning toward first dimension
aligned = CubeRotator.query_aligned_rotation(cube, query_vec)
# → Sets active_dimension to the index of highest cosine similarity
```

---

### `PatternEmergence` — Cross-Cube Pattern Detection

Detects emergent patterns where multiple cubes independently settle into compatible states after annealing. A "pattern" is a multi-topic answer formed by aligned cube configurations.

#### `extract_patterns(cubes, threshold=0.7)` → `list[dict]`

Extracts cross-cube patterns from an annealed configuration. Returns patterns sorted by strength (highest first).

```python
from omega_cube.annealer import PatternEmergence

patterns = PatternEmergence.extract_patterns(annealed_cubes, threshold=0.7)

for p in patterns[:5]:
    print(f"Pattern: {p['cube_topic']} (strength: {p['pattern_strength']:.3f})")
    for aligned in p["aligned_cubes"][:3]:
        print(f"  ↔ Cube {aligned['cube_id']} (alignment: {aligned['alignment']:.3f})")
```

**Return format:**
```python
{
    "anchor_cube": "COMFYUI",
    "cube_topic": "Image Generation",
    "aligned_cubes": [
        {"cube_id": "HBIT", "alignment": 0.85},
        {"cube_id": "HERMES", "alignment": 0.72},
    ],
    "pattern_strength": 0.785,  # Average alignment of aligned cubes
    "exposed_content": "SDXL checkpoint optimization..."
}
```

#### `_alignment_score(v1, v2)` → `float`

Internal utility: computes normalized alignment between two vectors, mapping cosine similarity from [-1, 1] to [0, 1].

---

## Configuration Reference

| Parameter | Default | Impact |
|-----------|---------|--------|
| `initial_temp` | 1.0 | Higher = more exploration at start; lower = faster convergence but risk of local minima |
| `cooling_rate` | 0.95 | Lower = faster cooling (less late-stage exploration); higher = slower convergence |
| `min_temp` | 0.01 | Stop condition: annealing terminates when temperature drops below this threshold |
| `steps_per_temp` | 5 | More steps per temperature = more thorough search at each level |
| `tunneling_prob` | 0.1 | Higher = more frequent tunneling jumps (better for complex energy landscapes) |

## Integration with Engine

The engine uses annealing in several contexts:
- **Diffusion sampler**: Anneals diffusion parameters for optimal sampling trajectory
- **Topology optimization**: AutoResearch loop uses annealing to find optimal graph configurations
- **Pattern emergence**: After query, cubes are annealed and patterns extracted via `PatternEmergence.extract_patterns()`

## Dependencies

- Standard library: `math`, `random`
- No external dependencies required
