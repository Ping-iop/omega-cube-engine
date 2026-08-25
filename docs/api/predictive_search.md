# PredictiveContextSearch — Context-Aware Auto-Complete

**Module:** `omega_cube.predictive_search` (467 lines, 19,910 bytes)  
**Purpose:** Hierarchical auto-complete search with domain-awareness and gray-scale boosting. Solves the "Windows search" problem: type "Ma" in a ComfyUI conversation → no Marcian/Matrix results; type "Ma" in Evony context → Marcian first.

## Architecture

Three cooperating components:
1. **HierarchicalTrie** — O(k) prefix lookup with per-node domain tracking and frequency scoring
2. **ContextTracker** — Sliding window of recent domains with exponential decay weighting
3. **PredictiveContextSearch** — Orchestrator combining both + Omega-Cube gray-scale boosting

## Class: `HierarchicalTrieNode`

### Structure

```python
class HierarchicalTrieNode:
    children: dict[str, 'HierarchicalTrieNode']  # char → child node
    best_match: Optional[str]                    # Best full-text match at this prefix
    total_hits: int                              # Total insertions through this node
    domain_hits: dict[str, int]                  # Per-domain hit counts
```

### Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `insert()` | `(text: str, domain: str)` | Insert text with domain; updates all ancestor nodes' counts and best_match (most frequent wins ties) |
| `search()` | `(prefix: str, context_domain=None, max_results=10) → list[dict]` | DFS prefix lookup collecting completions with scoring |
| `stats()` | `() → dict` | Returns total_nodes, root_children, root_total_hits |

### Scoring Algorithm (`_contextual_score`)

```
score = normalized_frequency * 0.5
      + context_boost (3.0 if primary match in domain, else ratio*2.0)
      - cross_domain_penalty (0.1x multiplier if mostly wrong domain)
```

- **Base frequency**: proportion of hits from the most-represented domain
- **Context boost**: 5x multiplier when completion is primarily from active context domain (>50% hits)
- **Cross-domain penalty**: Heavy 0.1x multiplier if >80% of hits are from wrong domains

## Class: `ContextTracker`

### Structure

```python
class ContextTracker:
    window: list[tuple[str, float]]          # (domain, timestamp) — sliding window
    window_size: int = 20                    # Max entries in window
    decay_rate: float = 0.9                  # Exponential decay per time unit
    domain_frequencies: dict[str, float]     # Decayed frequency weights per domain
```

### Methods

| Method | Description |
|--------|-------------|
| `observe(domain)` | Add domain to window; trim if exceeds size; update frequencies with exponential decay |
| `active_domain()` | Return most active domain (highest frequency weight) or None |
| `domain_weights()` | Return all domain weights normalized to sum=1 |

### Frequency Calculation

```python
weight = position_weight * time_decay
position_weight = (i + 1) / len(window)        # Newer items get higher weight
time_weight = decay_rate ** age                 # Exponential decay by age
```

## Class: `PredictiveContextSearch`

### Initialization

```python
class PredictiveContextSearch:
    def __init__(self, omega_cube_engine=None):
        self.trie = HierarchicalTrie()              # Auto-created
        self.context = ContextTracker()             # Auto-created
        self.cube = omega_cube_engine               # Optional engine reference
        self._access_log: dict[str, float]          # node_id → last_access_time
```

### Core Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `index_node()` | `(text, domain, node_id=None)` | Insert into trie; also index individual significant words (>2 chars) for partial matching |
| `index_from_cube()` | `() → int` | Index all nodes from connected Omega-Cube engine, returning count indexed |
| `search()` | `(prefix, max_results=10) → list[dict]` | Context-aware search; boosts results with gray-scale if cube connected |
| `predict()` | `(prefix, max_results=5) → list[str]` | Simple auto-complete: return predicted full texts (truncated to 120 chars) |
| `feed_context()` | `(text)` | Feed conversation text to update active context domain |

### Domain Detection (`_detect_domain`)

Two-priority system:
1. **Exact match**: Check if any of 14 known domains appear in text (COMFYUI, EVONY, HERMES, HBIT, ML, MEDICAL, FINANCE, LEGAL, PHYSICS, MUSIC, PYTHON, OMEGA, SECURITY, PHARM)
2. **Keyword matching**: Domain-specific keyword lists for fallback detection

Domain keywords per domain:
- **COMFYUI**: comfyui, sdxl, checkpoint, vae, lora, workflow, ipadapter, upscale, inpaint
- **EVONY**: evony, marcian, hermes, akechi, tamar, f2p, pvp, ranged, mounted, siege, rally
- **HERMES**: hermes agent, mcp server, cron job, skill pack, state.db, hermes config
- **HBIT**: h-bit, hbit, grayscale, steganograph, verify, bit chain
- Plus MEDICAL, FINANCE, LEGAL, ML, PHYSICS, MUSIC, PHARM keyword sets

### Gray-Scale Boosting (`_gray_scale_boost`)

```python
for node in cube.nodes.values():
    if text_lower in node.content.lower():
        composite = gray_validator.composite_score(node.gray_scale)
        best_gs = max(best_gs, composite / 100)
return best_gs * 0.5   # Up to +0.5 score boost
```

### Stats Output

```python
{
    "total_nodes": trie._node_count,
    "root_children": len(trie.root.children),
    "root_total_hits": trie.root.total_hits,
    "active_domain": context.active_domain(),
    "context_window_size": len(context.window),
}
```

## Demo: Windows Search vs Predictive Context Search

The `demo_predictive_vs_flat()` function demonstrates the key difference:

| Prefix | Active Context | Result Behavior |
|--------|---------------|-----------------|
| `"S"` | None (ambigous) | All domains mixed |
| `"SD"` | COMFYUI | SDXL/SD1.5 results only |
| `"Ma"` | COMFYUI | NO Marcian, NO Matrix — correct! |
| `"Ma"` | EVONY | Marcian first, not Matrix — context-aware! |

This is the "Windows search cannot do" effect: hierarchical trie + domain context filtering eliminates cross-domain noise.

## Usage Example

```python
from omega_cube.predictive_search import PredictiveContextSearch

pcs = PredictiveContextSearch()

# Index nodes with domains
pcs.index_node("SDXL base checkpoint at J:/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors", "COMFYUI")
pcs.index_node("Marcian #1 Ranged PvP: +45% attack, +30% defense vs mounted", "EVONY")

# Feed conversation context
pcs.feed_context("I need to configure SDXL in ComfyUI")

# Search — gets COMFYUI-prioritized results even for ambiguous prefixes
results = pcs.search("Ma", max_results=5)  # Returns Marcian entries, not Matrix
```
