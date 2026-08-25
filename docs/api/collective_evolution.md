# CollectiveHierarchyEngine — Evolution from User Behavior

**Module:** `omega_cube.collective_evolution` (615 lines, 27,261 bytes)  
**Purpose:** Evolves knowledge hierarchies from collective user behavior across multiple sessions. Discovers emergent subcategories, strengthens frequent paths, and adapts tensor dimensions based on real usage patterns.

## Architecture

Two cooperating components:
1. **SessionSignals** — Extracts evolution signals (topic transitions, co-occurrences, decision chains) from individual Hermes session transcripts
2. **CollectiveHierarchyEngine** — Aggregates signals across sessions to evolve a hierarchy driven by behavior, not expert design

## Class: `SessionSignalExtractor`

### Inicialización

```python
from omega_cube.collective_evolution import SessionSignalExtractor

extractor = SessionSignalExtractor(state_db_path=None)  # Default: Hermes state.db
```

El path default apunta a `C:\Users\GPAMD\AppData\Local\hermes\state.db`.

### Método Principal: `extract_from_session()`

**Firma:**
```python
def extract_from_session(self, session_id: str) -> Optional[SessionSignals]
```

Lee el SQLite de Hermes para extraer señales evolutivas de una sesión individual.

### Método Batch: `extract_from_all_sessions()`

**Firma:**
```python
def extract_from_all_sessions(self, limit: int = 50) -> list[SessionSignals]
```

Extrae señales de todas las sesiones disponibles (limitadas).

### Métodos Internos

| Método | Descripción |
|--------|-------------|
| `_detect_domain(text)` | Detecta dominio principal en texto (devuelve el top-1) |
| `_detect_all_domains(text)` | Devuelve lista ordenada por score de todos los dominios que aparecen |

### Domain Detection Keywords

```python
domain_keywords = {
    "COMFYUI": ["comfyui", "sdxl", "checkpoint", ...],
    "EVONY":   ["evony", "marcian", "f2p", ...],
    "HERMES":  ["hermes", "mcp", "cron", ...],
    "HBIT":    ["h-bit", "grayscale", "steganograph", ...],
    "OMEGA":   ["omega", "graph", "tensor", ...],
    "ML":      ["diffusion", "transformer", "fine-tuning", ...],
    "PYTHON":  ["python", "script", "import", ...],
}
```

Scoreado por conteo de keywords; retorna lista ordenada por score descendente.

---

## Class: `CollectiveHierarchyEngine`

### Initialization

```python
class CollectiveHierarchyEngine:
    hierarchy_weights: dict[str, float]            # path → accumulated weight
    transition_matrix: dict[tuple[str,str], float]  # (from, to) → total weight
    co_occurrence_matrix: dict[tuple[str,str], float]  # (a, b) → count
    emergent_categories: dict[str, dict[str,float]]  # category → {domain: weight}
    total_signals_processed: int
    total_sessions: int
    evolution_log: list[dict]                       # Per-session evolution records
```

### Core Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `ingest_session_signals()` | `(signals: SessionSignals)` | Process one session's signals into the collective hierarchy |
| `ingest_multiple_sessions()` | `(signals_list: list[SessionSignals])` | Batch process multiple sessions |
| `get_evolved_hierarchy()` | `(min_weight=0.1) → dict` | Return evolved structure ready for Omega-Cube integration |
| `apply_to_omega_cube()` | `(cube_engine) → int` | Apply evolved hierarchy to engine: adjust tensor positions, add emergent nodes |
| `stats()` | `() → dict` | Returns total_sessions, total_signals, unique_transitions, etc. |

### Signal Processing (per `ingest_session_signals`)

1. **Topic Transitions** — Strengthen `TRANSITION.{from}.{to}` hierarchy path by weight; update transition matrix
2. **Co-occurrences** — Update co_occurrence_matrix with normalized key ordering (alphabetical)
3. **Active Domains** — Normalize per-turn count and strengthen `DOMAIN.{domain}.ACTIVITY` path
4. **Decision Chains** — Strengthen `DECISION.{d1}.{d2}` for consecutive decisions in same session
5. **Emergent Categories** — If ≥2 domains active, create emergent category from co-occurring pairs

### Emergent Category Detection

When two domains co-occur heavily within a single session:
```python
# Both COMFYUI(8 turns) and PYTHON(4 turns) active → "COMFYUI+PYTHON" emerges
emergent_categories["COMFYUI+PYTHON"]["COMFYUI"] += 8
emergent_categories["COMFYUI+PYTHON"]["PYTHON"] += 4
```

### Applying to Omega-Cube (`apply_to_omega_cube`)

Two effects:
1. **Tensor Position Adjustment** — Co-occurring domains pulled closer in tensor space by normalized transition weight (max 0.1 shift)
2. **Emergent Node Creation** — Categories with total signal ≥3 become new CONCEPT nodes with hierarchy `EMERGENT.{category}` and `COLLECTIVE.INTELLIGENCE`

```python
cube_engine.add_node(
    content=f"Emergent category: {category} from user behavior patterns",
    hierarchies=[f"EMERGENT.{category}", f"COLLECTIVE.INTELLIGENCE"],
    tensor_position=[0.5, 0.5],
    node_type="CONCEPT",
    confidence=0.6 + min(0.3, sum(domains.values()) / 100),
    tags=["emergent", "collective", "evolved"],
)
```

### Evolved Hierarchy Output (`get_evolved_hierarchy`)

Returns structured dict with:
- **transitions**: Top 20 domain→domain transitions by weight
- **co_occurrences**: Top 20 co-occurring domain pairs
- **emergent_categories**: Top 10 emergent categories with top 5 domains each
- **domain_activity**: Activity score per domain (DOMAIN.X.ACTIVITY)
- **top_paths**: Top 20 hierarchy paths by accumulated weight

### Persistence

```python
engine.save("collective_hierarchy.json")   # JSON persistence
engine.load("collective_hierarchy.json")    # Restore from disk
```

## Demo: Static vs Evolved Hierarchy

The demo (`demo_collective_evolution()`) simulates 5 users with different behavior patterns and compares:

**Static hierarchy** (expert-designed):
- COMFYUI → MODELS, WORKFLOWS, NODES
- EVONY → GENERALS, F2P, PvP

**Evolved hierarchy** (user-behavior-driven):
- DISCOVERED: COMFYUI↔PYTHON (6 co-occurrences), COMFYUI↔HBIT (4), EVONY↔COMFYUI (6)
- EMERGENT CATEGORIES: "COMFYUI+PYTHON", "EVONY+COMFYUI", "OMEGA+HERMES"
- TOP TRANSITIONS: OMEGA→PYTHON, HERMES→OMEGA, COMFYUI→PYTHON

Key insight: static hierarchies MISS connections that users actually make. The evolved hierarchy captures real navigation patterns.

## Usage Example

```python
from omega_cube.collective_evolution import SessionSignals, CollectiveHierarchyEngine

# Extract signals from existing sessions
extractor = SessionSignals()
signals = extractor.extract_from_all_sessions(limit=50)

# Feed into collective engine
engine = CollectiveHierarchyEngine()
engine.ingest_multiple_sessions(signals)

# Get evolved hierarchy
evolved = engine.get_evolved_hierarchy()
print(f"Discovered {len(evolved['emergent_categories'])} emergent categories")

# Apply to Omega-Cube
changes = engine.apply_to_omega_cube(cube_engine)
print(f"Applied {changes} changes to graph topology")
```
