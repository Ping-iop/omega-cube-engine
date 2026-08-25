# OmegaCubeEngine API Reference

**Ubicación:** `omega_cube/engine.py`  
**Versión:** 1.0.0  
**Última actualización:** 2026-06-29  

---

## Resumen

`OmegaCubeEngine` es el punto de entrada principal del motor. Orquestra todos los sub-motores (TensorIndex, HolographicEncoder, DiffusionSampler, etc.) y maneja la persistencia en `memory/omega_cube_memory.json`.

---

## Inicialización

```python
from omega_cube.engine import OmegaCubeEngine

engine = OmegaCubeEngine(
    memory_dir=None,        # Path a directorio de memoria (default: auto-detect)
    holographic_dim=256,    # Dimensión de vectores holográficos
    tensor_grid_size=10     # Tamaño de grilla para TensorIndex
)
```

### Carga automática

El motor carga automáticamente nodos desde `memory/omega_cube_memory.json` al inicializarse:

```python
engine = OmegaCubeEngine()
print(f"Nodos cargados: {len(engine.nodes)}")  # → Nodos cargados: 18
```

---

## Métodos Públicos

### `add_node()` — Agregar Nodo al Grafo

**Firma:**
```python
def add_node(
    self,
    content: str,
    hierarchies: list[str],
    tensor_position: list[float] = None,
    node_type: str = "CONCEPT",
    confidence: float = 0.9,
    tags: list[str] = None,
) -> TensorNode
```

**Parámetros:**

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `content` | `str` | Sí | Texto del nodo (ej: "AWS es mejor para escalabilidad") |
| `hierarchies` | `list[str]` | Sí | Múltiples paths jerárquicos (ej: `["COMFYUI.WORKFLOWS", "QUALITY.HIGH"]`) |
| `tensor_position` | `list[float]` | No | Coordenadas normalizadas [0,1]^N en espacio tensorial |
| `node_type` | `str` | No | Tipo: AXIOM / CONCEPT / INSTANCE / SESSION (default: "CONCEPT") |
| `confidence` | `float` | No | Confianza base 0.0-1.0 (default: 0.9) |
| `tags` | `list[str]` | No | Tags semánticos para búsqueda adicional |

**Retorna:** `TensorNode` — El nodo creado con ID asignado.

**Ejemplo:**
```python
node = engine.add_node(
    content="SDXL is better for quality",
    hierarchies=[
        "COMFYUI.WORKFLOWS.GENERATION",
        "QUALITY.IMAGE_RESOLUTION.HIGH"
    ],
    tensor_position=[0.7, 0.8],
    node_type="CONCEPT",
    confidence=0.93
)

print(f"Nodo ID: {node.node_id}")
# → Nodo ID: a1b2c3d4e5f6g7h8
```

---

### `query()` — Búsqueda Semántica

**Firma:**
```python
def query(
    self,
    query: str,
    mode: str = "combined",
    top_k: int = 10
) -> list[dict]
```

**Parámetros:**

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `query` | `str` | Sí | Query de búsqueda (ej: "best image generation workflow") |
| `mode` | `str` | No | Modo de búsqueda: `holographic` / `diffusion` / `combined` (default: "combined") |
| `top_k` | `int` | No | Número de resultados a retornar (default: 10) |

**Modos disponibles:**

| Modo | Descripción | Complejidad |
|------|-------------|-------------|
| `"holographic"` | Búsqueda O(1) aproximada via HolographicEncoder | O(1) |
| `"diffusion"` | Muestreo no-autoregresivo con annealing | O(log n) |
| `"combined"` | Hibrido (default) — combina ambos enfoques | O(n log n) |

**Retorna:** `list[dict]` — Lista de resultados ordenados por relevancia.

**Estructura del resultado:**
```python
[
    {
        "node_id": "a1b2c3d4e5f6g7h8",
        "content": "SDXL is better for quality",
        "hierarchies": ["COMFYUI.WORKFLOWS.GENERATION", "QUALITY.IMAGE_RESOLUTION.HIGH"],
        "score": 0.85,
        "node_type": "CONCEPT",
        "confidence": 0.93
    },
    # ... más resultados
]
```

**Ejemplo:**
```python
results = engine.query("best image generation workflow", mode="diffusion")

for r in results[:5]:
    print(f"{r['score']:.2f} → {r['content']} [{', '.join(r['hierarchies'])}]")
# 0.85 → SDXL is better for quality [COMFYUI.WORKFLOWS.GENERATION]
# 0.72 → ComfyUI workflow para alta resolución [QUALITY.IMAGE_RESOLUTION.HIGH]
```

---

### `find_patterns()` — Detección de Patrones Emergentes

**Firma:**
```python
def find_patterns(
    self,
    query: str,
    threshold: float = 0.7
) -> list[dict]
```

**Parámetros:**

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `query` | `str` | Sí | Query para buscar patrones multi-tópico |
| `threshold` | `float` | No | Umbral de alineación (default: 0.7) |

**Retorna:** `list[dict]` — Patrones emergentes donde múltiples cubos se alinean.

**Estructura del resultado:**
```python
[
    {
        "anchor_cube": "COMFYUI",
        "cube_topic": "ComfyUI Workflows",
        "aligned_cubes": [
            {"cube_id": "QUALITY", "alignment": 0.85},
            {"cube_id": "DECISIONS", "alignment": 0.72}
        ],
        "pattern_strength": 0.78,
        "exposed_content": "SDXL recommended for high-res generation"
    }
]
```

**Ejemplo:**
```python
patterns = engine.find_patterns("cloud provider comparison")

for p in patterns:
    print(f"Patrón en {p['cube_topic']} (fuerza: {p['pattern_strength']:.2f})")
    for aligned in p['aligned_cubes']:
        print(f"  → Alineado con: {aligned['cube_id']} ({aligned['alignment']:.2f})")
```

---

### `load()` — Cargar Estado Persistido

**Firma:**
```python
def load(self) -> bool
```

Carga estado desde `memory_dir/omega_cube_memory.json`. Retorna `True` si éxito, `False` si el archivo no existe.

```python
success = engine.load()
if success:
    print(f"Estado cargado: {len(engine.nodes)} nodos")
```

---

### `save()` — Persistir Estado

**Firma:**
```python
def save(self) -> bool
```

Persiste estado actual a `memory_dir/omega_cube_memory.json`. Retorna `True` si éxito.

```python
engine.add_node("Nuevo nodo", ["TOPIC.NEW"])
engine.save()  # Persiste cambios
```

---

## Atributos Públicos

| Atributo | Tipo | Descripción |
|----------|------|-------------|
| `nodes` | `dict[str, TensorNode]` | Diccionario de nodos por ID |
| `tensor_index` | `TensorIndex` | Índice espacial para búsqueda rápida |
| `holographic_encoder` | `HolographicEncoder` | Codificador holográfico (256-dim) |
| `diffusion_sampler` | `DiffusionGraphSampler` | Muestreador difusivo |
| `annealer` | `QuantumInspiredAnnealer` | Annealer con túnel cuántico |

---

## Ejemplo Completo

```python
from omega_cube.engine import OmegaCubeEngine

# 1. Inicializar motor
engine = OmegaCubeEngine()

# 2. Agregar nodos
engine.add_node(
    content="AWS es mejor para escalabilidad",
    hierarchies=["CLOUD.PROVIDERS.AWS", "SCALABILITY.HIGH"],
    tensor_position=[0.8, 0.9]
)

engine.add_node(
    content="Azure ofrece mejor integración con enterprise",
    hierarchies=["CLOUD.PROVIDERS.AZURE", "ENTERPRISE.INTEGRATION.HIGH"]
)

# 3. Búsqueda semántica
results = engine.query("cloud provider comparison")
print(f"Resultados: {len(results)}")

# 4. Detectar patrones emergentes
patterns = engine.find_patterns("enterprise cloud strategy")
for p in patterns:
    print(f"Patrón detectado en {p['cube_topic']}")

# 5. Persistir cambios
engine.save()
```

---

## Notas de Implementación

- **Búsqueda holográfica:** Usa `HolographicEncoder.partial_match()` para O(1) aproximado
- **Difusión:** Aplica `DiffusionGraphSampler.sample()` con denoising iterativo
- **Combinación:** Weighted average de scores holográficos + difusivos
- **Persistencia:** Serializa `nodes` dict a JSON en `memory/omega_cube_memory.json`

---

**Ver también:** [TensorNode API](tensor_node.md), [HolographicEncoder API](holographic.md)
