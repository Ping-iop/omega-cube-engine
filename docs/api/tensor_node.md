# TensorNode API Reference

**Ubicación:** `omega_cube/tensor_node.py`  
**Versión:** 1.0.0  
**Última actualización:** 2026-06-29  

---

## Resumen

`TensorNode` es la unidad fundamental de conocimiento en Omega-Cube. Cada nodo existe simultáneamente en N dimensiones jerárquicas, con coordenadas normalizadas en espacio tensorial [0,1]^N que permiten búsqueda espacial rápida.

`TensorIndex` es el índice spatial que soporta búsquedas por proximidad en el espacio tensorial.

---

## TensorNode

### Estructura de Datos

```python
from omega_cube.tensor_node import TensorNode, TensorIndex

node = TensorNode(
    content="AWS es mejor para escalabilidad",
    hierarchies=["CLOUD.PROVIDERS.AWS", "SCALABILITY.HIGH"],
    tensor_position=[0.8, 0.9],      # Coordenadas [0,1]^2
    node_type="CONCEPT",              # AXIOM / CONCEPT / INSTANCE / SESSION
    confidence=0.93,                  # Confianza base 0-1
    tags=["cloud", "aws", "scalable"],
    associations=[],                  # IDs de nodos asociados
    holographic_signature=None,       # Firma 256-dim (generada por HolographicEncoder)
    gray_scale={}                     # Evaluación multi-bit {dim: score_0-100}
)
```

### Atributos

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `content` | `str` | Texto del nodo (ej: "AWS es mejor para escalabilidad") |
| `hierarchies` | `list[str]` | Múltiples paths jerárquicos simultáneos |
| `tensor_position` | `list[float]` | Coordenadas normalizadas [0,1]^N en espacio tensorial |
| `node_type` | `str` | Tipo: AXIOM / CONCEPT / INSTANCE / SESSION (default: "CONCEPT") |
| `confidence` | `float` | Confianza base 0.0-1.0 (default: 0.9) |
| `tags` | `list[str]` | Tags semánticos para búsqueda adicional |
| `associations` | `list[str]` | IDs de nodos asociados |
| `holographic_signature` | `list[float]` | Firma holográfica (256-dim) opcional |
| `gray_scale` | `dict` | Evaluación multi-bit {dim: score_0-100} opcional |

### Propiedades

```python
# Jerarquía primaria (primera jerarquía definida)
print(node.primary_hierarchy)
# → "CLOUD.PROVIDERS.AWS"

# Número de dimensiones del espacio tensorial
print(node.dimension_count)
# → 2 (si tiene 2 hierarchies) o len(tensor_position)
```

### Métodos

#### `distance_to()` — Distancia Euclidiana Normalizada

```python
dist = node.distance_to(other_node)
# → float [0,1] — distancia normalizada en espacio tensorial
```

Calcula la distancia euclidiana entre dos nodos en su espacio tensorial compartido. Útil para encontrar vecinos cercanos.

#### `matches_dimension()` — Verificación de Dimensión

```python
matches = node.matches_dimension(
    dimension="CLOUD",
    tolerance=0.1
)
# → bool — True si la jerarquía del nodo está dentro de la dimensión especificada
```

Verifica si el nodo pertenece a una dimensión jerárquica específica con tolerancia opcional.

#### `gray_scale_score()` — Score Gray-Scale por Dimensión

```python
score = node.gray_scale_score("factuality")
# → float [0,100]
```

Retorna la puntuación gray-scale para una dimensión específica (factuality, relevance, recency, etc.). Usa el valor de `gray_scale` si está disponible.

#### `to_dict()` — Serialización

```python
d = node.to_dict()
# → dict con todos los campos del nodo
```

Serializa el nodo a un diccionario para persistencia JSON.

#### `from_dict()` — Deserialización (Estático)

```python
node = TensorNode.from_dict(d)
# → TensorNode reconstruido desde dict
```

Reconstruye un TensorNode desde su representación en diccionario.

---

## TensorIndex

### Resumen

Índice espacial para búsqueda rápida por proximidad en el espacio tensorial. Permite encontrar nodos vecinos dentro de un radio especificado.

### Métodos

#### `insert()` — Agregar Nodo al Índice

```python
index = TensorIndex()
index.insert(node)
# → Indexa el nodo para búsquedas posteriores
```

#### `query()` — Búsqueda por Proximidad

```python
neighbors = index.query(
    tensor_position=[0.5, 0.6],
    radius=0.3
)
# → list[TensorNode] — nodos dentro del radio especificado
```

Busca nodos cuyo `tensor_position` esté dentro de `radius` de la posición dada.

#### `node_map` — Acceso Directo a Todos los Nodos

```python
all_nodes = index.node_map  # dict[str, TensorNode]
```

Diccionario completo de todos los nodos indexados por ID.

---

## Ejemplo Completo

```python
from omega_cube.tensor_node import TensorNode, TensorIndex

# Crear índices y nodos
index = TensorIndex()

node1 = TensorNode(
    content="AWS es mejor para escalabilidad",
    hierarchies=["CLOUD.PROVIDERS.AWS", "SCALABILITY.HIGH"],
    tensor_position=[0.8, 0.9],
    confidence=0.93
)

node2 = TensorNode(
    content="Azure ofrece mejor integración enterprise",
    hierarchies=["CLOUD.PROVIDERS.AZURE", "ENTERPRISE.INTEGRATION.HIGH"],
    tensor_position=[0.7, 0.6],
    confidence=0.88
)

node3 = TensorNode(
    content="GCP tiene mejor pricing para startups",
    hierarchies=["CLOUD.PROVIDERS.GCP", "PRICING.STARTUP_FAVORABLE"],
    tensor_position=[0.9, 0.4],
    confidence=0.85
)

# Insertar en índice
index.insert(node1)
index.insert(node2)
index.insert(node3)

# Buscar vecinos de node1 dentro de radio 0.3
neighbors = index.query(
    tensor_position=node1.tensor_position,
    radius=0.3
)
print(f"Vecinos encontrados: {len(neighbors)}")
# → Vecinos encontrados: 2 (node2 y posiblemente node3)

# Distancia entre nodos
dist = node1.distance_to(node2)
print(f"Distancia AWS-Azure: {dist:.3f}")
# → Distancia AWS-Azure: 0.316

# Verificar dimensión
matches_cloud = node1.matches_dimension("CLOUD", tolerance=0.1)
print(f"¿Es cloud? {matches_cloud}")
# → ¿Es cloud? True
```

---

## Notas de Implementación

- **Normalización:** `tensor_position` se normaliza a [0,1]^N automáticamente
- **Distancia euclidiana:** Calculada como √(Σ(x_i - y_i)²) / √N para normalizar por dimensión
- **Coherencia dimensional:** Si dos nodos tienen diferente número de dimensiones, la distancia se calcula sobre las dimensiones compartidas

---

**Ver también:** [OmegaCubeEngine API](engine.md), [HolographicEncoder API](holographic.md)
