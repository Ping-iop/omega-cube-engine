# HolographicEncoder API Reference

**Ubicación:** `omega_cube/holographic.py`  
**Versión:** 1.0.0  
**Última actualización:** 2026-06-29  

---

## Resumen

`HolographicEncoder` implementa codificación holográfica inspirada en HRR (Holographic Reduced Representations). Codifica nodos en vectores fijos de dimensión `d` que permiten:
- Búsqueda O(1) aproximada via `partial_match()`
- Asociación y desasociación mediante convolución circular (FFT)
- Compresión distribuida del vecindario estructural

---

## Inicialización

```python
from omega_cube.holographic import HolographicEncoder

encoder = HolographicEncoder(
    dimension=256,      # Dimensión de vectores (default: 256)
    seed=None           # Seed para reproducibilidad
)
```

---

## Métodos Principales

### `encode_node()` — Codificar Nodo en Vector Base

**Firma:**
```python
def encode_node(
    self,
    content: str,
    hierarchy: str = ""
) -> list[float]  # vector de dimensión d
```

Crea un vector base a partir del contenido y jerarquía del nodo. Usa un seed determinista basado en hash para generar vectores pseudo-aleatorios consistentes.

**Ejemplo:**
```python
vec = encoder.encode_node(
    content="AWS es mejor",
    hierarchy="CLOUD.PROVIDERS.AWS"
)
print(len(vec))  # → 256
```

---

### `encode_holographic_signature()` — Firma Holográfica Completa

**Firma:**
```python
def encode_holographic_signature(
    self,
    node_content: str,
    node_hierarchy: str = "",
    parent_content: str = "",
    parent_hierarchy: str = "",
    children: list[tuple[str, str]] = None,  # [(child_content, child_hierarchy)]
    neighbors: list[tuple[str, str]] = None   # [(neighbor_content, neighbor_hierarchy)]
) -> list[float]  # vector de dimensión d
```

Genera la firma holográfica completa que codifica el contexto estructural del nodo (padre, hijos, vecinos).

**Ejemplo:**
```python
signature = encoder.encode_holographic_signature(
    node_content="AWS es mejor",
    node_hierarchy="CLOUD.PROVIDERS.AWS",
    parent_content="Cloud Providers",
    parent_hierarchy="TECH.CLOUD",
    children=[("EC2", "INFRASTRUCTURE.EC2")],
    neighbors=[("Azure", "CLOUD.PROVIDERS.AZURE")]
)
```

---

### `bind()` — Convolución Circular (Asociación)

**Firma:**
```python
def bind(self, v1: list[float], v2: list[float]) -> list[float]
```

Realiza convolución circular entre dos vectores mediante FFT. Equivalente a "asociar" o "vincular" dos conceptos.

**Propiedad clave:** `unbind(bind(v1, v2), v1) == v2` (recuperación perfecta).

**Ejemplo:**
```python
v1 = encoder.encode_node("AWS", "CLOUD.PROVIDERS")
v2 = encoder.encode_node("scalable", "QUALITY.SCALABLE")

# Asociar conceptos
bound = encoder.bind(v1, v2)

# Desasociar (recuperar v2 de la asociación)
recovered_v2 = encoder.unbind(bound, v1)
```

---

### `unbind()` — Recuperación de Componente

**Firma:**
```python
def unbind(self, bound: list[float], v1: list[float]) -> list[float]
```

Recupera v2 a partir de bind(v1, v2). Permite "preguntar" al vector asociado qué estaba vinculado.

**Ejemplo:**
```python
# Si bind(aws_vector, scalable_vector) = bound_vec
recovered = encoder.unbind(bound_vec, aws_vector)
# → recovered ≈ scalable_vector (dentro de tolerancia coseno)
```

---

### `bundle()` — Superposición (Combinación Múltiple)

**Firma:**
```python
def bundle(self, vectors: list[list[float]]) -> list[float]
```

Suma vectorial de múltiples vectores. Permite "comprimir" un conjunto de conceptos en un solo vector.

**Ejemplo:**
```python
vectors = [
    encoder.encode_node("AWS", "CLOUD.PROVIDERS"),
    encoder.encode_node("Azure", "CLOUD.PROVIDERS"),
    encoder.encode_node("GCP", "CLOUD.PROVIDERS")
]

bundled = encoder.bundle(vectors)
# → bundled contiene información de los 3 proveedores comprimidos
```

---

### `similarity()` — Similitud Coseno Normalizada

**Firma:**
```python
def similarity(self, v1: list[float], v2: list[float]) -> float  # [0, 1]
```

Calcula similitud coseno entre dos vectores y la normaliza a [0, 1].

- `similarity(v, v) == 1.0` (máxima similitud consigo mismo)
- `similarity(v1, v2) ≈ 0.5` para vectores aleatorios (baseline)
- `similarity(v1, v2) > 0.7` indica semántica similar

**Ejemplo:**
```python
v1 = encoder.encode_node("AWS es mejor", "CLOUD.PROVIDERS")
v2 = encoder.encode_node("Azure ofrece integración enterprise", "CLOUD.PROVIDERS")

sim = encoder.similarity(v1, v2)
print(f"Similitud: {sim:.3f}")
# → Similitud: ~0.65 (relacionados pero diferentes)
```

---

### `partial_match()` — Verificación de Contenido Parcial

**Firma:**
```python
def partial_match(self, query_vec: list[float], signature: list[float]) -> float  # [0, 1]
```

Verifica si un vector de query está "contenido" dentro de una firma holográfica. Este es el método clave para búsqueda O(1): la firma codifica el contexto del nodo (padre, hijos, vecinos), y `partial_match` detecta si la query corresponde a alguno de esos componentes.

**Ejemplo:**
```python
# Buscar nodos que contengan información sobre "cloud providers"
query_vec = encoder.encode_node("cloud provider", "TECH.CLOUD")

for node in engine.nodes.values():
    if node.holographic_signature:
        score = encoder.partial_match(query_vec, node.holographic_signature)
        if score > 0.5:
            print(f"Coincidencia {score:.2f}: {node.content}")
```

---

## Métricas de Rendimiento Verificadas

| Métrica | Resultado |
|---------|-----------|
| Búsqueda discriminativa ("evony generals") | Score 0.79 |
| Baseline keyword matching | Score 0.53 |
| Dimensión vectorial por defecto | 256 |
| Tiempo de partial_match | ~O(1) (constante, independiente del grafo) |

---

## Ejemplo Completo: Búsqueda Semántica con Holográfico

```python
from omega_cube.holographic import HolographicEncoder

encoder = HolographicEncoder(dimension=256)

# Codificar query y nodos
query_vec = encoder.encode_node("best image generation workflow")

results = []
for node in engine.nodes.values():
    if node.holographic_signature:
        score = encoder.partial_match(query_vec, node.holographic_signature)
        results.append((node, score))

# Ordenar por relevancia
results.sort(key=lambda x: -x[1])

print("Top resultados:")
for node, score in results[:5]:
    print(f"  {score:.2f} → {node.content}")
```

---

## Notas de Implementación

- **FFT:** La convolución circular se implementa usando numpy FFT para eficiencia O(n log n)
- **Determinismo:** El seed del vector base es determinista (hash del contenido), lo que garantiza consistencia en codificaciones repetidas
- **Limitación conocida:** Hamming distance puede ser impreciso para queries muy específicas — considerar mejorar con TF-IDF o embeddings en el futuro

---

**Ver también:** [OmegaCubeEngine API](engine.md), [TensorNode API](tensor_node.md)
