# DiffusionGraphSampler API Reference

**Ubicación:** `omega_cube/diffusion_sampler.py`  
**Versión:** 1.0.0  
**Última actualización:** 2026-06-29  

---

## Resumen

`DiffusionGraphSampler` implementa recuperación no-autoregresiva inspirada en Google DeepMind's DiffusionGemma (2026). En vez de predecir el siguiente nodo secuencialmente (O(n)), muestrea todos los candidatos simultáneamente y refinan mediante denoising iterativo guiado por estructura jerárquica.

---

## Inicialización

```python
from omega_cube.diffusion_sampler import DiffusionGraphSampler

sampler = DiffusionGraphSampler(
    num_steps=20,           # Iteraciones de denoising (default: 20)
    guidance_scale=3.0,     # Peso del guidance jerárquico (default: 3.0)
    seed=None               # Seed para reproducibilidad
)
```

### Parámetros configurables

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `num_steps` | `int` | 20 | Iteraciones de denoising. Más pasos = más preciso pero más lento |
| `guidance_scale` | `float` | 3.0 | Peso del guidance jerárquico. Más alto = resultados más enfocados en estructura |
| `seed` | `int \| None` | None | Seed para reproducibilidad (None = no determinista) |

---

## Métodos Principales

### `sample()` — Muestreo Difusivo con Denoising Iterativo

**Firma:**
```python
def sample(
    self,
    query: str,
    index: TensorIndex,
    holographic_encoder: HolographicEncoder,
    top_k: int = 10,
    temperature: float = 0.1
) -> list[tuple[TensorNode, float]]
```

**Fase 1 — Generar vector de query (señal):**
El encoder codifica la query como un vector base en el espacio holográfico.

**Fase 2 — Inicializar candidatos con ruido:**
Todos los nodos del índice se inicializan con scores aleatorios (punto de partida de la difusión).

**Fase 3 — Denoising iterativo:**
Cada paso aplica:
1. **Señal base:** Similaridad holográfica entre query y cada nodo
2. **Ruido decreciente:** Schedule coseno que reduce ruido con el tiempo
3. **Guidance jerárquico:** Boost para nodos cuyos vecinos tienen scores altos (efecto magnético)

**Fase 4 — Organización por proximidad jerárquica + diversidad:**
Los resultados se re-ordenan para asegurar variedad (penalización de nodes demasiado similares entre sí).

**Parámetros:**

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `query` | `str` | Sí | Query de búsqueda |
| `index` | `TensorIndex` | Sí | Índice con todos los nodos del grafo |
| `holographic_encoder` | `HolographicEncoder` | Sí | Encoder para calcular similitudes |
| `top_k` | `int` | No | Número de resultados a retornar (default: 10) |
| `temperature` | `float` | No | Nivel de ruido. Bajo = determinista, alto = exploratorio (default: 0.1) |

**Retorna:** `list[tuple[TensorNode, float]]` — Nodos ordenados por relevancia con scores [0,1].

**Ejemplo:**
```python
from omega_cube.tensor_node import TensorIndex
from omega_cube.holographic import HolographicEncoder

index = TensorIndex()
for node in engine.nodes.values():
    index.insert(node)

encoder = HolographicEncoder(dimension=256)

results = sampler.sample(
    query="best image generation workflow",
    index=index,
    holographic_encoder=encoder,
    top_k=5,
    temperature=0.1  # Bajo ruido = resultados más consistentes
)

for node, score in results:
    print(f"{score:.3f} → {node.content}")
```

---

### `sample_multi_topic()` — Muestreo Multi-Tópico Independiente

**Firma:**
```python
def sample_multi_topic(
    self,
    query: str,
    index: TensorIndex,
    holographic_encoder: HolographicEncoder,
    topic_dimensions: list[str],
    top_k_per_topic: int = 3
) -> dict[str, list[tuple[TensorNode, float]]]
```

Muestrea independientemente por dimensión temática y luego combina resultados organizados por clusters naturales.

**Parámetros:**

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `query` | `str` | Sí | Query de búsqueda |
| `index` | `TensorIndex` | Sí | Índice con todos los nodos |
| `holographic_encoder` | `HolographicEncoder` | Sí | Encoder para similitudes |
| `topic_dimensions` | `list[str]` | Sí | Lista de prefijos jerárquicos a muestrear (ej: ["EVONY", "COMFYUI"]) |
| `top_k_per_topic` | `int` | No | Resultados por dimensión temática (default: 3) |

**Retorna:** `dict[str, list[tuple[TensorNode, float]]]` — Diccionario mapeando topic → resultados.

**Ejemplo:**
```python
results = sampler.sample_multi_topic(
    query="cloud provider comparison",
    index=index,
    holographic_encoder=encoder,
    topic_dimensions=["CLOUD.PROVIDERS", "QUALITY.SCALABILITY"],
    top_k_per_topic=3
)

for topic, topic_results in results.items():
    print(f"\n{topic}:")
    for node, score in topic_results:
        print(f"  {score:.2f} → {node.content}")
```

---

## Schedule de Ruido Coseno

El sampler usa un schedule coseno para reducir gradualmente el ruido:

```python
def _cosine_noise_schedule(self, step: int, total_steps: int) -> float:
    progress = step / total_steps
    return 0.5 * (1 + math.cos(math.pi * progress))
```

- Paso 0: ruido máximo (~1.0) — exploración amplia
- Paso final: ruido mínimo (~0.0) — convergencia precisa

---

## Guidance Jerárquico

El guidance jerárquico es el "efecto magnético" que atrae nodos hacia clusters de alta relevancia:

```python
def _hierarchical_guidance(self, node, index, base_scores, candidates):
    # Encontrar vecinos en espacio tensorial (radio 0.3)
    neighbors = index.query(node.tensor_position, radius=0.3)
    
    if not neighbors:
        return 0.0
    
    # Promedio de scores de vecinos — nodos rodeados de alta-score se benefician
    neighbor_scores = [base_scores[candidates.index(n)] for n in neighbors]
    return sum(neighbor_scores) / len(neighbor_scores)
```

---

## Re-ranking por Diversidad

Para evitar resultados redundantes, el sampler penaliza nodos similares a los ya seleccionados:

```python
def _diversity_rerank(self, results, holographic_encoder, top_k, diversity_weight=0.3):
    for node, score in results[1:]:
        # Penalizar similitud con nodes ya seleccionados
        penalty = max(
            encoder.similarity(node.signature, selected.signature)
            for selected, _ in final
        )
        adjusted_score = score * (1 - diversity_weight * penalty)
```

---

## Ejemplo Completo: Búsqueda con Diffusion

```python
from omega_cube.diffusion_sampler import DiffusionGraphSampler
from omega_cube.tensor_node import TensorIndex
from omega_cube.holographic import HolographicEncoder

# Setup
sampler = DiffusionGraphSampler(
    num_steps=20,
    guidance_scale=3.0,
    seed=42  # Reproducible
)

index = TensorIndex()
for node in engine.nodes.values():
    index.insert(node)

encoder = HolographicEncoder(dimension=256)

# Búsqueda difusiva
results = sampler.sample(
    query="evony generals meta",
    index=index,
    holographic_encoder=encoder,
    top_k=10,
    temperature=0.1
)

print("Resultados diffusion:")
for node, score in results:
    print(f"  {score:.3f} → [{node.primary_hierarchy}] {node.content}")
```

---

## Comparación con Búsqueda Holográfica Simple

| Aspecto | Holográfico simple | Diffusion |
|---------|-------------------|-----------|
| Complejidad | O(1) por query | O(n · steps · log n) |
| Precisión para queries específicas | ~0.53 (baseline) | ~0.79 (verificado) |
| Organizació multi-tópico | No nativa | Sí (sample_multi_topic) |
| Diversidad de resultados | No garantizada | Re-ranking automático |
| Guidance jerárquico | No | Boost por vecinos coherentes |

---

**Ver también:** [OmegaCubeEngine API](engine.md), [HolographicEncoder API](holographic.md)
