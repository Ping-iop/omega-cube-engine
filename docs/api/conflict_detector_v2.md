# ConflictDetector v2 API Reference

**Ubicación:** `omega_cube/conflict_detector_v2.py` (261 líneas, 9,943 bytes)  
**Autor:** Bit/Hermes  
**Última actualización:** 2026-06-29  

## Resumen

`ConflictDetectorV2` detecta contradicciones explícitas primero (contradiction-first), bypassando el filtro de similitud. Luego calcula confianza basada en superposición jerárquica y similaridad TF-IDF coseno. Resuelve el problema de v1 donde contradictores semánticos pasaban desapercibidos por el filtro de similitud textual.

**Mejoras sobre v1:**
- Contradicciones explícitas detectadas antes que similitud (bypass filter)
- Tokenización inteligente (maneja underscores, compound words)
- Similaridad TF-IDF coseno + token overlap Jaccard
- Pares contradictorios expandidos con sinónimos contextuales + exclusiones de vendor
- Co-ocurrencia jerárquica como señal de confianza (60% weight)

## Clase: `ConflictDetectorV2`

### Inicialización

```python
from omega_cube.conflict_detector_v2 import ConflictDetectorV2

detector = ConflictDetectorV2(
    similarity_threshold=0.3   # Umbral mínimo para flaggear como MEDIUM severity
)
```

El umbral se bajó de 0.85 (v1) a 0.3 porque la detección contradiction-first no depende de este filtro para el filtrado inicial.

### Constantes de Clase

**`CONTRADICTORY_PATTERNS`** — Pares directos:
```python
("selected", "rejected"), ("approved", "denied"),
("yes", "no"), ("true", "false"), ("positive", "negative"),
("increase", "decrease"), ("expand", "contract"),
("adopt", "abandon"), ("chosen", "discarded"),
("preferred", "rejected"), ("favored", "disfavored"),
("recommended", "advised_against"), ("implemented", "deferred"),
("pursued", "abandoned"), ("committed_to", "backed_off_from"),
("moved_forward", "halted"), ("go_ahead", "no_go"),
("proceed", "pause"), ("accept", "decline"),
("include", "exclude"), ("enable", "disable"),
("proactive", "reactive"), ("internal", "outsourced"),
("build", "buy")
```

**`VENDOR_EXCLUSIONS`** — Mutual exclusion tecnológica:
```python
("aws", "azure"), ("aws", "gcp"), ("azure", "gcp"),
("kubernetes", "docker_compose"), ("monolith", "microservices")
```

### Método Principal: `detect_conflicts()`

```python
def detect_conflicts(
    self,
    new_node,                     # Nodo nuevo a verificar
    existing_nodes: list          # Nodos existentes para comparar
) -> list[dict]                  # Lista de conflictos encontrados
```

**Algoritmo:**
1. Para cada nodo existente, checkear patrones contradictorios explícitos (bypassa similarity filter)
2. Si se detecta contradicción: calcular `composite_score = 0.4 * text_sim + 0.6 * hierarchy_sim`
3. Reportar solo si `composite_score >= self.similarity_threshold`

**Estructura de retorno (por conflicto):**
```python
{
    "node_a": str,                    # node_id del nodo existente
    "node_b": str,                    # node_id del nodo nuevo
    "type": str,                      # VENDOR_EXCLUSION / DECISION_OUTCOME_CONFLICT / EXPLICIT_CONTRADICTION
    "severity": str,                  # HIGH (composite > 0.7) / MEDIUM (>= threshold)
    "similarity_score": float,        # Composite score (rounded to 3 decimals)
    "text_similarity": float,         # TF-IDF cosine similarity component
    "hierarchy_overlap": float,       # Jaccard sobre hierarchies listas
    "contradiction_pairs": list[tuple],# Pares de patrones detectados
    "description": str                # Descripción legible del conflicto
}
```

**Ejemplo:**
```python
from omega_cube.tensor_node import TensorNode

detector = ConflictDetectorV2()

node_a = TensorNode(content="selected_aws as primary cloud", hierarchies=["DECISIONS.CLOUD"])
node_b = TensorNode(content="preferred_gcp for cost optimization", hierarchies=["DECISIONS.CLOUD"])

conflicts = detector.detect_conflicts(node_b, [node_a])
# → [{"type": "VENDOR_EXCLUSION", "severity": "HIGH", ...}]
```

## Métodos Internos (Privados)

| Método | Descripción |
|--------|-------------|
| `_tokenize_smart(text)` | Separa compound words por underscores; filtra tokens < 2 chars |
| `_calculate_semantic_similarity(text_a, text_b)` | TF-IDF coseno (60%) + Jaccard token overlap (40%), retorna float 0-1 |
| `_calculate_hierarchy_overlap(hier_a, hier_b)` | Jaccard similarity sobre sets de hierarchy paths |
| `_detect_explicit_contradiction(text_a, text_b)` | Detecta patrones en `CONTRADICTORY_PATTERNS` y `VENDOR_EXCLUSIONS`; retorna `(bool, dict)` con type classification (VENDOR_EXCLUSION / DECISION_OUTCOME_CONFLICT / EXPLICIT_CONTRADICTION) |

## Tipos de Contradicción

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| `VENDOR_EXCLUSION` | Elegir uno implica rechazar el otro | aws vs gcp, k8s vs docker_compose |
| `DECISION_OUTCOME_CONFLICT` | Resultados opuestos a misma decisión | "chosen" vs "discarded", "approved" vs "denied" |
| `EXPLICIT_CONTRADICTION` | Antónimos directos en texto | "selected" vs "rejected", "build" vs "buy" |

## Compatibilidad hacia atrás

```python
from omega_cube.conflict_detector_v2 import ConflictDetector  # Alias de ConflictDetectorV2
```

## Uso Completo: Pipeline de Detección

```python
from omega_cube.conflict_detector_v2 import ConflictDetectorV2
from omega_cube.tensor_node import TensorNode

detector = ConflictDetectorV2(similarity_threshold=0.3)

existing_nodes = [
    TensorNode(content="selected_aws as primary cloud", hierarchies=["DECISIONS.CLOUD"]),
    TensorNode(content="approved_budget_request Q1 2026", hierarchies=["DECISIONS.BUDGET"]),
]

new_node = TensorNode(
    content="preferred_gcp for cost optimization",
    hierarchies=["DECISIONS.CLOUD"]
)

all_conflicts = detector.detect_conflicts(new_node, existing_nodes)
for c in all_conflicts:
    print(f"[{c['severity']}] {c['type']}")
    print(f"  Score: {c['similarity_score']:.3f} | Overlap: {c['hierarchy_overlap']:.3f}")
```

## Métricas de Rendimiento Verificadas (5 tests)

| Test | Caso | Tipo Detectado | Severidad | Resultado |
|------|------|----------------|-----------|-----------|
| 1 | aws vs gcp | VENDOR_EXCLUSION | HIGH | ✅ 1 conflicto |
| 2 | selected_aws vs preferred_aws | — | — | ✅ 0 conflictos (no contradicción) |
| 3 | approved vs denied | DECISION_OUTCOME_CONFLICT | MEDIUM | ✅ 1 conflicto |
| 4 | k8s vs budget (temas diferentes) | — | — | ✅ 0 conflictos (sin falsos positivos) |
| 5 | build vs buy | EXPLICIT_CONTRADICTION | HIGH | ✅ 1 conflicto |

---

**Ver también:** [DecisionNode API](decision_node.md), [TensorNode API](tensor_node.md)
