# Axion-Cube Engine — Multi-dimensional Hierarchical Graph Memory

**Versión:** 3.0.0 — Cube Move + Cadena de Color + Gate de Validez
**Autor:** Axion Research
**Fecha:** 2026-08-10

---

## 🆕 v3.0 — Cube Move + Cadena de Color (PLAN 2026-08-09)

Las metáforas de diseño del plan se implementaron como matemática exacta y se
verificaron con baterías de pruebas contra el grafo vivo
(`docs/pruebas/FASE_{0,1,2,3}_2026-08-09.md`, `FASE_4_AB_2026-08-10.md`).

| Módulo | Rol |
|---|---|
| `omega_cube/embeddings.py` | SemanticEmbedder — embeddings reales 768d (nomic-embed-text vía ollama), cache en disco |
| `omega_cube/cube_move.py` | Fase 2D vectorizada (`scores = V̂ @ q̂`, una operación matricial) + Fase 3D (giros de cara por asociaciones cruzadas) |
| `omega_cube/color_chain.py` | Cadena de color: tono por axioma (F3), degradado `sat=λ^depth` λ=0.7 (F4), mezcla por media circular vectorial (F5), linaje (F6) |
| `omega_cube/validity_gate.py` | APPROVED / VETOED / FLAGGED por linaje de color, extendiendo Gray-Scale Validation |
| `omega_cube/orchestrator.py` | Orquestador Fase 4: MARP → cube_move → gate → brief enriquecido + validación de salida de subagentes |

### Resultados verificados (output real, no estimado)

- **cube_move:** τ=0.60 calibrado con distribución de scores real (techo de temas
  inexistentes = 0.595). Giros 3D verificados en ambas direcciones. Mediana 46.6ms
  (50 nodos) / 63.5ms (71 nodos). Diff vs baseline holográfico: trae el nodo
  correcto a 0.793 donde el baseline ponía un axioma irrelevante en top-1.
- **Cadena de color:** 17 axiomas con tonos espaciados ≥21°; degradado exacto
  (0 discrepancias contra `λ^depth`); mezcla detectada con padres correctos en caso
  geométrico y disperso; linaje roto → FLAGGED señalando el nodo exacto.
- **Gate:** cadena real de 4 niveles APPROVED (sat 1.0→0.7→0.49→0.34); Gray-Scale
  existente sin regresión (determinismo exacto).
- **Prueba A/B real con subagentes (Fase 4):** preguntas sobre proyectos privados:
  sin protocolo 0/7 hechos ("NO LO SÉ"); con protocolo 7/7 hechos exactos,
  ensamblados APPROVED 12/12. MARP clasificó dominios con confianza 0.85.

### Historia de bugs críticos (lecciones pagadas)

| Bug | Causa raíz | Fix |
|---|---|---|
| Corrupción 926MB del store | `load()` no idempotente: doble load duplicaba `axiom_ids` exponencialmente | `load()` resetea estado; sin `load()` redundante tras el constructor |
| Split-brain | Indexer escribía a `cube_state.json`, que el motor nunca lee | Store único `memory/omega_cube_memory.json` vía `scripts/omega_store.py` |
| Dedup rota | `hash()` de Python es aleatorio entre procesos (PYTHONHASHSEED) | node_ids por sha256 |
| Embeddings falsos | TurboVec bridge caía a fallback de vectores aleatorios | nomic-embed-text 768d reales (ollama) |

**Regla de trabajo:** toda mejora se verifica con diff real (output antes/después);
prohibido reportar porcentajes sin evidencia. Baterías de prueba en `docs/pruebas/`.

---

## ⚛ ¿Qué es Axion-Cube?

Motor de memoria jerárquica multi-dimensional del protocolo **AXION** (AXiomatic ONtological engine).
Organiza conocimiento como "cubos magnéticos" — dominios temáticos con jerarquías internas que rotan,
se conectan y coalescen para formar patrones que responden a consultas complejas multi-tópico.

### Innovaciones integradas (10 componentes + 4 mejoras v2)

| # | Componente | Descripción |
|---|------------|-------------|
| 1 | **Tensor Hierarchies** | Nodos existen en espacios jerárquicos N-dimensionales simultáneos |
| 2 | **Holographic Encoding** | Firmas comprimidas 256D para búsqueda O(1) aproximada |
| 3 | **Quantum-Inspired Annealing** | Optimización de topología mediante recocido simulado con túnel cuántico |
| 4 | **Diffusion Graph Sampling** | Recuperación paralela no-autoregresiva inspirada en DiffusionGemma |
| 5 | **Gray-Scale Validation** | Evaluación multi-bit de verdad (inspirado en H-Bit) |
| 6 | **AutoResearch Loop** | Descubrimiento automático de conocimiento faltante |
| 7 | **Predictive Context Search** | Búsqueda predictiva con trie jerárquico y tracker de contexto |
| 8 | **Collective Hierarchy Evolution** | Evolución colectiva de jerarquías basada en señales de sesión |
| 9 | **Probabilistic Hierarchy Engine** | Motor de jerarquía probabilística para incertidumbre |
| 10 | **MARP Router** | Protocolo de enrutamiento multi-proveedor (Model-Agnostic Routing Protocol) |

### Novedades v2.0 (arXiv 2026 papers)

| # | Mejora | Paper | Impacto |
|---|--------|-------|---------|
| 11 | **HierarchicalSummarizer** | H²MT (2605.24930) | Routing O(log n) coarse-to-fine, 534x más rápido |
| 12 | **TypedSchema** | VirtualSet (2607.18821) | 100% ops inválidas bloqueadas, 0 falsos positivos |
| 13 | **BoundaryController** | PAGE-RAG (2607.19301) | Filtra resultados sin grounding suficiente |
| 14 | **HallucinationDetector** | 2607.00447 | Detecta task-retrieval y key-selection bias |

### MARP v2 (6 mejoras aplicadas)

| # | Mejora | Resultado |
|---|--------|-----------|
| 1 | Router → HierarchicalSummarizer | 100% routing jerárquico |
| 2 | Context con BoundaryController | Grounding scores activos |
| 3 | HallucinationDetector integrado | Bias detection en clasificación |
| 4 | Holographic context nodes (256D) | 100% con firma holográfica |
| 5 | AdaptiveScheduler | 8 prefetch hits, 62.5% accuracy |
| 6 | Evolving keyword rules (CORTEX) | Keywords desde el grafo |

### Benchmarks (datos reales, 2026-07-26)

**Axion-Cube Engine:**

| Métrica | v1 | v2 | Delta |
|---------|-----|-----|-------|
| Hierarchical routing | N/A | 2.6ms | 730x vs diffusion |
| Combined P@5 | 15.0% | 18.0% | +20% |
| Typed Schema blocks | 0 | 52 | 100% ops inválidas |
| Bias detections | 0 | 3 | Nuevo |

**MARP Router:**

| Métrica | v1 | v2 | Delta |
|---------|-----|-----|-------|
| Routing accuracy | 52.5% | 60.0% | +7.5% |
| Context nodes/query | 0.6 | 7.3 | 12x más contexto |
| Hierarchical routing | 0% | 100% | Nuevo |
| Holographic context | 0% | 100% | Nuevo |
| Adaptive prefetch | 0 hits | 8 hits | Nuevo |

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    Omega-Cube Engine                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ TensorNode  │  │ Holographic  │  │ QuantumInspired      │  │
│  │ (N-dim)     │◄─│ Encoder      │  │ Annealer             │  │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                │                     │              │
│         ▼                ▼                     ▼              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ TensorIndex │  │ Diffusion    │  │ GrayScale            │  │
│  │ (spatial)   │◄─│ Sampler      │  │ Validator            │  │
│  └─────────────┘  └──────────────┘  └──────────────────────┘  │
│                              │                  │              │
│                              ▼                  ▼              │
│                    ┌─────────────────────────────────────┐     │
│                    │    Pattern Emergence                │     │
│                    │  (cross-cube alignment detection)   │     │
│                    └─────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌──────────────────┐              ┌──────────────────────────┐
│  DecisionNode    │              │  Provenance Exporter     │
│  (W3C PROV-O)   │◄────────────►│  (JSON-LD / CSV / RDF)  │
└──────────────────┘              └──────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                  ConflictDetector v2                        │
│  - Contradiction-first detection                           │
│  - TF-IDF cosine similarity + token overlap                │
│  - Vendor exclusion pairs                                  │
│  - Hierarchical co-occurrence scoring                      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   OmegaCube Bridge                          │
│  (fabric_recall ↔ Omega-Cube semantic search)              │
│  - HolographicEncoder O(1) approximate retrieval           │
│  - Disciminative scoring: 0.79 vs 0.53 baseline            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Instalación y Setup

### Requisitos

- Python 3.11+
- numpy (para operaciones vectoriales)

```bash
pip install numpy
```

### Estructura de directorios

```
axioma-omega-protocol/
├── omega_cube/                    # Motor principal
│   ├── __init__.py               # Exportaciones públicas
│   ├── engine.py                 # OmegaCubeEngine (entry point)
│   ├── tensor_node.py            # TensorNode + TensorIndex
│   ├── holographic.py            # HolographicEncoder
│   ├── annealer.py               # QuantumInspiredAnnealer + CubeRotator
│   ├── diffusion_sampler.py      # DiffusionGraphSampler
│   ├── grayscale.py              # GrayScaleValidator
│   ├── decision_node.py          # DecisionNode (W3C PROV-O)
│   ├── conflict_detector_v2.py   # ConflictDetector v2
│   ├── provenance_export.py      # ProvenanceExporter
│   ├── autoresearch.py           # AutoResearchLoop
│   ├── predictive_search.py      # PredictiveContextSearch + HierarchicalTrie
│   ├── collective_evolution.py   # CollectiveHierarchyEngine
│   └── probabilistic_hierarchy.py# ProbabilisticHierarchyEngine
├── memory/                       # Estado persistente
│   └── omega_cube_memory.json    # Base de datos de nodos (18+ nodes)
├── scripts/
│   ├── omega_auto_indexer.py     # Indexador automático de sesiones
│   └── index_and_notify.sh       # Wrapper para cron job (6h)
├── dashboard.html                # Dashboard web interactivo
└── README.md                     # Esta documentación
```

---

## 🚀 Uso Rápido

### Carga del motor

```python
from omega_cube import OmegaCubeEngine

engine = OmegaCubeEngine()  # Auto-carga desde memory/omega_cube_memory.json
print(f"Cargados: {len(engine.nodes)} nodos")
# → Cargados: 18 nodos
```

### Agregar un nodo

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
print(f"Nodo: {node.node_id}")
# → Nodo: a1b2c3d4e5f6g7h8
```

### Búsqueda semántica (Holográfica O(1))

```python
results = engine.query("best image generation workflow", mode="diffusion")
for r in results[:3]:
    print(f"{r['score']:.2f} → {r['content']} [{', '.join(r['hierarchies'])}]")
# 0.85 → SDXL is better for quality [COMFYUI.WORKFLOWS.GENERATION]
# 0.72 → ComfyUI workflow para alta resolución [QUALITY.IMAGE_RESOLUTION.HIGH]
```

### Búsqueda multi-tópico

```python
results = engine.query(
    "cloud provider comparison",
    mode="combined",
    top_k=10
)
# Retorna resultados organizados por dimensión jerárquica
```

---

## 📚 Documentación de Componentes API

### 1. `OmegaCubeEngine` — Motor Central

**Ubicación:** `omega_cube/engine.py`  
**Responsabilidad:** Orquestar todos los sub-motores y manejar persistencia.

#### Métodos principales

| Método | Parámetros | Retorna | Descripción |
|--------|-----------|---------|-------------|
| `__init__` | `memory_dir: str = None`, `holographic_dim: int = 256`, `tensor_grid_size: int = 10` | `OmegaCubeEngine` | Inicializa motor con directorio de memoria |
| `add_node` | `content: str`, `hierarchies: list[str]`, `tensor_position: list[float] = None`, `node_type: str = "CONCEPT"`, `confidence: float = 0.9`, `tags: list = None` | `TensorNode` | Agrega nodo al grafo multi-dimensional |
| `query` | `query: str`, `mode: str = "combined"`, `top_k: int = 10` | `list[dict]` | Ejecuta búsqueda semántica (holográfica/difusión/hibrida) |
| `find_patterns` | `query: str`, `threshold: float = 0.7` | `list[dict]` | Detecta patrones emergentes entre cubos |
| `load` | — | `bool` | Carga estado desde memory_dir (True si éxito) |
| `save` | — | `bool` | Persiste estado a memory_dir |

#### Modos de búsqueda disponibles

```python
engine.query("query", mode="holographic")  # O(1) aproximado via HolographicEncoder
engine.query("query", mode="diffusion")    # No-autoregresivo con annealing
engine.query("query", mode="combined")     # Hibrido (default)
```

---

### 2. `TensorNode` — Nodo Multi-Dimensional

**Ubicación:** `omega_cube/tensor_node.py`  
**Responsabilidad:** Representar conocimiento en N dimensiones jerárquicas simultáneas.

#### Atributos

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `content` | `str` | Texto del nodo (ej: "AWS es mejor para escalabilidad") |
| `hierarchies` | `list[str]` | Múltiples paths jerárquicos ej: `["COMFYUI.WORKFLOWS", "QUALITY.HIGH"]` |
| `tensor_position` | `list[float]` | Coordenadas normalizadas [0,1]^N en espacio tensorial |
| `node_type` | `str` | AXIOM / CONCEPT / INSTANCE / SESSION |
| `confidence` | `float` | Confianza base 0.0-1.0 |
| `tags` | `list[str]` | Tags semánticos para búsqueda adicional |
| `associations` | `list[str]` | IDs de nodos asociados |
| `holographic_signature` | `list[float]` | Firma holográfica (256-dim) |
| `gray_scale` | `dict` | Evaluación multi-bit {dim: score_0-100} |

#### Métodos útiles

```python
node = TensorNode(content="test", hierarchies=["TOPIC.SUB"])

# Acceso a jerarquía primaria
print(node.primary_hierarchy)  # → "TOPIC.SUB"

# Conteo de dimensiones
print(node.dimension_count)    # → 1 (o N si múltiples hierarquías)

# Distancia euclidiana normalizada en espacio tensorial
dist = node.distance_to(other_node)  # → float [0,1]

# Verificación gray-scale por dimensión
score = node.gray_scale_score("factuality")  # → float 0-100
```

---

### 3. `HolographicEncoder` — Compresión Distribuida

**Ubicación:** `omega_cube/holographic.py`  
**Responsabilidad:** Codificar nodos en vectores fijos que codifican su vecindario estructural completo.

#### Concepto clave

> Un nodo puede ser decodificado para recuperar información sobre sus vecinos **sin traversar el grafo**.

#### Métodos principales

| Método | Parámetros | Retorna | Descripción |
|--------|-----------|---------|-------------|
| `encode_node` | `content: str`, `hierarchy: str` | `list[float]` (256-dim) | Crea vector base desde contenido + jerarquía |
| `bind` | `v1: list[float]`, `v2: list[float]` | `list[float]` | Convolución circular (asociación) |
| `unbind` | `bound: list[float]`, `v1: list[float]` | `list[float]` | Recupera v2 de bind(v1, v2) |
| `bundle` | `vectors: list[list[float]]` | `list[float]` | Superposición (combinación múltiple) |
| `encode_holographic_signature` | `node_content`, `node_hierarchy`, `parent_*`, `children`, `neighbors` | `list[float]` | Firma completa del contexto estructural |
| `similarity` | `v1: list[float]`, `v2: list[float]` | `float` [0,1] | Similitud coseno normalizada |
| `partial_match` | `query_vec`, `signature` | `float` [0,1] | Verifica si query está contenido en signature |

#### Ejemplo de uso

```python
encoder = HolographicEncoder(dimension=256)

# Codificar nodo con contexto completo
signature = encoder.encode_holographic_signature(
    node_content="AWS es mejor",
    node_hierarchy="CLOUD.PROVIDERS.AWS",
    parent_content="Cloud Providers",
    parent_hierarchy="TECH.CLOUD",
    children=[("EC2", "INFRASTRUCTURE.EC2")],
    neighbors=[("Azure", "CLOUD.PROVIDERS.AZURE")]
)

# Buscar nodos similares (O(1))
results = encoder.partial_match(query_signature, node.signature)
```

---

### 4. `QuantumInspiredAnnealer` — Optimización Topológica

**Ubicación:** `omega_cube/annealer.py`  
**Responsabilidad:** Encontrar configuraciones óptimas de cubos mediante recocido simulado con túnel cuántico.

#### Concepto clave

> Cada "cubo" (dominio temático) busca su configuración óptima simultáneamente. El sistema converge a un estado de energía mínima donde todos los cubos están alineados para responder la consulta.

#### Métodos principales

| Método | Parámetros | Retorna | Descripción |
|--------|-----------|---------|-------------|
| `anneal` | `cubes: list[dict]`, `energy_fn`, `neighbor_fn`, `max_iterations=500` | `(best_config, best_energy, energy_history)` | Ejecuta annealing para encontrar configuración óptima |
| `multi_objective_anneal` | `cubes`, `energy_fns: list[(fn, weight)]`, `neighbor_fn` | `(best_config, history)` | Annealing multi-objetivo con pesos |

#### Parámetros configurables

```python
annealer = QuantumInspiredAnnealer(
    initial_temp=1.0,         # Temperatura inicial
    cooling_rate=0.95,        # Factor de enfriamiento (0-1)
    min_temp=0.01,            # Temperatura mínima (stop condition)
    steps_per_temp=5,         # Iteraciones por nivel de temperatura
    tunneling_prob=0.1,       # Probabilidad de túnel cuántico (escape de mínimos locales)
    seed=42                   # Seed determinista opcional
)
```

---

### 5. `DiffusionGraphSampler` — Recuperación Paralela

**Ubicación:** `omega_cube/diffusion_sampler.py`  
**Responsabilidad:** Generar salida de grafo no-autoregresiva mediante denoising iterativo inspirado en DiffusionGemma.

#### Concepto clave

> En vez de predecir el siguiente nodo secuencialmente (O(n)), muestreamos todos los candidatos simultáneamente y refinamos basados en estructura jerárquica (O(log n)).

#### Métodos principales

| Método | Parámetros | Retorna | Descripción |
|--------|-----------|---------|-------------|
| `sample` | `query: str`, `index: TensorIndex`, `holographic_encoder`, `top_k=10`, `temperature=0.1` | `list[(TensorNode, float)]` | Muestreo difusivo con denoising iterativo |
| `sample_multi_topic` | `query`, `index`, `encoder`, `topic_dimensions: list[str]`, `top_k_per_topic=3` | `dict[str, list[(node, score)]]` | Muestreo independiente por dimensión temática |

#### Parámetros configurables

```python
sampler = DiffusionGraphSampler(
    num_steps=20,           # Iteraciones de denoising (más = más preciso)
    guidance_scale=3.0,     # Peso del guidance jerárquico (más alto = más enfocado)
    seed=None               # Seed para reproducibilidad
)

# Uso con TensorIndex existente
results = sampler.sample(
    query="best image generation",
    index=tensor_index,
    holographic_encoder=encoder,
    top_k=5,
    temperature=0.1  # Bajo = determinista; alto = exploratorio
)
```

---

### 6. `GrayScaleValidator` — Evaluación Multi-Bit de Verdad

**Ubicación:** `omega_cube/grayscale.py`  
**Responsabilidad:** Evaluar nodos en múltiples dimensiones de verdad (no binario), produciendo perfiles gray-scale con cuantificación de incertidumbre.

#### Dimensiones evaluadas

| Dimensión | Descripción | Peso por defecto |
|-----------|-------------|-----------------|
| `factuality` | Anclaje a axiomas verificados | 0.35 |
| `relevance` | Alineamiento con query | 0.25 |
| `recency` | Frescura temporal (half-life: 30 días) | 0.10 |
| `coherence` | Consistencia interna con nodos relacionados | 0.15 |
| `provenance` | Trazabilidad a fuente | 0.10 |
| `specificity` | Granularidad de detalle (indicadores: fechas, paths, configs) | 0.05 |

#### Métodos principales

| Método | Parámetros | Retorna | Descripción |
|--------|-----------|---------|-------------|
| `evaluate_node` | `node`, `query: str = ""`, `axioms: list = None`, `related_nodes: list = None` | `dict[str, float]` {dim: score_0-100} | Evalúa nodo en todas las dimensiones |
| `composite_score` | `gray_profile: dict`, `weights: dict = None` | `float` [0-100] | Combina dimensiones con pesos personalizables |
| `partial_evidence_score` | `gray_profile`, `available_dimensions: list[str]` | `float` [0-100] | Score incluso con evidencia parcial (estilo H-Bit) |
| `verify_against_axioms` | `node`, `axioms: list`, `threshold=60.0` | `(bool, float, str)` | Verifica anclaje factual contra axiomas conocidos |
| `compute_gray_scale_hash` | `gray_profile: dict[str, float]` | `str` (12 chars) | Hash fingerprint para comparación rápida |

#### Ejemplo de uso

```python
validator = GrayScaleValidator()

# Evaluar nodo completo
profile = validator.evaluate_node(
    node=my_node,
    query="best image generation",
    axioms=[axiom1, axiom2],
    related_nodes=[related1, related2]
)
print(profile)
# → {factuality: 85.0, relevance: 92.3, recency: 78.5, ...}

# Score compuesto con pesos custom
composite = validator.composite_score(
    profile, 
    weights={"factuality": 0.5, "relevance": 0.3, "recency": 0.2}
)

# Verificación parcial (solo 2 de 6 dimensiones disponibles)
partial = validator.partial_evidence_score(profile, ["factuality", "provenance"])
```

---

### 7. `DecisionNode` — Nodos de Decisión con Proveniencia W3C PROV-O

**Ubicación:** `omega_cube/decision_node.py`  
**Responsabilidad:** Representar decisiones con metadata estructurada y trazabilidad completa.

#### Atributos específicos (además de TensorNode)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `category` | `str` | Categoría de decisión (ej: "vendor_selection") |
| `scenario` | `str` | Descripción del escenario (ej: "Choose cloud provider for HIPAA workload") |
| `reasoning` | `str` | Cadena de razonamiento (lista separada por `\n`) |
| `outcome` | `str` | Resultado final (ej: "selected_aws") |

#### Métodos principales

```python
decision = DecisionNode(
    category="vendor_selection",
    scenario="Choose cloud provider for HIPAA workload",
    reasoning="AWS offers BAA, mature HIPAA tooling...",
    outcome="selected_aws",
    confidence=0.93
)

# Trazabilidad W3C PROV-O
prov = decision.trace_chain()
# → {"@context": "https://www.w3.org/ns/prov.jsonld", "@type": "prov:Entity", ...}

# Exportar como dict estructurado
d = decision.to_decision_dict()
# → {"id": "...", "category": "vendor_selection", "scenario": "...", "reasoning": "...", "outcome": "selected_aws", ...}

# Reconstrucción desde dict
restored = DecisionNode.from_dict(d)
```

---

### 8. `ConflictDetector v2` — Detección Semántica de Contradicciones

**Ubicación:** `omega_cube/conflict_detector_v2.py`  
**Responsabilidad:** Detectar contradicciones explícitas primero (bypass similarity filter), luego calcular confianza basada en superposición jerárquica.

#### Mejoras sobre v1

- ✅ Tokenización inteligente (maneja underscores, compound words)
- ✅ Similaridad coseno TF-IDF para mejor captura semántica
- ✅ Pares contradictorios expandidos con sinónimos contextuales + exclusiones de vendor
- ✅ Detección contradiction-first: patrones explícitos bypassan filtro de similitud
- ✅ Co-ocurrencia jerárquica como señal de confianza

#### Patrones detectados

```python
# Contradicciones directas (antónimos)
("selected", "rejected"), ("approved", "denied"), ("yes", "no")

# Sinónimos de resultado de decisión
("chosen", "discarded"), ("preferred", "rejected"), ("adopted", "abandoned")

# Exclusiones de vendor/tecnología (elegir uno implica rechazar el otro)
("aws", "azure"), ("aws", "gcp"), ("kubernetes", "docker_compose")
```

#### Ejemplo de uso

```python
from omega_cube.conflict_detector_v2 import ConflictDetectorV2

detector = ConflictDetectorV2(similarity_threshold=0.3)

conflicts = detector.detect_conflicts(
    new_node=my_new_node,
    existing_nodes=[existing_node1, existing_node2]
)

for c in conflicts:
    print(f"Tipo: {c['type']} | Severidad: {c['severity']}")
    # → Tipo: VENDOR_EXCLUSION | Severidad: HIGH
    # → Tipo: DECISION_OUTCOME_CONFLICT | Severidad: MEDIUM
```

---

### 9. `ProvenanceExporter` — Exportación W3C PROV-O

**Ubicación:** `omega_cube/provenance_export.py`  
**Responsabilidad:** Exportar proveniencia de nodos en formatos estandarizados (JSON-LD, CSV, RDF).

#### Métodos principales

| Método | Parámetros | Retorna | Descripción |
|--------|-----------|---------|-------------|
| `export_prov_o_jsonld` | `node_id`, `node_data`, `activity_data=None` | `dict` (JSON-LD) | Exporta en formato W3C PROV-O JSON-LD |
| `export_csv` | `nodes: list[dict]`, `output_path="provenance_export.csv"` | `str` (path) o `"No data to export"` | Exporta múltiples nodos a CSV |
| `export_simple_json` | `node_id`, `node_data`, `output_path=None` | `str` (JSON string) o path | Exportación simple para debugging |

---

### 10. `OmegaCube Bridge` — Integración con fabric_recall

**Ubicación:** `omega_cube/omega_cube_bridge.py`  
**Responsabilidad:** Traducir entre la API de fabric y las queries semánticas del motor Omega-Cube.

#### Métodos principales

```python
from omega_cube.omega_cube_bridge import OmegaCubeBridge

bridge = OmegaCubeBridge()

# Búsqueda semántica (reemplaza keyword matching)
results = bridge.semantic_search("evony generals meta", top_k=5)
# → [{"node_id": "...", "content": "...", "hierarchies": [...], "score": 0.85}, ...]

# Query fabric entries via Omega-Cube
results = bridge.query_fabric(
    query="como configurar ComfyUI", 
    agent=None, 
    project=None, 
    max_results=10
)
```

#### Métricas de rendimiento verificadas

| Métrica | Resultado |
|---------|-----------|
| Nodos cargados | 18 (CONCEPT: 14, AXIOM: 3, SESSION: 1) |
| Búsqueda discriminativa | 0.79 vs 0.53 baseline (keyword matching) |
| Tiempo promedio recuperación | ~0ms (O(1) holográfico) |

---

## 🌐 Dashboard Web Interactivo

**Ubicación:** `dashboard.html`  
**Acceso:** http://localhost:9091/dashboard.html  

### Funcionalidades

- Visualización de jerarquías principales (EVONY: 10, COMFYUI: 5, HBIT: 4, OMEGA-CUBE: 4)
- Nodos recientes con contenido real
- Filtros por dimensión jerárquica
- Tema oscuro con CSS variables personalizables

### Ejecución

```bash
python -m http.server 9091 --directory .
# → Dashboard accesible en http://localhost:9091/dashboard.html
```

---

## ⏰ Automatización con Cron Job

**Script:** `scripts/index_and_notify.sh`  
**Frecuencia:** Cada 6 horas (`0 */6 * * *`)  

### Ejecución manual

```bash
cd /c/Users/GPAMD/.hermes/axioma-omega-protocol
python scripts/omega_auto_indexer.py
# → Indexa sesiones recientes en Omega-Cube
```

### Integración con cron (Windows Task Scheduler)

```powershell
schtasks /create /tn "OmegaCubeIndexer" /tr "python C:\Users\GPAMD\.hermes\axioma-omega-protocol\scripts\omega_auto_indexer.py" /sc daily /st 06:00
```

---

## 🧪 Verificación y Testing

### Script de verificación rápido

```bash
cd /c/Users/GPAMD/.hermes/axioma-omega-protocol
python -c "from omega_cube.omega_cube_bridge import OmegaCubeBridge; b=OmegaCubeBridge(); s=b.get_stats(); print('Nodes:', s['total_nodes']); r=b.semantic_search('evony generals', top_k=3); print('Search OK:', len(r)>0)"
# → Nodes: 18 | Search OK: True
```

### Tests de ConflictDetector v2 (5 casos)

| Test | Caso | Resultado esperado |
|------|------|-------------------|
| 1 | aws vs gcp | 1 conflicto (VENDOR_EXCLUSION HIGH) |
| 2 | selected_aws vs preferred_aws | 0 conflictos (no contradicción) |
| 3 | approved vs denied | 1 conflicto (DECISION_OUTCOME_CONFLICT) |
| 4 | k8s vs budget | 0 conflictos (temas diferentes) |
| 5 | build vs buy | 1 conflicto (decision contradiction) |

---

## 📊 Estado Actual del Sistema

- **Nodos totales:** 18+ (ver `memory/omega_cube_memory.json`)
- **Jerarquías principales:** EVONY, COMFYUI, HBIT, OMEGA-CUBE, DECISIONS, QUALITY
- **Motor verificado:** Bridge operativo con búsqueda discriminativa
- **ConflictDetector v2:** 5/5 tests passing
- **Dashboard:** Funcional (19 nodos cargados)

---

## 🔗 Referencias

- [Holographic Reduced Representations](https://dl.acm.org/doi/10.1.1.47.6830) (Plate, 1995)
- [Tensor Product Representations](https://link.springer.com/article/10.3758/BF03203279) (Smolensky, 1990)
- [DiffusionGemma](https://blog.google/technology/research/diffusiongemma/) (Google DeepMind, 2026)
- [H-Bit Protocol](https://github.com/h-bit) — Esteganografía criptográfica universal
- [W3C PROV-O](https://www.w3.org/TR/vocab-prov-o/) — Ontología de proveniencia

---

**Última actualización:** 2026-06-29  
**Mantenimiento:** Omega-Cube Research Team
