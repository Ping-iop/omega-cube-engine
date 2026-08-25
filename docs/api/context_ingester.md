# Context Ingester API Reference

**File:** `omega_cube/context_ingester.py` (308 lines, 12,731 bytes)  
**Purpose:** Trocea conversaciones largas en segmentos temáticos y las ingiere como nodos jerárquicos en Omega-Cube, permitiendo recuperar contexto relevante sin perder detalle por límite de tokens.

## Concepto Central

En lugar de meter todo el historial de una conversación en el prompt (donde se pierde calidad por el límite de tokens), este módulo:
1. **Trocea** la conversación en segmentos por tema detectado automáticamente
2. Cada segmento → nodo en Omega-Cube con jerarquía multi-dimensional
3. Los nodos se conectan por co-ocurrencia (grafo)
4. Al buscar, Omega-Cube devuelve solo los N nodos más relevantes
5. Contexto perfecto sin perder detalle — permite conversaciones de 100k+ tokens inyectando menos de 4k tokens

## Clases

### `ConversationChunker`

Trocea conversación en segmentos por tema usando detección por señales léxicas.

#### `chunk_by_topic(text: str, max_chars: int = 500) -> list[dict]`

Divide texto en segmentos temáticos. Separa por párrafos (`\n\n`) y detecta cambios de tema usando las señales predefinidas. Cuando cambia el tema o se excede `max_chars`, flush del segmento actual.

**Retorna:** Lista de dicts con:
```python
{
    "topic": str,           # Tema detectado (math, code, science, etc.)
    "hierarchy": str,       # Ruta jerárquica (ej: "code.code_42")
    "content": str,         # Texto del segmento
    "confidence": float,    # Siempre 0.7 (fijo)
    "timestamp": str,       # ISO timestamp
    "id": str               # MD5 hash corto (12 chars)
}
```

#### `_detect_topic(text: str) -> dict`

Detecta el tema más probable para un segmento de texto usando keyword matching sobre `TOPIC_SIGNALS`. Retorna `{"topic": "general", "hierarchy": "general"}` si no hay señales.

### `OmegaContextIngester`

Ingiere contexto de conversación en Omega-Cube con detección automática de temas y búsqueda predictiva.

#### `__init__()`
Inicializa engine, PCS (PredictiveContextSearch), chunker y stats counters.

```python
self.engine = OmegaCubeEngine()
self.pcs = PredictiveContextSearch()
self.chunker = ConversationChunker()
self.stats = {"nodes_added": 0, "chunks_processed": 0}
```

#### `ingest(text: str, source: str = "conversation") -> list[dict]`

Ingiere texto en Omega-Cube con jerarquías detectadas. Procesa cada chunk calculando posición tensor y agregándolo al engine + trie PCS.

```python
ingester = OmegaContextIngester()
nodes = ingester.ingest("Mi conversación de 50k tokens...")
# nodes: lista de dicts con topic, hierarchy, content, etc.
print(ingester.stats)  # {"nodes_added": N, "chunks_processed": M}
```

#### `search_context(query: str, top_k: int = 5) -> list[dict]`

Busca los N nodos más relevantes combinando PCS (prefix search por dominio) + engine query (ranking holográfico). Merge de resultados eliminando duplicados.

```python
results = ingester.search_context("como configuré el servidor", top_k=3)
# results: lista de dicts ordenados por relevancia
```

#### `to_prompt_context(query: str, top_k: int = 5) -> str`

Genera contexto en formato prompt listo para inyectar al worker. Formato:
```
<context from omega-cube>
[1] (math.calculus_123): derivada de x^2...
[2] (code.python_456): async function con error handling...
</context>
```

#### `save(path: str = None)` / `load(path: str = None)`

Persiste/carga el estado del engine en `context_state.json` por defecto.

### `_hash_to_float(h: str) -> float` (privado)

Convierte hash MD5 a float normalizado [0,1] para posición tensor.

### `_confidence_to_pos(conf: float) -> float` (privado)

Mapea confianza [0.9] → posición [0.75]. Fórmula: `0.3 + conf * 0.5`.

## TOPIC_SIGNALS — Temas Detectables

| Tema | Keywords principales |
|------|---------------------|
| math | calculus, derivative, integral, algebra, equation, theorem |
| code | python, javascript, rust, function, class, api, docker |
| science | quantum, physics, chemistry, biology, dna, neuron |
| engineering | circuit, bridge, mechanical, sensor, motor, pid |
| law | contract, nda, patent, copyright, court, statute |
| medical | diagnosis, symptom, treatment, surgery, disease |
| business | revenue, profit, investment, npv, startup, valuation |
| philosophy | ethics, kant, free will, consciousness, morality |
| gaming | game, rpg, roguelike, moba, player, mechanic |
| language | translate, grammar, syntax, linguistics, poem |
| omega-cube | tensor, holographic, hierarchical, predictive |
| h-bit | steganography, spectrum, verification, payload |
| evony | march, rally, ranged, mounted, monarch, boss |
| hermes | agent, cron, mcp, skill, session, tool |

## Uso CLI

```bash
# Ingerir texto directo
echo "Mi conversación..." | python context_ingester.py

# Desde archivo
python context_ingester.py --from-file chat_log.txt

# Buscar contexto para query
python context_ingester.py --query "como configuré el servidor"

# Modo interactivo
python context_ingester.py --interactive
# Comandos: /search <q>  /stats  /save  /load  quit
```

## Arquitectura del Pipeline

```
Conversación → TopicDetector (TOPIC_SIGNALS) → Chunker
                                                        ↓
                          PredictiveContextSearch ← OmegaCubeEngine.add_node()
                                                        ↓
                              Query → PCS prefix + Engine holographic → Top-5 nodos
                                                        ↓
                                              to_prompt_context() → Worker responde
```

## Notas de Implementación

- **Confianza fija:** Todos los chunks se crean con confidence=0.7 (no hay scoring dinámico)
- **ID determinístico:** MD5 del contenido (12 chars), permite deduplicación
- **Jerarquía dinámica:** `{tema}.{tema}_{timestamp % 1000}` — crea rutas únicas por segmento
- **PCS integration:** Prefijo de 4 chars del tema se inserta en trie para búsqueda rápida
- **Persistencia separada:** Usa `context_state.json` (no el memory dir estándar)
