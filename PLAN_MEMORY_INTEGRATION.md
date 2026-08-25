# Plan: Integración Unificada Axioma-Omega → Omega-Cube → MARP

## Objetivo
Crear pipeline de memoria unificado donde cada decisión, concepto o instancia fluya automáticamente entre los tres sistemas.

---

## Estado Actual (Jun 28 2026)

| Sistema | Nodos | Scripts Crónicos | Status |
|---------|-------|------------------|--------|
| Axioma-Omega | 21 (1 axioma, 0 conceptos) | ✅ Funciona | ⚠️ Vacío de contenido real |
| Omega-Cube | ~15 nodos sintéticos | ❌ Scripts faltan | 🛑 Motor sin gasolina |
| MARP Router | Qwen0.8B GPU | ⚠️ Healthchecks fallan | ⚡ Intermitente |
| Fabric Memory | Activo | ✅ Cron jobs activos | ✅ Operativo |

---

## Fase 1: Crear Scripts de Automatización (Hoy)

### ✅ omega_auto_indexer.py
- Lee sesiones recientes via session_search  
- Trocea en segmentos por tema
- Ingresa a Omega-Cube como nodos multi-dimensionales
- Crea asociaciones cruzadas entre dominios

### ✅ omega_daily_evolution.py  
- Ejecuta AutoResearch loop semanal
- Analiza patrones de uso del grafo
- Sugiere nuevas asociaciones basadas en co-ocurrencia

### ✅ omega_autoresearch_weekly.py
- Busca papers relevantes (arXiv, web)
- Extrae conocimiento y lo ingresa como CONCEPT/INSTANCE
- Actualiza axiomas si hay nueva evidencia contradictoria

### ✅ omega_autopublisher.py
- Verifica que el grafo esté saludable (>50 nodos, asociaciones cruzadas)
- Genera release notes automáticas
- Sube a GitHub si score > 0.45

---

## Fase 2: Pipeline Integrado (Mañana)

### Flujo de Decisión Unificado
```
Usuario toma decisión → 
  ├─ Axioma-Omega: Guarda como CONCEPT con jerarquía
  │   └─ hierarchy="DECISIONES.PROYECTO.X.Y"
  │
  ├─ Omega-Cube: Tensor N-dim (sesión, turno, tema, tipo)
  │   └─ PredictiveSearch activa para recuperación contextual
  │
  ├─ Fabric Memory: Persistencia compartida cross-agent
  │   └─ training_value=high si es decisión con outcome
  │
  └─ MARP Router: Clasifica dominio → decide thinking ON/OFF
      └─ math/science → thinking ON (Qwen27B)
      └─ general → thinking OFF (respuesta rápida)
```

### Hook de Integración
Crear función `unified_memory_save()` que:
1. Recibe decisión/concepto/instancia
2. Guarda en Axioma-Omega con jerarquía semántica
3. Indexa en Omega-Cube como TensorNode N-dim
4. Escribe en fabric memory con metadata de trazabilidad
5. Crea asociaciones cruzadas si detecta conexión entre dominios

---

## Fase 3: Maintenance Localizado (Semanal)

Basado en paper arXiv: "localized maintenance is more cost-efficient than global reorganization"

### Loop de Mantenimiento Automatizado
```python
def localized_maintenance():
    # 1. Detectar entradas stale (>7 días sin uso)
    stale = find_stale_entries(days=7, usage_count=0)
    
    # 2. Detectar duplicados o contradictorios
    duplicates = find_duplicate_or_contradictory()
    
    # 3. Reforzar asociaciones cruzadas débiles
    weak_bridges = find_weak_cross_domain_bridges()
    strengthen(weak_bridges, weight=1.2)
    
    # 4. Archivar nodos sin uso en >30 días
    archive_if_unused(nodes, threshold_days=30)
```

---

## Fase 4: External Knowledge Ingestion (Próxima Semana)

### Pipeline SpiderBolt-like para Extracción Masiva
```
URLs externas → SpiderBolt scraper (500 threads) → 
  Categorización automática por paths →
  Extracto semántico →
  Axioma-Omega (como CONCEPT) →
  Omega-Cube (TensorNode con source_url) →
  Fabric Memory (con citation)
```

### Integración con Loopy Pattern
Crear loops reutilizables para:
- `knowledge_discovery_loop`: Buscar nuevo conocimiento → evaluar relevancia → ingresar si score > 0.7
- `memory_audit_loop`: Revisar grafo semanalmente → detectar gaps → sugerir ingestas
- `decision_trace_loop`: Trazar decisiones pasadas → encontrar precedentes similares

---

## Métricas de Éxito

| Métrica | Actual | Objetivo (30 días) |
|---------|--------|-------------------|
| Nodos Axioma-Omega | 21 | 200+ |
| Conceptos estructurados | 0 | 50+ |
| Asociaciones cruzadas | ~6 | 30+ |
| Uso de memoria (recalls/usages) | Bajo | >70% usage rate |
| Latencia query (MARP) | ~941ms | <500ms con predicción |

---

## Cron Jobs Nuevos a Crear

| Job | Frecuencia | Script | Descripción |
|-----|-----------|--------|-------------|
| omega-auto-indexer | Cada 15 min | `scripts/omega_auto_indexer.py` | Indexa conversaciones en Omega-Cube |
| omega-daily-evolution | Diario 2am | `scripts/omega_daily_evolution.py` | AutoResearch + patrones de uso |
| omega-autoresearch-weekly | Domingo 3am | `scripts/omega_autoresearch_weekly.py` | Búsqueda papers externos |
| omega-autopublisher | Domingo 4am | `scripts/omega_autopublisher.py` | Publica release si score > 0.45 |
| memory-maintenance-daily | Diario 6am | `scripts/memory_maintenance.py` | Maintenance localizado (stale, dupes) |

---

## Dependencias y Bloqueadores

### ✅ Sin bloqueadores críticos
- Scripts existentes están completos (solo faltan los de automatización)
- MCP servers funcionando (axioma, omega-cube, marp)
- GPU disponible (RTX 3090, ~5.8GB libre)

### ⚠️ Riesgos
- Scripts nuevos deben usar paths Windows nativos (no MSYS)
- Fabric memory tiene límite de chars — mantener entries compactos
- Cron jobs sin `deliver` no notifican al usuario en TUI

---

## Próximos Pasos Inmediatos

1. **HOY**: Crear `scripts/omega_auto_indexer.py` (el más urgente, el cron ya lo apunta)
2. **HOY**: Verificar que MARP routers estén corriendo (healthcheck manual)
3. **MAÑANA**: Crear los otros 3 scripts de automatización
4. **MAÑANA**: Implementar `unified_memory_save()` como hook central
5. **SEMANA 2**: Pipeline de external knowledge ingestion con SpiderBolt-like scraper
