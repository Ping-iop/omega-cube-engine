# Axioma-Omega — Arquitectura del Stack de Memoria

> **Fuente de verdad** sobre cómo está cableado el sistema. Cualquier sesión
> futura que modifique este stack debe leer esto primero y respetar las
> invariantes. Última actualización: 2026-08-09.

## Invariantes (romperlas reintroduce bugs ya pagados)

1. **STORE ÚNICO**: el motor consultable (MCP `omega-cube`, `OmegaCubeEngine`)
   lee y escribe `memory/omega_cube_memory.json`. TODO script que necesite
   nodos de Omega-Cube usa `scripts/omega_store.py` (`load_state`/`save_state`).
2. **`omega_cube/cube_state.json` NO EXISTE**: fue archivado en
   `memory/backups/cube_state.json.archived-*`. Ningún código debe crearlo,
   leerlo ni escribirlo. Si reaparece, hay un split-brain nuevo.
3. **Las firmas holográficas NO se persisten**: se recalculan en `load()`
   (son derivables de content+hierarchy). Ahorra ~90% del tamaño.
4. **node_id estables con sha256**, nunca con `hash()` de Python (aleatorio
   entre procesos por PYTHONHASHSEED → dedup rota).
5. **El indexer no ingiere archivos de estado del motor**: guarda en
   `STATE_FILES` y excluye `memory/backups/`.
6. **Los crons de memoria son `no_agent`** (Python puro, sin LLM, sin GPU).
   El worker LLM (puerto 8084) lo apaga el usuario para liberar VRAM — nada
   de infraestructura de memoria puede depender de él.
7. **Toda rotura debe ser ruidosa**: el guardián `axioma-memory-guardian`
   corre el selftest cada 6h. Silencio = sano; alerta = rotura. No agregar
   componentes que fallen en silencio.

## Mapa de componentes

```
┌──────────────────────────── CONSULTA ────────────────────────────┐
│  MCP axioma (axioma_mcp_server.py)                               │
│    → SessionContextEngine → memory/unified_memory.json           │
│    → tools: axioma_query, axioma_learn, axioma_telemetry,        │
│      axioma_mark_used, session_* ...                             │
│                                                                  │
│  MCP omega-cube (omega_cube/omega_cube_mcp_server.py)            │
│    → OmegaCubeEngine → memory/omega_cube_memory.json (STORE ÚNICO)│
│                                                                  │
│  axion_brief_enricher.py                                         │
│    → enriquece briefs de subagentes con contexto de Axioma       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────── CUBE MOVE + COLOR (2026-08-09) ──────┐
│  omega_cube/embeddings.py   — SemanticEmbedder (nomic 768d, cache│
│                                memory/semantic_embeddings.json)   │
│  omega_cube/cube_move.py    — Fase 2D (V@q) + 3D (giros) + gate  │
│                                τ=0.60 calibrado, ~46ms/50 nodos   │
│  omega_cube/color_chain.py  — hue axiomas, degradado λ^depth,    │
│                                mezcla (media circular), linaje    │
│  omega_cube/validity_gate.py— APPROVED/VETOED/FLAGGED por linaje │
│  Pruebas: docs/pruebas/FASE_{0,1,2,3}_2026-08-09.md              │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────── INGESTA ─────────────────────────────┐
│  scripts/omega_auto_indexer.py (cron 15 min, no_agent)           │
│    → sesiones + docs de repos → STORE ÚNICO vía engine_put()     │
└──────────────────────────────────────────────────────────────────┘

┌────────────────────────── MANTENIMIENTO ─────────────────────────┐
│  scripts/omega_store.py          ← acceso único al STORE ÚNICO   │
│  scripts/omega_daily_evolution.py (cron diario, no_agent)        │
│  scripts/omega_autoresearch_weekly.py (domingo 3am, no_agent)    │
│  scripts/omega_autopublisher.py   (domingo 4am, no_agent)        │
│  scripts/memory_maintenance.py    (localizado, arXiv:2606.24775) │
│  scripts/unified_memory.py        (hook decisión→3 sistemas)     │
│  scripts/omega_turbovec_bridge.py (capa vectorial, cron 7pm)     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────── VERIFICACIÓN ────────────────────────┐
│  scripts/axioma_selftest.py   ← 17 checks end-to-end             │
│  axioma_selftest_guard.py     ← cron no_agent cada 6h, silencioso│
│  marp_watchdog.py             ← cron no_agent cada 5min (8082)   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────── MARP ────────────────────────────────┐
│  Router Qwen3.5-0.8B — puerto 8082 (llama-server, auto-restart)  │
│  Worker Qwen 27B    — puerto 8084 (GESTIÓN MANUAL del usuario)   │
│  marp_benchmark_cron.py — cron 6h, no_agent                      │
└──────────────────────────────────────────────────────────────────┘
```

## Puertos (arquitectura fija, decisión 2026-08-06)

| Puerto | Rol | Gestión |
|--------|-----|---------|
| 8082 | Router MARP (Qwen0.8B) | Auto — watchdog reinicia si cae |
| 8084 | Worker grande (Qwen 27B) | **Manual** — usuario lo apaga para liberar VRAM |
| 8081 | ComfyUI | Manual |

## Telemetría recalls/usages

`memory/telemetry.json` — métrica de salud definida 2026-05-07:
- `recalls`: consultas a la memoria (automático en `query()`)
- `usages`: confirmaciones de integración (manual vía `mark_used()` /
  tool MCP `axioma_mark_used`)
- **ALERTA** si recalls>0 con usages=0 → memoria decorativa

## Historia de bugs (para no repetir)

| Fecha | Bug | Causa raíz |
|-------|-----|-----------|
| 2026-08-09 | Split-brain | Indexer + 5 scripts escribían a cube_state.json, que el motor nunca lee |
| 2026-08-09 | Corrupción 926MB | `load()` no idempotente: doble load (auto-init + explícito) duplicaba axiom_ids exponencialmente hasta truncar el JSON |
| 2026-08-09 | Crecimiento infinito | `hash()` aleatorio → dedup nunca funcionaba |
| 2026-08-09 | Auto-ingestión | Indexer leía sus propios dumps como "sesiones" |
| 2026-08-09 | Ranking saturado | guidance_scale=3.0 sumado saturaba scores a 1.0 |
| 2026-08-09 | Watchdog muerto | .bat ejecutado como Python → SyntaxError |
| 2026-08-09 | Crons "Connection error" | Modo agente dependía del worker 8084 (apagado por VRAM) |
| 2026-08-09 | Drift de scripts | Crons corrían COPIAS viejas en hermes/scripts/ → ahora wrappers delgados |

## Backups

`memory/backups/`:
- `cube_state.json.bak-20260809` (2.37 MB, pre-compactación)
- `omega_cube_memory.json.bak-20260809` (pre-firmas)
- `cube_state.json.archived-20260809-final` (estado final del legado)
