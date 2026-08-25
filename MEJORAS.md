# 📋 Plan de Mejoras — Axioma-Omega Protocol + Motor Axion-Cube/MARP

> **Creado:** 2026-08-25 · Basado en revisión completa de ambos repos (motor `.hermes` + protocolo `K:\`).
> **Alcance:** motor desplegado en `C:\Users\GPAMD\.hermes\axioma-omega-protocol\` + protocolo axiomático en `K:\Proyectos\DESARROLLO_APPS\Axioma-Omega_Protocol\` (GitHub `Ping-iop/Axioma-Omega_Protocol`).
> **Convención de seguimiento:** marcar `[x]` al completar cada casilla y añadir debajo una línea `✅ YYYY-MM-DD: <resultado real>` con evidencia (comando, archivo o métrica).

---

## P0 — Riesgos activos (antes de cualquier cosa nueva)

### P0.1 — El MCP de Omega-Cube sirve el motor viejo (v1) en vez del v2
El servidor activo en config Hermes (`omega_cube/omega_cube_mcp_server.py`) importa `OmegaCubeEngine` (v1). Las mejoras arXiv 2026 (TypedSchema/VirtualSet, resumen jerárquico H²MT, BoundaryController PAGE-RAG, detector de alucinación, índice HNSW O(log n)) viven en `engine_v2.py` sin estar cableadas.

**Compatibilidad verificada 2026-08-25:**
- ✅ Compatible tal cual: `add_node`, `associate`, `save`, `load`, `stats`, `gray_validator.*`, `diffusion.*`, `query(mode=hierarchical|holographic|diffusion|combined)`
- ⚠️ Memoria: v2 usa `omega_cube_memory_v2.json`; si no existe, **migra sola desde `omega_cube_memory.json`** (re-encoda firmas holográficas a semánticas). Sin pérdida de datos.
- ❌ Falta en v2: `query_multi_topic()` y `find_patterns()` (usados por 2 tools del MCP), y modos `annealing`/`tensor`.

Casillas:
- [x] Smoke test de v2 contra memoria actual (nodos/axioms correctos, 4 modos de query OK)
  - ✅ 2026-08-25: 76 nodos / 17 axiomas cargados (auto-migración desde v1 funcionando); 4 modos devuelven resultados con score 1.000.
- [x] Migrar import del MCP a `OmegaCubeEngineV2`
  - ✅ 2026-08-25: server reescrito completo sobre engine v2; banner de arranque movido a stderr (stdout es canal JSON-RPC).
- [x] Cubrir `omega_cube_multi_topic`: loop de `query()` por tema a nivel MCP (semántica equivalente, sin tocar el engine)
  - ✅ 2026-08-25: mejor opción elegida — porte directo de `query_multi_topic()` al engine v2 (misma semántica v1 vía `diffusion.sample_multi_topic`); verificado contra grafo real (HERMES: 2 hits 0.497 · SISTEMA: 2 hits 0.580).
- [x] Decidir `omega_cube_patterns`: portar lógica v1 (usa `annealer`+`pattern_emergence`, presentes en v2) o retirar la tool si no aporta uso real
  - ✅ 2026-08-25: DECISIÓN = portar. `_build_cubes`, `_pattern_energy` y `find_patterns` migrados tal cual a engine_v2; 12 patrones detectados en prueba real.
- [x] Actualizar modos válidos en docstring + mensaje de error de `omega_cube_query`
  - ✅ 2026-08-25: hierarchical (default), holographic, diffusion, combined; el mensaje de error lista los 4.
- [x] Verificación end-to-end: tools invocadas contra el server real y resultados correctas
  - ✅ 2026-08-25: handshake MCP REAL por stdio (`initialize` → `tools/list` 8/8 → `tools/call omega_cube_stats`): server responde como "omega-cube", motor v2, 76 nodos. Nota: reiniciar Hermes para que recargue el server.

### P0.2 — Dos servidores MCP con el mismo nombre (`CORREGIDO: no eran duplicados`)
`omega_cube/omega_cube_mcp_server.py` (motor Omega-Cube, el activo en Hermes) ≠ `omega_cube/marp/omega_cube_mcp_server.py` (server MARP worker/support :8084/:8091, independiente y legítimo). Solo comparten nombre → riesgo de confusión/edición equivocada.

- [x] Renombrar `marp/omega_cube_mcp_server.py` → `marp/marp_support_server.py`
  - ✅ 2026-08-25: renombrado; 5 scripts actualizados (check_marp.py, check_marp_router.py, marp_monitor.py ×2, marp_router_cron.py); 0 referencias viejas restantes.

### P0.3 — Blindaje de integridad del store
Antecedente: corrupción de 926 MB (2026-08-09) por doble `load()`. Hoy `memory/omega_cube_memory.json` está sano (124 KB) pero sigue siendo escritura directa sin protección.
- [x] Guardado atómico: escribir a temp + `os.replace()` (en engine v2 `save()`)
  - ✅ 2026-08-25: implementado (tmp → fsync → validación JSON del tmp → os.replace). Sin .tmp residual tras save normal.
- [x] Checksum/validación JSON tras cada save; log de error si falla
  - ✅ 2026-08-25: implementado como validación JSON del tmp DENTRO de save() (falla ruidoso antes del replace; checksum separado innecesario con esta barrera).
- [x] Corregir duplicación de axiomas en doble `load()` (bug de v1 **presente también en v2**: `self.axioms.append` sin guard)
  - ✅ 2026-08-25: guard por set de IDs en load(); test triple load = 1/1/1 axiomas.
- [x] Guard del cron (`axioma-memory-guardian`, cada 6 h): fallar ruidoso si el JSON no parsea — ✅ 2026-08-25: `scripts/axioma_selftest.py` hace json.load del store (l.159/164) → crash → guard imprime alerta `[FAIL]` (verificado leyendo ambos scripts)
- [x] Test: matar proceso a mitad de `save()` → archivo anterior intacto
  - ✅ 2026-08-25: crash simulado justo entre tmp y replace: store previo byte-a-byte intacto; save posterior se recupera solo (2/2 nodos).

### P0.4 — Ejecutar F8: la prueba reina a escala (≥1000 nodos, ≥100 preguntas)
La tesis central (*navegación jerárquica + cube_move > búsqueda plana por coseno*, F1 ≥ +5 puntos) nunca se midió. Grafo vivo: ~71 nodos.

**Ejecutada 2026-08-25** (dim=2048, 1028 nodos, 2844 aristas, 120 preguntas, seed=42).
Veredicto: **MATIZADA (+1.1 pp F1)** — jerárquico gana en las 3 métricas (MRR +12 pp) pero no alcanza el umbral de +5 pp. **cube_move REFUTADO**: su fase 2D replica al plano (0.231≈0.233) y su fase 3D resta −7.5 pp. Además: encoder saturado a 256d (hallazgo crítico) y 2 defectos reales corregidos en `cube_move.py`. Detalle completo: `docs/pruebas/F8_INFORME_2026-08-25.md`.

- [x] Generar población sintética hasta ≥1000 nodos → 1028 nodos deterministas (seed=42), `scripts/f8_battle.py`
- [x] Suite de ≥100 preguntas etiquetadas → 120 preguntas × 3 dificultades
- [x] Duelo F8: navegación jerárquica vs coseno plano → tabla Precision/Recall/F1 en `docs/pruebas/` (`F8_2026-08-25_d2048.md`, + control d256 y ablación)
- [x] Decisión documentada: tesis confirmada / refutada / matizada → **matizada**; cube_move refutado como mecanismo; accionables en el informe §6

**Nuevas acciones derivadas (candidatas P1):**
- [x] Subir `holographic_dim` de producción 256 → 2048 (el espacio saturado estrangula toda la recuperación) — **HECHA 2026-08-25**: backup `.bak-20260825-dim256`; fix previo en `engine.py::load()` (dim aplicada ANTES de recodificar firmas + recalculo SIEMPRE, antes solo si estaban vacías); verificado: load con dim 2048, 76 firmas {2048}, roundtrip save→load idempotente, query real OK; indexer pausado/reactivado durante la migración.
- [x] Desactivar/rediseñar fase 3D de cube_move (evidencia en contra; pesos propios medidos) — **HECHA 2026-08-25**: default `expand_3d=False` en `cube_move.py::cube_move()` (comentario con evidencia F8; disponible vía flag explícito para experimentos); scripts F8 pasan argumentos explícitos → reproducibilidad intacta.

### P0.5 — Infraestructura de crons degradada (casa donde vive el sistema)
Diagnóstico 2026-08-25: los 3 jobs Harness usaban el modelo LOCAL `qwen3.8-27b-abliterated` (llama_cpp) → timeouts 180 s en loop agéntico multi-llamada (Evolution ×13, Critic ×4) y 503 "Loading model" al despertar el modelo del disco (Gardener). `vigilar-skillui`: ejecuta OK pero entrega rota (`deliver=origin,all` con `origin: null` → nada resoluble).
- [x] Migrar Harness Evolution/Critic/Gardener a `stealth/ox-alpha` (openrouter) — 2026-08-25, resuelve la CAUSA (latencia/frialdad del modelo local), mejor que re-agendar horarios contra disponibilidad de VRAM
- [x] Reparar target de entrega de `vigilar-skillui` — `deliver=bot-chat` (2026-08-25); se conserva su modelo local porque ejecuta OK
- [x] Timeout/retry de Harness: sin cambio de configuración — la causa raíz era el proveedor local, no el umbral; la migración de modelo elimina el problema
- [ ] 7 días consecutivos sin fallos de estos 4 jobs (ventana iniciada 2026-08-25; canary: corrida manual del Gardener disparada hoy)

---

## P1 — Consolidación (deuda que ya cobra intereses)

### P1.6 — MARP: 4–5 clasificadores coexistiendo
`router.py` (grafo nativo) · `router_service.py` (SmolLM2) · `gpu_router.py` (Qwen 0.8B :8082) · `qwen_classifier.py` · pre-filtro+GBNF de `intelligent_pipeline.py`. Cada uno con taxonomía propia.
- [x] Elegir UNO por defecto con cadena de degradación explícita (ej.: keywords → grafo nativo → LLM pequeño) — **HECHA 2026-08-25**: `gpu_router.py::QwenGPURouter` es EL clasificador (keywords ES/EN → GPU+GBNF → general conf 0.05). Logging diario JSONL añadido (`~/.hermes/logs/marp_router/marp_YYYYMMDD.jsonl`) + `marp/log_analyzer.py` extraído + wrapper `omega_cube/marp_analyze_today.py`. Verificado en vivo: tier GPU 183–628 ms, 3/3 precisión.
- [x] Eliminar el resto (sin caminos muertos) — **HECHA 2026-08-25**: borrados `qwen_classifier.py` (0 importers), `intelligent_pipeline.py`, `router_service.py`; backups `*.bak-20260825-p16`. Consumidor único real: `axion_brief_enricher.py` (sin cambios necesarios).

### P1.7 — Taxonomía de dominios definida en 3 sitios
GBNF hardcodeada del pipeline (12 dominios) · `STANDARD_DOMAINS` de `router.py` · auto-descubrimiento desde el grafo.
- [x] Unificar en `marp/protocol.py` como única fuente — **HECHA 2026-08-25**: `STANDARD_DOMAINS` (10 ricos) + `EXTRA_DOMAINS` (general, memory) viven en protocol.py; router.py importa el alias, gpu_router.py deriva `VALID_DOMAINS`.
- [x] Gramática GBNF generada desde ahí (dominio nuevo ≠ tocar código en 3 lugares) — **HECHA 2026-08-25**: `gbnf_domain_grammar()` genera la gramática y `domains_prompt_line()` interpola el prompt; byte-idéntica a la hardcodeada (verificado). Dominios descubiertos por el grafo pueden fusionarse al dict y todo se regenera.

### P1.8 — Puertos y rutas hardcodeados
`127.0.0.1:8082`/`:8084` repartidos por scripts.
- [x] Extraer a `config.yaml` único del despliegue — **HECHA 2026-08-25** (variante env vars, más simple que config.yaml): `MARP_ROUTER_URL` / `MARP_WORKER_PORT` en gpu_router.py, model_orchestrator.py y marp_support_server.py (defaults intactos 8082/8084). Eliminados 19 archivos muertos o con convención de puertos invertida (monitores del 8084 como "router", benchmarks RURL/WURL cruzados, tests de modelos desmontados): backups `*.bak-20260825-p18`. Verificado: classify OK con override activo.

### P1.9 — Embeddings con fallback silencioso
Si ollama/nomic no responde se usaban vectores aleatorios sin marcarlo.
- [x] Etiquetar nodos con embedding degradado en metadatos + reporte del guard — **HECHA 2026-08-25**: flag runtime `_degraded_embedding` (no persiste, to_dict explícito), `GateVerdict.degraded_embeddings`, nota en reason y downgrade APPROVED→FLAGGED; verificado E2E con τ=0: FLAGGED "⚠ 20/20 embeddings degradados". Limpieza de flags al recuperar ollama incluida
- [x] Persistir modelo generador por vector (mezclar nomic-768d con otros dims corrompe scores coseno) — **HECHA 2026-08-25**: campo `gen` en cada entrada del cache (`semantic_embeddings.json`) + `query_compatible_with_cache()` que bloquea la query si el cache es de otro generador/dim (caller cae al fallback holográfico consistente); unit test verifica ambas direcciones (scripts/p19_unit.py)

### P1.10 — Orfanos a 0 + invariante permanente
Quedan 2 nodos sin cadena de tono (41→2).
- [x] Cerrar los 2 restantes — **HECHA 2026-08-25**: en realidad quedaban 7 huérfanos (el indexador activo acumuló nuevos). Migración `scripts/p110_migrate.py`: 2 basura eliminados (`semantic_embeddings` cache ingerido como nodo + turno de conversación efímero), 3 herramientas `HERRAMIENTAS.*` re-hogadas bajo `DEV.TOOLS.*`, repintado completo con `propagate()`; backup `omega_cube_memory.json.bak-20260825-p110`. De paso: FIX ruta de repos del indexador (`~/Documents/GEMINI` → `K:/Proyectos/DESARROLLO_APPS`, los 9 repos volvieron a indexar)
- [x] Test/invariante: todo CONCEPT/INSTANCE debe tener `hue_origin` verificable; si no, el indexador rechaza — **HECHA 2026-08-25**: invariante implementado como re-hogado automático en `engine_put()` (dominio sin axioma → `DEV.TOOLS.*`) + `propagate()` obligatorio antes de cada `save()` del indexador. Canary real: corrida completa con ingesta de repos → 59 coloreados / 0 huérfanos, store persistido verificado

### P1.11 — El motor no tiene respaldo remoto
Protocolo en GitHub ✓; motor (cube_move, cadena de color, gate, benchmarks) solo en `.hermes`. Último commit menciona "2 repos fuentes de verdad" pero `omega-cube-engine` no está localizado.
- [ ] Verificar si existe `omega-cube-engine` en remoto (GitHub)
- [ ] Si no: crear repo y empujar (excluyendo `memory/` personal)

---

## P2 — Protocolo (repo `K:`): ejecutar su propio Roadmap

`Roadmap/Roadmap.md` ya prioriza 16 tareas, todas ⬜. Orden recomendado:

### P2.A — Tests primero (fase 2 del roadmap)
Hoy ~9% cobertura (506 líneas de test vs ~5.700 de código).
- [ ] Tests unitarios de `domain_reasoner` (veto APPROVED/VETOED/FLAGGED) — objetivo >80%
- [ ] Tests unitarios de `axiom_registry` (condiciones de contorno)

### P2.B — Empaquetado (fase 3)
- [ ] `pyproject.toml` + CLI `axioma query "..." --domain PHYSICS`

### P2.C — Higiene rápida (fase 1)
- [ ] `requirements.txt`: descomentar lo mínimo real
- [ ] Commitear o borrar `IDEA.md` (30 bytes sueltos)
- [ ] Mover `Documentation/` → `docs/teoria/`

### P2.D — Esquema compartido protocolo ↔ motor (NO contemplado en su roadmap)
Dos definiciones paralelas de "verdad": `Axiom/AxiomLayer/ValidationVerdict` (protocolo) vs AXIOM/CONCEPT/INSTANCE + gate con veredictos string (motor).
- [ ] Extraer dataclasses comunes (axioma, veredicto, capas de certeza) a módulo mínimo compartido, **o** documentar mapeo oficial capas ATOMIC/DOMAIN/SITUATIONAL/CREATIVE ↔ depth/saturación λ^depth

### P2.E — Sincronizar biblioteca de axiomas ↔ grafo vivo
`standard_library.py`: 20 axiomas en 13 dominios · grafo: 17.
- [ ] Comando `sync_axioms()`: cada axioma de la librería existe como nodo AXIOM con tono asignado

---

## P3 — Producto e innovación (después del veredicto de F8)

### P3.12 — Decidir F7 (binding holográfico) por datos
Convolución circular para multi-cadena, postergada "si F5 se queda corto". Criterio ya escrito: similitud coseno >0.95 al deshacer enganche.
- [ ] Micro-benchmark una vez + decisión archivada (lleva meses pendiente sin costar medirla)

### P3.13 — Tablero de telemetría MARP
Logs diarios existen (`~/.hermes/logs/marp_router`); scheduler v2 trackea `prefetch_hits/misses` pero nada los lee.
- [ ] Vista agregada: precisión real del router, tasa de aciertos prefetch, ahorro real vs estimado
- [ ] Calibrar τ y umbrales con datos (como exige el plan)

### P3.14 — Visualizar cadena de color en dashboard existente
`Axioma_Base/dashboard/app.py` + diseño tono+saturación inherentemente visual.
- [ ] Grafo coloreado por tono con saturación por profundidad

### P3.15 — Índice teoría ↔ prueba
14 docs teóricos en `Documentation/` + 5 baterías en `docs/pruebas/`.
- [ ] Índice que mapee afirmación teórica → prueba que la respalda (o "sin probar")

### P3.16 — Poblar axiomas antes de leer métricas del gate
17 axiomas para 13 dominios → ensamblados caen FLAGGED por falta de cadena, no por malos.
- [ ] Continuar población 71→más nodos antes de interpretar tasas APPROVED/FLAGGED

---

## Lo que NO se hará (anti-sobre-ingeniería)

- ❌ Migrar el store a base de datos (71→1000 nodos caben en RAM; el problema es idempotencia+atomicidad, P0.3)
- ❌ Optimizar latencia de cube_move (46.6 ms < objetivo 100 ms) hasta que F8 diga si la tesis vale
- ❌ Unificar los dos repos en uno (separación librería/despliegue es sana; falta esquema compartido P2.D y respaldo P1.11)

---

## Registro de ejecución

| Fecha | Ítem | Resultado |
|-------|------|-----------|
| 2026-08-25 | Revisión completa | Listado creado; P0.2 corregido (no duplicados); compatibilidad v1↔v2 mapeada |
| 2026-08-25 | P0.1 completo | MCP sobre engine v2 (handshake stdio OK, 8 tools); multi_topic+patterns portados a v2 |
| 2026-08-25 | P0.2 completo | Renombre marp_support_server.py + 5 refs actualizadas |
| 2026-08-25 | P0.3 casi completo | Save atómico + anti-duplicación + test de crash OK; falta guard del cron |
