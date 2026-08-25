# PLAN 2026-08-09 — Cube Move + Cadena de Color (Matrioshka)

> **Propósito de este documento**: si la sesión se hace larga o se abre una nueva,
> este plan contiene TODO el contexto necesario para continuar sin perder el hilo.
> Estado: **FASES 0–3 COMPLETADAS (2026-08-09)** — ver docs/pruebas/FASE_*.md.
> Fase 4 pendiente. La sesión del 2026-08-09 fue diseño conceptual + ejecución.

## RESUMEN DE EJECUCIÓN (2026-08-09)

**Hallazgo crítico pre-ejecución:** el store único estaba CORRUPTO (926MB, JSON
truncado) por `load()` no idempotente. Recuperado (nodes dict intacto), bug
causal fijado en 3 sitios, selftest 17/17. Detalle: docs/pruebas/FASE_0_*.md.

| Fase | Estado | Evidencia |
|---|---|---|
| 0 split-brain | ✅ PASS 0.1–0.3 | nodo test visible vía MCP fresco, +1 exacto, indexer idempotente |
| 1 cube_move | ✅ PASS 1.1–1.5 | τ=0.60 calibrado con datos; giros 3D en ambas direcciones; mediana 46.6ms |
| 2 color_chain | ✅ PASS 2.1–2.5 | 5 axiomas con hue espaciado 72°; degradado λ^depth exacto; mezcla detectada (caso geométrico + disperso) |
| 3 gate | ✅ PASS 3.1–3.3 | APPROVED/FLAGGED con ruta de linaje real; gray-scale intacto |
| 4 integración | 🟡 EN PROGRESO | 4A población: 50→71 nodos, 5→17 axiomas, 8 puentes, indexer 1→9 repos, orfanos 41→2. Orquestador + MARP integrados. A/B real: sin protocolo 0/7 hechos vs con protocolo 7.5/7 (docs/pruebas/FASE_4_AB_2026-08-10.md). FALTA: F7 binding holográfico y F8 a escala (≥100 preguntas/≥1000 nodos) |

**Decisiones de implementación divergentes del plan (documentadas con datos):**
1. τ = 0.60 (el plan sugería 0.3 inicial → calibración lo descartó: ruido).
2. F3 refinado: tonos de axioma espaciados uniformemente (sha256 mod 360 los
   agrupaba en 39° → mezcla geométricamente ambigua).
3. Capa vectorial: nomic-embed-text 768d vía ollama (los embeddings del TurboVec
   bridge eran aleatorios — fallback). Cache en memory/semantic_embeddings.json.
4. F2 corregida: giros 3D compiten con fichas 2D por el hueco (top-k de la unión);
   la versión "presupuesto sobrante" dejaba 0 slots cuando 2D llenaba el hueco.
5. El grafo real tiene 50 nodos y solo 5 axiomas (2 dominios cubiertos) → 41
   nodos orfanos. La Fase 4 debe poblar axiomas por dominio antes de exigir
   tasas altas de APPROVED.

**Preguntas abiertas del plan (sección 5) — estado:**
- Curva de saturación exponencial λ=0.7: implementada así, verificada exacta (F4).
- Fase 4 con delegate_task real vs simulado: sigue abierta para el usuario.

---

## 1. CONTEXTO: qué se diseñó y por qué

### 1.1 Punto de partida (conversación 2026-08-09)

Se discutieron agentes autónomos de IA: cómo se construyen (orquestador + workers),
cómo funcionan los subagentes (sesiones LLM frescas, aisladas, diferenciadas por
6 ejes: objetivo, contexto, rol, herramientas, modelo, restricciones), y cómo MoE
cambia la ecuación económica del multi-agente.

El punto clave que conecta todo con Omega: **el cuello de botella del resumen**.
En orquestador-worker, cada frontera entre agentes es una compresión donde muere
el detalle (un worker lee 40K tokens y devuelve 300). Axioma-Omega puede resolver
esto de raíz: los workers escriben sus hallazgos en el grafo y el orquestador
compone la respuesta cargando solo los pedazos relevantes.

### 1.2 Las dos metáforas que hay que implementar

**A) Cubo Move (puzzle 2D + giro 3D):**
- Fase 2D: dado un query, TODOS los nodos "se mueven" simultáneamente hacia el
  query (operación vectorial sobre los embeddings de TurboVec, no combinatoria).
  Las K fichas más relevantes "caen en el hueco" (el hueco = presupuesto de contexto).
- Fase 3D: seguir asociaciones cruzadas entre dominios ("giros de cara del Rubik")
  para anidar conceptos de temas diferentes en la respuesta.
- Validación: el ensamblado debe pasar la validación axiomática (APPROVED/VETOED).
- NOTA del usuario: el Rubik es 30x30 pero solo giran las caras externas;
  el interior no gira (gradiente de estabilidad: núcleo rígido, periferia flexible).

**B) Matrioshka + cadena de color (degradado, NO escala de grises):**
- Cubos anidados: centro = verdad inmutable (IMMUTABLE), hacia afuera = capas
  de mayor flexibilidad. Esto YA EXISTE como las 4 capas Bayesianas del
  ProbabilisticHierarchyEngine (IMMUTABLE → PROBABILISTIC → EMERGENT → FLUID).
- Cada axioma tiene un TONO (hue) único. La información derivada degrada la
  SATURACIÓN del mismo tono hacia afuera:
  - AXIOMA: tono puro, saturación 100% (certeza 1.0)
  - CONCEPTO: mismo tono, saturación ~70%
  - INSTANCIA: mismo tono, saturación ~40%
  - EMERGENTE: mismo tono, saturación ~10%
- El TONO = identidad de la cadena (de qué axioma viene).
  La SATURACIÓN = profundidad derivacional / certeza.
  Los dos ejes son ortogonales.
- La mezcla de cadenas es DETECTABLE geométricamente: si un nodo cae en un tono
  intermedio entre dos axiomas, es un híbrido y se sabe de cuáles dos viene.
- Esto es la extensión natural del componente Gray-Scale Validation (monocromo)
  de Omega-Cube: el gris dice "qué tan verdadero", el tono dice "verdadero
  respecto a QUÉ axioma".
- El usuario corrigió explícitamente: son DEGRADADOS DE COLOR, no de gris.
  No llamarlo "cadena de grises".

### 1.3 Decisiones de diseño acordadas

1. El movimiento de cubos es vectorizado continuo (producto punto batcheado),
   NO resolución discreta de puzzle (NP-hard a escala).
2. El "hueco" no restringe el movimiento (los vectores se superponen sin chocar),
   solo restringe la carga final (solo K conceptos entran en la respuesta).
3. Asignar colores es trivial: los node_ids ya son sha256 (espacio 2^256).
   El color es una dirección de origen, no decoración.
4. Para mezcla simple: detección geométrica de tono. Para mezcla compleja
   (varias cadenas sin perder provenance): binding holográfico ⊗ (ya existe
   el mecanismo en Holographic Encoding, circular convolution de Plate 2003).
5. Verificación mecánica de validez: hacer unbind del color y comprobar que el
   axioma origen sigue en el registro. Si no → FLAGGED.

---

## 2. ESTADO ACTUAL DEL SISTEMA (verificado 2026-08-09)

- Grafo vivo: **28 nodos** tras compactación (2 axiomas, 3 conceptos, 17 instancias,
  27 sesiones indexadas en unified_memory; cube reducido de 1008→28 nodos).
  Tamaño perfecto para prototipar en RAM sin optimizaciones.
- **BUG CRÍTICO PENDIENTE (split-brain)**: el indexer escribe a
  `omega_cube/cube_state.json` pero el MCP consultable carga
  `memory/omega_cube_memory.json`. El contenido fresco NUNCA llega a la memoria
  consultable. Fix: que el indexer use `OmegaCubeEngine.add_node()` + `save()`
  sobre el archivo correcto.
- Backups de la compactación en `memory/backups/*.bak-20260809`.
- TurboVec: índice en `~/AppData/Local/hermes/scripts/omega_vector_index/`,
  bridge en `~/AppData/Local/hermes/scripts/omega_turbovec_bridge.py`.
- Path del proyecto: `C:\Users\GPAMD\.hermes\axioma-omega-protocol`
  (en bash: `/c/Users/GPAMD/.hermes/axioma-omega-protocol`).
- MCP tools disponibles: `mcp__axioma__axioma_query`, `axioma_stats`,
  `axioma_learn`, `axioma_associate`, `axioma_tree`, `axioma_hierarchy`, etc.

---

## 3. FASES DE EJECUCIÓN

### FASE 0 — Fix del split-brain (prerrequisito, ~1 sesión)
- [ ] Confirmar el bug leyendo ambos archivos y el indexer
      (`scripts/omega_auto_indexer.py` o equivalente del cron)
- [ ] Unificar: el indexer debe escribir vía `OmegaCubeEngine.add_node()` + `save()`
      al archivo que carga el MCP (`memory/omega_cube_memory.json`)
- [ ] Verificar con diff real: escribir un nodo de prueba, confirmar que aparece
      en `axioma_stats` y en `axioma_query`. SIN diff = no está arreglado.
- [ ] Revisar cron jobs activos (omega-auto-indexer cada 15 min) y confirmar
      que no vuelvan a auto-ingerir los dumps del propio motor.

### FASE 1 — `cube_move()` mínimo (la mecánica 2D+3D)
- [ ] Crear `omega_cube/cube_move.py` con:
  ```
  cube_move(query, k=20):
    1. Fase 2D: scores = dot(embeddings_nodos, embed(query)); top_k → seleccionados
       (usar TurboVec si está sincronizado, si no, embeddings directos)
    2. Fase 3D: para cada seleccionado, seguir asociaciones cruzadas (cross-domain)
       y añadir sus nodos (límite: 1 nivel de profundidad)
    3. Ensamblar jerarquía y validar contra el registro de axiomas
    4. Retornar: nodos fase 2D, nodos fase 3D, veredicto, costo
  ```
- [ ] NO optimizar para escala todavía. 28 nodos caben en RAM.
- [ ] Medición verificable: query "capital de Francia" → listar qué nodos
      selecciona la fase 2D, qué trae la fase 3D, y si pasa validación.
      Antes/después con diff.

### FASE 2 — Cadena de color (degradado por tono)
- [ ] Crear `omega_cube/color_chain.py`:
  - `assign_hue(axiom_node)`: tono único por axioma (del registro sha256 del
    node_id, mapeado a 0-360°)
  - `propagate()`: recorrer hijos del axioma, asignar mismo tono con saturación
    decreciente por nivel (1.0 → 0.7 → 0.4 → 0.1)
  - `detect_mixture(node)`: si el tono resultante cae entre dos tonos de axioma,
    reportar como híbrido e identificar los padres
  - `verify_lineage(node)`: desandar hasta el axioma origen; si no existe en el
    registro → FLAGGED
- [ ] Guardar tono+saturación como campos del nodo (no romper la carga existente:
    `load()` debe tolerar nodos sin color).
- [ ] Medición: (1) una query sigue solo su cadena de tono; (2) un nodo creado
      desde dos cadenas es detectado como mezcla de los dos tonos correctos.

### FASE 3 — Integración con el protocolo (gate de validez)
- [ ] El veredicto APPROVED/VETOED/FLAGGED usa el linaje de color:
      una respuesta es APPROVED si todos sus nodos tienen linaje verificable
      hasta un axioma del registro.
- [ ] Conexión con Gray-Scale Validation existente (extender, no reemplazar).

### FASE 4 — (Futuro, no empezar sin evidencia de Fases 1-2)
- Integración con agentes: orquestador que consulta Axioma antes de lanzar
  subagentes (brief enriquecido con axiomas del dominio) y valida su salida
  por linaje de color.
- Binding holográfico ⊗ para mezcla multi-cadena sin pérdida.

---

## 4. REGLAS DE TRABAJO (no negociables)

1. **Verificación con diff real**: toda "mejora" se mide con output antes/después.
   Prohibido reportar porcentajes de mejora sin diff verificable.
2. **Capas**: mínimo viable primero; cada fase sobre la anterior funcionando.
3. **Sin caminos muertos**: si algo se refactoriza, se elimina lo obsoleto.
4. **Python en el venv de Hermes**: `$HOME/AppData/Local/hermes/hermes-agent/venv/`
5. **Paths**: en scripts Python usar `r'C:\Users\GPAMD\...'` (NO estilo MSYS
   `/c/Users/...` — no resuelve imports). En bash usar `/c/Users/GPAMD/...`.
6. **Persistir**: siempre `engine.save()` tras modificar el grafo.
7. Los node_ids son sha256 (estables entre procesos). NO usar `hash()`.
8. **PRUEBAS REALES CON DATOS REALES (obligatorio)**: cada fase termina con una
   batería de pruebas ejecutadas contra el grafo vivo, mostrando la información
   real de los resultados en la conversación (no descripciones de lo que "debería"
   pasar). El formato mínimo por prueba:
   ```
   PRUEBA: <nombre>
   INPUT:  <query/datos exactos usados>
   OUTPUT REAL: <lo que devolvió el sistema, copiado literal>
   ESPERADO: <qué se esperaba>
   VEREDICTO: PASS / FAIL + por qué
   ```
   - Si una prueba FAIL, se corrige y se re-ejecuta hasta PASS antes de avanzar
     de fase. No se reporta "funciona" sin el output real de la prueba.
   - Los resultados numéricos (scores, tiempos, nodos seleccionados) se muestran
     en tablas con los valores reales, no aproximados.
   - Cada fase guarda sus resultados de prueba en
     `docs/pruebas/FASE_<n>_<fecha>.md` para tener histórico y comparar
     regresiones entre sesiones.

---

## 4b. BATERÍAS DE PRUEBA POR FASE (qué medir exactamente)

### Fase 0 — split-brain
| # | Prueba | Criterio de PASS |
|---|---|---|
| 0.1 | Escribir nodo de prueba vía indexer → leer con `axioma_query` | El nodo aparece en el output real del query |
| 0.2 | `axioma_stats` antes/después | El conteo sube exactamente +1 |
| 0.3 | Ejecutar el cron manualmente → verificar que no re-ingiere dumps | El conteo de nodos basura no crece |

### Fase 1 — cube_move
| # | Prueba | Criterio de PASS |
|---|---|---|
| 1.1 | Query con tema que existe en el grafo | Fase 2D devuelve nodos relevantes reales (mostrar scores) |
| 1.2 | Query con tema que NO existe | Fase 2D devuelve pocos/0 nodos, sin inventar |
| 1.3 | Query que cruza dos dominios asociados | Fase 3D trae nodos del segundo dominio (mostrar cuáles) |
| 1.4 | Medir tiempo total con 28 nodos | Mostrar ms reales; objetivo <100ms |
| 1.5 | Baseline vs cube_move con la MISMA query | Diff de resultados: qué trae cada método, cuál es más relevante y por qué |

### Fase 2 — cadena de color
| # | Prueba | Criterio de PASS |
|---|---|---|
| 2.1 | Asignar tono a los 2 axiomas existentes | Mostrar tono (°) asignado a cada uno |
| 2.2 | Propagar degradado hacia abajo | Mostrar tono+saturación REAL de cada nodo hijo |
| 2.3 | Query dentro de una cadena | Solo devuelve nodos del mismo tono (mostrar lista) |
| 2.4 | Crear nodo híbrido de dos cadenas | detect_mixture() lo identifica como mezcla de los 2 tonos correctos |
| 2.5 | verify_lineage() sobre nodo derivado | Desanda hasta el axioma origen real; sobre nodo huérfano → FLAGGED |

### Fase 3 — integración con protocolo
| # | Prueba | Criterio de PASS |
|---|---|---|
| 3.1 | Respuesta con linaje completo | Veredicto APPROVED + mostrar cadena de axiomas |
| 3.2 | Respuesta con linaje roto | Veredicto FLAGGED + mostrar dónde se rompe |
| 3.3 | Gray-Scale Validation existente sigue funcionando | Mismos resultados que antes (sin regresión) |

---

## 4c. FORMULAS MATEMÁTICAS DE LAS IDEAS (implementación exacta)

> El usuario diseña con metáforas; esta sección es la traducción matemática
> exacta de cada idea. Implementar ESTAS fórmulas, no interpretaciones.
> Cada fórmula lleva su criterio de prueba.

### Notación general

- `V` = conjunto de nodos del grafo (28 hoy)
- `v_i ∈ R^d` = embedding del nodo i (d=1536 con TurboVec)
- `q ∈ R^d` = embedding de la query
- `A` = conjunto de asociaciones (aristas): `(i, j, peso)`
- `depth(n)` = profundidad derivacional del nodo n desde su axioma origen
- `k` = presupuesto de contexto ("el hueco" del tablero)

### F1. cube_move — Fase 2D (el tablero se ordena)

**Normalización** (obligatoria, o el score es basura):
```
v̂ = v / ||v||          donde ||v|| = sqrt(Σ v[m]²)
```

**Score de relevancia** — todos los cubos se mueven a la vez:
```
s_i = v̂_i · q̂ = Σ_m v̂_i[m] · q̂[m] = cos(θ_i)     ∈ [-1, 1]
```
En código: UNA operación matricial `scores = V_norm @ q_norm` (miles de nodos, un paso).

**Selección** (las fichas que caen en el hueco):
```
S_2D = { i ∈ V : s_i ≥ τ }   ordenado por s_i descendente, máx k nodos
τ = umbral (probar τ = 0.3 inicial; medir y ajustar con datos reales)
```
⚠️ PRUEBA: si τ es muy bajo entra ruido, si es muy alto no entra nada.
El valor de τ se calibra con la batería 1.1/1.2, NO se fija por intuición.

### F2. cube_move — Fase 3D (giros de cara / cruce de dominios)

**Expansión lateral** — un solo nivel de giro (profundidad 1, no recursivo):
```
S_3D = S_2D ∪ { j : ∃ i ∈ S_2D tal que (i,j) ∈ A ∧ dominio(j) ≠ dominio(i) }
```
Regla: `|S_3D| ≤ k` total. Si la expansión desborda, ordenar por
`peso(i,j) · s_i` y cortar en k. El giro se pondera por la fuerza de la
asociación multiplicada por la relevancia del nodo origen.

**Validación del ensamblado:**
```
veredicto = VALIDATE(build_hierarchy(S_3D))   → APPROVED | VETOED | FLAGGED
```

### F3. Cadena de color — asignación de tono

**Del hash al tono** (determinista y estable):
```
H(nodo) = int(sha256(node_id)[:8], 16) mod 360     → tono en grados [0, 360)
```
Cada AXIOMA recibe su tono H. Los derivados NUNCA inventan tono: lo heredan.

### F4. Cadena de color — degradado por saturación

**Decaimiento exponencial con la profundidad:**
```
sat(n) = λ ^ depth(n)      con λ = 0.7
```
```
AXIOMA:      depth 0 → sat = 1.000
CONCEPTO:    depth 1 → sat = 0.700
INSTANCIA:   depth 2 → sat = 0.490
EMERGENTE:   depth 3 → sat = 0.343
```
(λ se calibra con datos; 0.7 es el inicial. PRUEBA 2.2 debe mostrar estos
valores reales por nodo.)

### F5. Cadena de color — el truco que resuelve el círculo

⚠️ PROBLEMA: el tono es circular. Promedio simple de 350° y 10° da 180°
(el lado OPUESTO) — error grave si se promedian grados como números.

**SOLUCIÓN — representar el tono como vector unitario:**
```
u(h) = (cos(h·π/180), sin(h·π/180))     ∈ R²
```

**Media circular correcta** (para detectar de qué tono viene un nodo):
```
R = Σ_j u(h_j)                            (suma de vectores)
h_media = atan2(R_y, R_x) · 180/π         (ángulo resultante)
concentración = ||R|| / n                  ∈ [0, 1]
```
- `concentración ≈ 1` → todos los aportes son del mismo tono → cadena pura
- `concentración baja` (< 0.5, calibrar) → MEZCLA detectada

**Detección de mezcla contra axiomas conocidos:**
```
Para cada axioma a con tono h_a:
   proyección_a = R · u(h_a)               (cuánto apunta hacia ese axioma)
Los 2 axiomas con mayor proyección positiva = padres de la mezcla
```
⚠️ PRUEBA 2.4: un nodo creado desde axiomas con tonos h1, h2 debe dar
proyección máxima exactamente sobre u(h1) y u(h2), no sobre otros.

### F6. Verificación de linaje (deshacer el camino)

**Regla de aprobación:**
```
APPROVED  ⟺  depth(n) ≤ D_max  ∧  axioma_origen(n) ∈ RegistroAxiomas  ∧  sat(n) ≥ sat_min
FLAGGED   en caso contrario
```
Valores iniciales: `D_max = 4`, `sat_min = 0.24` (≈ λ³). Calibrar con datos.

**Linaje se guarda como campo del nodo** (cadena de ids, no solo el padre):
```
lineage(n) = [axiom_id, ..., parent_id, n]     (ruta completa al centro)
```
Así `verify_lineage` es O(1) en lookup + verificación de que cada id existe.

### F7. Binding holográfico (solo si la mezcla supera a F5 — Fase 4)

**Bind** (engancha color a contenido) vía convolución circular:
```
bind(a, b) = IDFT( DFT(a) ⊙ DFT(b) )
```
**Unbind** (recupera el contenido):
```
unbind(c, b) = bind(c, reverse(b))    donde reverse(b)[m] = b[-m mod d]
```
Propiedad a verificar en prueba: `unbind(bind(a,b), b) ≈ a`
(similitud coseno > 0.95 con d ≥ 256). Si no se cumple, el binding NO
sirve y se descarta — decisión por datos, no por fe.

### F8. Métricas de evaluación (la prueba reina, Fase 1.5)

Para N preguntas de prueba con respuesta correcta etiquetada manualmente:
```
Precision@k = |relevantes ∩ seleccionados| / |seleccionados|
Recall@k    = |relevantes ∩ seleccionados| / |relevantes|
F1 = 2·(P·R)/(P+R)
```
**El criterio de éxito del proyecto completo:**
```
F1(navegación jerárquica + cube_move)  >  F1(búsqueda plana por coseno)
```
medido sobre ≥100 preguntas reales y ≥1000 nodos. Si esta desigualdad no
se cumple con margen claro (>5 puntos), la tesis central queda refutada
y el proyecto debe pivotar. Se mide, no se asume.

### Reglas de implementación de esta sección

1. Cada fórmula lleva su prueba en la batería 4b. Si una fórmula no tiene
   prueba asignada, no se implementa todavía.
2. Los parámetros libres (τ, λ, D_max, sat_min, umbrales de concentración)
   se fijan con valores iniciales aquí dados y se RECALIBRAN con los
   resultados reales guardados en docs/pruebas/. Nunca se ajustan "a ojo"
   sin registrar el antes/después.
3. Todo cálculo de tonos usa la forma vectorial u(h) — prohibido promediar
   grados como escalares (error circular de F5).

---

## 5. PREGUNTAS ABIERTAS PARA EL USUARIO

- En Fase 2: la curva de saturación es exponencial (F4: λ^depth, λ=0.7 inicial).
  ¿Se acepta así o se quiere una curva lineal? (Se recalibra con datos igual.)
- ¿El prototipo de integración con agentes (Fase 4) se hace sobre `delegate_task`
  de Hermes (subagentes reales) o primero simulado en local?

---

## 6. REFERENCIAS DE LA CONVERSACIÓN (2026-08-09)

- Skills: `axioma-omega-memory`, `axioma-omega-protocol`
- Doc: `references/magnetic-cubes-vision.md` (visión original de cubos magnéticos;
  este plan la extiende con cube_move + cadena de color)
- Auditoría 2026-08-09: split-brain, auto-ingestión, hash() no determinista,
  compactación 1008→28 nodos — todo documentado en el skill axioma-omega-memory.
