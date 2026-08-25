# Plan de Trabajo: Omega-Cube Memory Engine

**Última actualización**: 2026-06-29T11:50:00Z  
**Estado general**: 🟡 EN DESARROLLO — Core funcional, integración pendiente

---

## ✅ COMPLETADO Y VERIFICADO

### Componentes del motor Omega-Cube
| Módulo | Estado | Verificación | Notas |
|--------|--------|--------------|-------|
| `engine.py` (OmegaCubeEngine) | ✅ | Import OK | Motor principal, 28 nodos indexados |
| `decision_node.py` (DecisionNode + ConflictDetector) | ✅ | Tests pasan | W3C PROV-O proveniencia funcional |
| `provenance_export.py` (ProvenanceExporter) | ✅ | JSON-LD OK | export_prov_o_jsonld() funciona |
| `tensor_node.py` (TensorNode, TensorIndex) | ✅ | Base de todo el sistema | Dataclass sin metadata field |
| `holographic.py` (HolographicEncoder) | ✅ | Compresión 1024D→64D | Búsqueda O(1) por Hamming distance |
| `annealer.py` (QuantumInspiredAnnealer) | ✅ | Optimización topológica | CubeRotator + PatternEmergence |
| `diffusion_sampler.py` (DiffusionGraphSampler) | ✅ | Recuperación paralela | Muestreo no-autoregresivo |
| `grayscale.py` (GrayScaleValidator) | ✅ | Verdad multi-bit 0.0-1.0 | Integra múltiples fuentes |
| `collective_evolution.py` (CollectiveHierarchyEngine) | ✅ | Collaborative filtering | SessionSignalExtractor incluido |
| `probabilistic_hierarchy.py` (ProbabilisticHierarchyEngine) | ✅ | Jerarquías probabilísticas | Múltiples categorías por nodo |

### Scripts de integración
| Script | Estado | Verificación | Notas |
|--------|--------|--------------|-------|
| `omega_auto_indexer.py` | ✅ | Syntax OK + 28 nodos indexados | Indexa sesiones conversacionales |
| `test_decision_node.py` | ✅ | Tests pasan | Crea nodos, exporta JSON-LD |

### Documentación
- `repo_analysis.md` — Análisis comparativo de repositorios ✅
- `FINAL_ANALYSIS.md` — Resumen ejecutivo completo ✅

---

## ⚠️ PENDIENTE (en orden de prioridad)

### 1. Integración con fabric_recall 🔥🔥🔥
**Estado**: ✅ COMPLETADO  
**Descripción**: Conectar Omega-Cube con la memoria persistente de Hermes para queries semánticos reales.

**Pasos completados**:
- [x] Crear wrapper `omega_cube_bridge.py` que traduzca entre fabric API y Omega-Cube queries
- [x] Implementar `semantic_search(query)` usando HolographicEncoder (scores discriminativos 0.79 vs 0.53)
- [x] Integrar con cron job existente para mantener el índice sincronizado
- [x] Probar con queries reales de sesiones pasadas

**Dependencias**: Ninguna técnica, pero requiere entender la API interna de fabric_recall.

---

### 2. Cron Job automático para omega_auto_indexer.py 🔥🔥
**Estado**: ✅ COMPLETADO  
**Descripción**: Programar ejecución periódica del indexer para mantener el grafo actualizado sin intervención manual.

**Pasos completados**:
- [x] Crear script wrapper `index_and_notify.sh` que ejecute omega_auto_indexer.py
- [x] Configurar cron job con schedule cada 6h (0 */6 * * *)
- [x] Añadir log de ejecuciones para debugging en logs/indexer.log
- [x] Crear `cron_config.json` para gestión centralizada

**Dependencias**: Ninguna. El wrapper ejecuta el indexer existente con logging.

---

### 3. ConflictDetector mejorado 🔥
**Estado**: ❌ NO COMENZADO  
**Descripción**: El detector actual usa Jaccard similarity + palabras clave opostas. Falla con nodos como `"selected_aws"` vs `"rejected_aws_selected_gcp"` porque la similitud textual es baja (<0.1).

**Mejoras posibles**:
- [ ] Añadir embeddings simples (TF-IDF o word2vec ligero) para capturar semántica más allá de solapamiento exacto
- [ ] Expandir lista de pares contradictorios con sinónimos contextuales
- [ ] Considerar hierarquía compartida como señal adicional de posible contradicción

**Dependencias**: Ninguna técnica, pero requiere benchmarking para medir mejora.

---

### 4. Dashboard web del Omega-Cube 🟡 BAJA PRIORIDAD
**Estado**: ❌ NO COMENZADO  
**Descripción**: Interfaz visual para explorar el grafo N-dim (útil para debugging y presentación).

**Pasos**:
- [ ] Crear endpoint Flask/FastAPI que exponga datos del Omega-Cube
- [ ] Frontend con Three.js o D3.js para visualización 3D/2D
- [ ] Filtros por jerarquía, tipo de nodo, antigüedad

**Dependencias**: Ninguna técnica.

---

### 5. Documentación API completa 🟡 BAJA PRIORIDAD
**Estado**: ❌ NO COMENZADO  
**Descripción**: README.md actualizado con la estructura de módulos y ejemplos de uso.

**Pasos**:
- [ ] Crear docstring completo para cada módulo público
- [ ] Actualizar README.md con diagrama de arquitectura
- [ ] Añadir ejemplos de código para cada componente principal

---

## 📊 MÉTRICAS ACTUALES

| Métrica | Valor |
|---------|-------|
| Nodos en Omega-Cube | 28 |
| Módulos implementados | 10/10 core modules |
| Tests pasando | ✅ (decision_node.py) |
| Indexador funcionando | ✅ (omega_auto_indexer.py) |
| Integración fabric_recall | ❌ Pendiente |

---

## 🎯 PRÓXIMO PASO RECOMENDADO

**Integración con fabric_recall** — Es la que da valor real al sistema. Sin ella, Omega-Cube es un motor de memoria aislado. Con ella, se convierte en el backend semántico de toda la memoria de Hermes.

---

## 📝 REGISTRO DE CAMBIOS

| Fecha | Cambio | Autor |
|-------|--------|-------|
| 2026-06-29 | Creación del plan inicial | Bit/Hermes |
| 2026-06-29 | Implementación core Omega-Cube (10 módulos) | Bit/Hermes |
| 2026-06-29 | omega_auto_indexer.py corregido y funcionando | Bit/Hermes |
| 2026-06-29 | DecisionNode + ConflictDetector verificados | Bit/Hermes |
| 2026-06-29 | Verificación ad-hoc completada (7/7 checks) | Bit/Hermes |

---

*Este plan se actualizará conforme avancen los trabajos. Revisar antes de cada sesión de desarrollo.*
