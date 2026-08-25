# MARP v2 — Router Único (consolidación P1.6)

**Un solo clasificador**: `marp/gpu_router.py` (`QwenGPURouter`). Todo lo demás
fue eliminado el 2026-08-25 (backups `*.bak-20260825-p16` junto a cada archivo):

| Eliminado | Era | Motivo |
|-----------|-----|--------|
| `qwen_classifier.py` | Qwen GGUF in-process vía llama-cpp-python | 0 importers; muerto |
| `intelligent_pipeline.py` | CLI duplicada de pre-filtro+GBNF+worker | Lógica idéntica ya vive en `gpu_router.py`; worker :8084 es responsabilidad del orquestador, no del router |
| `router_service.py` | Servicio SmolLM2 + scheduler | Superseded por GPU router; su `LogAnalyzer` vive ahora en `log_analyzer.py` |

## Cadena de degradación (explícita, en orden)

```
Query
 │
 ├─ 1. Keyword pre-filter (~0.1ms)      → dominio, conf 0.90   [edge cases ES]
 ├─ 2. llama-server /completion + GBNF (~250ms) → dominio, conf 0.85
 └─ 3. Servidor caído / timeout          → ("general", conf 0.05), nunca lanza
```

Toda clasificación se registra en `~/.hermes/logs/marp_router/marp_YYYYMMDD.jsonl`.
Métricas del día: `python omega_cube/marp_analyze_today.py`.

## Servidor

```bash
llama-server -m P:/AI_INFRA/custom_models/Qwen/Qwen3.5-0.8B-Q6_K.gguf \
  -ngl 99 -c 1024 --port 8082 --host 127.0.0.1 --alias marp-router \
  --reasoning-format none
```

Sin servidor levantado el router **sigue funcionando** en modo degradado
(keywords → general); no requiere VRAM reservada.

## Uso

```python
from omega_cube.marp.gpu_router import QwenGPURouter

router = QwenGPURouter()
domains, confidence = router.classify("editar video con transiciones")
# (["general"], 0.90)   ← tier keyword
```

Benchmark incluido: `PYTHONPATH=. python -m omega_cube.marp.gpu_router`
(20 queries etiquetadas; referencia RTX 3090: 90% accuracy, ~250 ms avg).
