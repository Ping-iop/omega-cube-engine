---
name: marp-orchestrator
description: |
  MARP ModelOrchestrator — Always-on router + dynamic model loader.
  Forces the Qwen0.8B router to always run on GPU port 8084.
  Automatically selects and loads the best local model per query domain.
  Supports: Gemini 4 31B, Qwen3.6 35B MoE, GLM-4.7-Flash, Qwen3.5 27B Reasoning.
  Thinking ON/OFF per model. Vision capable when requested.
  Logs all queries to ~/.hermes/logs/marp_router/oracle_*.jsonl
---

# MARP ModelOrchestrator Skill

## When to activate
- Any query that needs model inference routing
- When you want the best local model for the task
- When you need vision, reasoning, or fast response models

## Architecture

```
User Query
    │
    ▼
Qwen0.8B Router (:8084) — siempre montado
    │  Clasifica: math,code,science,law,medical,business,etc.
    ▼
ModelSelector
    │  Según dominio + capacidades requeridas
    ├─ vision → Qwen3.6-27B-Omni-v4 (con mmproj)
    ├─ reasoning → Qwen3.5-27B-Reasoning (thinking ON)
    ├─ math/code/science → Gemma 4 31B (calidad⭐)
    ├─ speed → GLM-4.7-Flash (458ms)
    └─ multi-task → Qwen3.6-35B-MoE (687ms)
    ▼
Worker responde → resultado
```

## Available Models

| Modelo | Thinking | Latencia | Calidad |
|--------|----------|----------|---------|
| Gemma 4 31B Q4 | No | 1.18s | ⭐⭐⭐ Excelente |
| GLM-4.7-Flash | No | 458ms | ⭐⭐ Rápido |
| Qwen3.6 35B MoE | No | 687ms | ⭐⭐ Eficiente |
| Qwen3.6 27B Omni v4 | Optional | 847ms/30s+ | ⭐⭐ Con visión |
| Qwen3.5 27B Reasoning | Sí (forzado) | 5.5s | ⭐⭐⭐ Detallado |
| Qwen3.5 9B Q8 | No | 787ms | ⭐ Ligero |

## Usage

### Prerequisites
1. Router must be running: `llama-server -m qwen3.5-0.8b-instruct-Q4_K_M.gguf -ngl 99 -c 128 --port 8084`
2. Models in `J:/modelos_ia/`
3. llama.cpp CUDA 13.1 in `C:/Users/GPAMD/Downloads/Llama.cpp Cuda/`

### Commands

```bash
# Quick query (Router + auto model selection)
cd ~/.hermes/axioma-omega-protocol/omega_cube
python marp/model_orchestrator.py "What is the derivative of x squared?"

# Interactive mode
python marp/model_orchestrator.py --interactive

# Benchmark (10 queries across domains)
python marp/model_orchestrator.py --benchmark

# List available models
python marp/model_orchestrator.py --list-models

# Force reasoning/vision
python marp/model_orchestrator.py "Prove the Riemann hypothesis" --reasoning
python marp/model_orchestrator.py "What is in this image?" --vision
```

## Model Selection Logic

```python
math      → Gemma 4 31B (primary) | Qwen 27B Reasoning (thinking)
code      → Gemma 4 31B (primary) | GLM-4.7-Flash (fast)
science   → Qwen 27B Reasoning (thinking) | Gemma 4 31B
law       → Gemma 4 31B | Qwen MoE
medical   → Gemma 4 31B | Qwen MoE
business  → Qwen MoE | GLM-4.7-Flash
philosophy → Qwen 27B Reasoning | Gemma 4 31B
gaming    → Gemma 4 31B | Qwen MoE
language  → Gemma 4 31B | GLM-4.7-Flash
vision    → Qwen 27B Omni v4 (con mmproj)
```

## Domain Routing Rules

Quality-first: PRIMARY = best model for domain.
If reasoning needed → use thinking model.
If speed critical → use flash model.

## Logs

All queries logged to `~/.hermes/logs/marp_router/oracle_*.jsonl`
```json
{
  "timestamp": "2026-06-12T19:00:00",
  "query": "derivative of x squared",
  "domains": ["math"],
  "model": "Gemma 4 31B Q4",
  "latency_ms": 1180,
  "thinking": false
}
```
