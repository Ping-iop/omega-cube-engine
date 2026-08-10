# MARP Intelligent Pipeline v2 — Router 100% + Thinking Dinámico

El pipeline inteligente de MARP ahora tiene **router Qwen0.8B con 100% accuracy** y decisiones de thinking dinámicas por dominio.

## Novedades v2

### Router 100% de precisión
El secreto: **formato de completado single-user-message**.
- ❌ System + user prompt → el modelo 0.8B ignora el system y completa la pregunta → accuracy 0-40%
- ✅ Un solo mensaje user con formato `query: ... domain:` y 10 ejemplos few-shot → **accuracy 100%**

### Thinking Dinámico por Dominio
| Dominio | Thinking | Comportamiento |
|---------|----------|----------------|
| math, science, philosophy, code, engineering | 🧠 ON | max_tokens × 3, respuesta detallada paso a paso |
| language, business, gaming, law, medical | ⚡ OFF | respuesta directa, rápida |
| Queries simples (ej: "what is 2+2?") | ⚡ OFF | detectado por regex SIMPLE_PATTERNS |

### Fallback Inteligente
Si thinking consume todos los tokens sin completar la respuesta, se re-consulta automáticamente sin thinking → respuesta garantizada.

## Instalación

### Servidores requeridos
```bash
# Router Qwen0.8B (siempre activo)
llama-server -m J:/modelos_ia/qwen3.5-0.8b-instruct-Q4_K_M.gguf -ngl 99 -c 256 --port 8084 --host 127.0.0.1 --reasoning-format none

# Worker Qwen 27B Omni 
llama-server -m J:/modelos_ia/Qwen3.6-27B-Omni-v4-Q4_K_M.gguf -ngl 99 -c 4096 --port 8082 --host 127.0.0.1 --mlock --reasoning-format none
```

### Auto-inicio (Windows)
Ver `scripts/start_marp_servers.bat` — Task Scheduler compatible.

## Uso

```bash
cd omega_cube
PYTHONPATH="$PWD" python marp/intelligent_pipeline.py                # Interactivo
PYTHONPATH="$PWD" python marp/intelligent_pipeline.py "tu pregunta"  # Una query
PYTHONPATH="$PWD" python marp/intelligent_pipeline.py --benchmark    # Benchmark 10 queries
PYTHONPATH="$PWD" python marp/intelligent_pipeline.py --test         # Quick test 3 queries
```

## Benchmark (10 queries, Qwen 27B Omni)

```
Router accuracy: 10/10 (100%)
🧠 Thinking ON:  5 queries  → avg 16.5s (razonamiento profundo)
⚡ Thinking OFF: 5 queries  → avg  3.2s (respuesta directa)
```

## Arquitectura

```
Query → Router Qwen0.8B (:8084) → clasifica dominio
    ↓
Decide thinking ON/OFF según dominio + simplicidad
    ↓
Worker Qwen 27B Omni (:8082) → respuesta
    ↓
[Fallback] Si thinking no completa → re-consulta sin thinking
```
