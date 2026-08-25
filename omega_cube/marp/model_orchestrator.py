#!/usr/bin/env python3
"""
MARP ModelOrchestrator — Router + Dynamic Model Loader.

Siempre montado. Decide qué modelo local cargar (o API llamar) según:
- Dominio de la consulta (clasificado por Qwen0.8B router)
- Capacidades requeridas (visión, razonamiento, velocidad)
- Thinking config (ON/OFF según modelo)
- Calidad > velocidad siempre

Arquitectura:
  Query → Qwen0.8B Router (siempre en GPU, 85ms, 93% acc)
           ↓ domain
         ModelSelector
           ├─ if vision → Qwen3.6-27B-Omni-v4 (con mmproj)
           ├─ if reasoning → Qwen3.5-27B-Reasoning (thinking ON)
           ├─ if math/code/science → Gemma 4 31B (calidad⭐)
           ├─ if speed → GLM-4.7-Flash (458ms)
           ├─ if multi-task → Qwen3.6-35B-MoE (687ms, 3.5B activos)
           └─ fallback → API (OpenAI/Claude)
           ↓
         Worker responde → respuesta al usuario

Uso:
  python model_orchestrator.py "mi consulta aquí"
  python model_orchestrator.py --interactive
  python model_orchestrator.py --benchmark

Requisitos:
  - Qwen0.8B router en 127.0.0.1:8082 (siempre montado)
  - Modelos locales en J:/modelos_ia/
  - llama.cpp CUDA 13.1 en Downloads/LLAMA~1.CPP/
"""

import os, sys, json, time, re, subprocess, signal, requests
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─── Paths ───────────────────────────────────────────────────────
HOME = Path.home()
LLAMA_DIR = Path("C:/Users/GPAMD/Downloads/LLAMA~1.CPP/llama-b9045-bin-win-cuda-13.1-x64")
LLAMA_SERVER = LLAMA_DIR / "llama-server.exe"
MODELS_DIR = Path("J:/modelos_ia")
LOG_DIR = HOME / ".hermes" / "logs" / "marp_router"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Model Registry ──────────────────────────────────────────────
@dataclass
class ModelProfile:
    name: str
    file: str
    size_gb: float
    type: str           # dense, moe
    capabilities: list  # math, code, science, vision, reasoning, speed
    thinking: str       # "on", "off", "optional" (puede ambas)
    thinking_latency: str
    no_thinking_latency: str
    quality: int        # 1-5
    vram_gb: int
    notes: str

MODEL_REGISTRY = {
    "gemma-4-31b": ModelProfile(
        name="Gemma 4 31B Q4",
        file="gemma-4-31B-it-Q4_K_M.gguf",
        size_gb=18, type="dense",
        capabilities=["math", "code", "science", "language", "general"],
        thinking="none", thinking_latency="N/A", no_thinking_latency="1.18s",
        quality=5, vram_gb=20, notes="Mejor calidad general. Sin thinking forzado."
    ),
    "glm-4.7-flash": ModelProfile(
        name="GLM-4.7-Flash Q4",
        file="GLM-4.7-Flash-Q4_0.gguf",
        size_gb=17, type="dense",
        capabilities=["math", "code", "science", "speed"],
        thinking="none", thinking_latency="N/A", no_thinking_latency="458ms",
        quality=3, vram_gb=19, notes="M�s r�pido de todos. Calidad decente."
    ),
    "qwen3.6-35b-moe": ModelProfile(
        name="Qwen3.6 35B MoE",
        file="Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf",
        size_gb=19, type="moe",
        capabilities=["math", "code", "science", "multi-task"],
        thinking="none", thinking_latency="N/A", no_thinking_latency="687ms",
        quality=4, vram_gb=21, notes="Solo 3.5B activos. Eficiente."
    ),
    "qwen3.6-27b-omni": ModelProfile(
        name="Qwen3.6 27B Omni v4 Q4",
        file="Qwen3.6-27B-Omni-v4-Q4_K_M.gguf",
        size_gb=16, type="dense",
        capabilities=["math", "code", "science", "vision", "general"],
        thinking="optional", thinking_latency=">30s", no_thinking_latency="847ms",
        quality=4, vram_gb=23, notes="Con mmproj para visi�n. Thinking muy verboso."
    ),
    "qwen3.5-27b-reasoning": ModelProfile(
        name="Qwen3.5 27B Reasoning Distilled",
        file="Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.i1-Q4_K_M.gguf",
        size_gb=16, type="dense",
        capabilities=["math", "science", "reasoning", "analysis"],
        thinking="on", thinking_latency="5.5s", no_thinking_latency="N/A (no funciona)",
        quality=5, vram_gb=18, notes="Solo thinking. Respuestas muy detalladas."
    ),
    "qwen3.5-9b": ModelProfile(
        name="Qwen3.5 9B Q8",
        file="Qwen3.5-9B-Q8_0.gguf",
        size_gb=9, type="dense",
        capabilities=["math", "code", "science", "lightweight"],
        thinking="none", thinking_latency="N/A", no_thinking_latency="787ms",
        quality=3, vram_gb=10, notes="Ms ligero. Cabe con router + otro worker."
    ),
    "gemma-4-coder-v2": ModelProfile(
        name="Gemma 4 Coder v2 Q8",
        file="Gemma/gemma4-coder-v2-Q8_0.gguf",
        size_gb=12, type="dense",
        capabilities=["code", "math", "general"],
        thinking="none", thinking_latency="N/A", no_thinking_latency="800ms",
        quality=4, vram_gb=14, notes="Modelo Gemma especializado en código de alta precisión."
    ),
    "qwen3.6-35b-heretic-apex": ModelProfile(
        name="Qwen3.6 35B Heretic APEX",
        file="Qwen/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-I-Quality.gguf",
        size_gb=23.5, type="dense",
        capabilities=["general", "vision", "reasoning"],
        thinking="none", thinking_latency="N/A", no_thinking_latency="1.2s",
        quality=4, vram_gb=24, notes="Variante sin censura 35B A3B con soporte visual."
    ),
    "qwen3.6-40b-deck-opus": ModelProfile(
        name="Qwen3.6 40B Deck Opus",
        file="Qwen/Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-IQ4_XS.gguf",
        size_gb=22, type="dense",
        capabilities=["code", "math", "reasoning", "vision"],
        thinking="none", thinking_latency="N/A", no_thinking_latency="1.4s",
        quality=5, vram_gb=24, notes="Modelo Deckard/Opus de 40B optimizado para lógica matemática y software complejo."
    ),
    "colibri": ModelProfile(
        name="Colibri GLM-5.2 744B MoE",
        file="colibri.bat",
        size_gb=370, type="moe",
        capabilities=["math", "science", "reasoning", "analysis"],
        thinking="on", thinking_latency="heavy", no_thinking_latency="N/A",
        quality=5, vram_gb=0, notes="Corre en CPU via streaming desde disco (370GB int4)."
    ),
}

# ─── Domain routing rules ────────────────────────────────────────
DOMAIN_ROUTES = {
    "math": {
        "primary": "gemma-4-31b",
        "thinking": "colibri",
        "fast": "glm-4.7-flash",
    },
    "code": {
        "primary": "gemma-4-31b",
        "fast": "glm-4.7-flash",
        "light": "qwen3.5-9b",
    },
    "science": {
        "primary": "colibri", 
        "fast": "gemma-4-31b",
        "light": "glm-4.7-flash",
    },
    "engineering": {
        "primary": "gemma-4-31b",
        "fast": "glm-4.7-flash",
    },
    "law": {
        "primary": "gemma-4-31b",
        "fast": "qwen3.6-35b-moe",
    },
    "medical": {
        "primary": "gemma-4-31b",
        "fast": "qwen3.6-35b-moe",
    },
    "business": {
        "primary": "qwen3.6-35b-moe",
        "fast": "glm-4.7-flash",
    },
    "philosophy": {
        "primary": "colibri",
        "fast": "gemma-4-31b",
    },
    "gaming": {
        "primary": "gemma-4-31b",
        "fast": "qwen3.6-35b-moe",
    },
    "language": {
        "primary": "gemma-4-31b",
        "fast": "glm-4.7-flash",
    },
    "general": {
        "primary": "gemma-4-31b",
        "fast": "glm-4.7-flash",
    },
}

# ─── Router client ────────────────────────────────────────────────
class Router:
    """Always-on Qwen0.8B router. Must be on 127.0.0.1:8082."""
    
    ROUTER_URL = os.environ.get("MARP_ROUTER_URL", "http://127.0.0.1:8082") + "/v1/chat/completions"
    
    FEW_SHOT_PROMPT = """Map queries to EXACT domains from: math,code,science,engineering,language,law,medical,business,philosophy,gaming,general.
Output ONLY: domain or domain1,domain2.
Examples:
"derivative of x squared" -> math
"Python async function" -> code
"quantum entanglement" -> science
"Docker compose deploy" -> code
"NDA agreement draft" -> law
"diabetes treatment" -> medical
"NPV calculation excel" -> business
"Kant ethics" -> philosophy
"RPG game design" -> gaming
"translate English Spanish" -> language
"bridge structural analysis" -> engineering"""

    def __init__(self):
        self._available = None

    def check(self) -> bool:
        """Check if router is alive."""
        try:
            r = requests.get(os.environ.get("MARP_ROUTER_URL", "http://127.0.0.1:8082") + "/health", timeout=3)
            self._available = r.json().get("status") == "ok"
        except:
            self._available = False
        return self._available

    def classify(self, query: str) -> tuple[list[str], float, float]:
        """Classify query. Returns (domains, confidence, latency_ms)."""
        t0 = time.perf_counter()
        r = requests.post(self.ROUTER_URL, json={
            "messages": [
                {"role": "system", "content": self.FEW_SHOT_PROMPT},
                {"role": "user", "content": query[:200]}
            ],
            "max_tokens": 10, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}
        }, timeout=10)
        elapsed = (time.perf_counter() - t0) * 1000
        raw = r.json()["choices"][0]["message"]["content"].strip().lower()
        valid = {"math","code","science","engineering","language","law",
                 "medical","business","philosophy","gaming","general"}
        domains = [d.strip() for d in raw.replace("domains:","").split(",")[:2]]
        domains = [d for d in domains if d in valid] or ["general"]
        confidence = 0.7 if len(domains) == 1 else 0.5
        if domains == ["general"]: confidence = 0.2
        return domains, confidence, elapsed


# ─── Model Server Manager ────────────────────────────────────────
class ModelServer:
    """Manages llama-server lifecycle. Kills old, starts new."""
    
    WORKER_PORT = int(os.environ.get("MARP_WORKER_PORT", "8084"))
    
    def __init__(self):
        self._current_model = None
        self._process: Optional[subprocess.Popen] = None
    
    def kill(self):
        """Kill only the llama-server running on WORKER_PORT."""
        try:
            output = subprocess.check_output("netstat -ano", shell=True, text=True)
            pids = set()
            for line in output.strip().split('\n'):
                if f":{self.WORKER_PORT}" in line and "LISTENING" in line:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        pids.add(parts[-1])
            for pid in pids:
                subprocess.run(["taskkill.exe", "/F", "/PID", pid], capture_output=True)
            time.sleep(1)
        except Exception as e:
            if self._process:
                try:
                    self._process.kill()
                except:
                    pass
        self._current_model = None
        self._process = None
    
    def start(self, model_key: str) -> bool:
        """Start a model server. Returns True if successful."""
        if model_key == self._current_model:
            return True  # Already running
        
        profile = MODEL_REGISTRY.get(model_key)
        if not profile:
            print(f"[ModelServer] Unknown model: {model_key}")
            return False

        if model_key == "colibri":
            self.kill()
            time.sleep(2)
            colibri_bat = Path.home() / ".hermes" / "colibri" / "colibri.bat"
            if not colibri_bat.exists():
                print(f"[ModelServer] Colibri wrapper not found at {colibri_bat}")
                return False
            cmd = [
                str(colibri_bat),
                "--model", "C:/Users/GPAMD/.hermes/colibri/models/glm5.2-int4",
                "--experts-dir", "C:/Users/GPAMD/.hermes/colibri/experts/",
                "--server",
                "--port", str(self.WORKER_PORT),
                "--host", "127.0.0.1"
            ]
            print(f"[ModelServer] Loading Colibri (744B MoE CPU Streaming)...")
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            for i in range(10):
                try:
                    r = requests.get(f"http://127.0.0.1:{self.WORKER_PORT}/health", timeout=2)
                    if r.status_code == 200 or r.json().get("status") == "ok":
                        print(f"[ModelServer] Colibri READY ({i+1}s)")
                        self._current_model = "colibri"
                        return True
                except:
                    pass
                time.sleep(1)
            print(f"[ModelServer] TIMEOUT loading Colibri (Verifica que el modelo de 370GB este descargado)")
            return False
        
        model_path = MODELS_DIR / profile.file
        if not model_path.exists():
            print(f"[ModelServer] File not found: {model_path}")
            return False
        
        # Kill current
        self.kill()
        time.sleep(2)
        
        # Determine context size dynamically based on model key
        new_ctx = 2048
        if "35b" in model_key or "31b" in model_key or "27b" in model_key:
            new_ctx = 98304
        elif "flash" in model_key or "9b" in model_key or "12b" in model_key:
            new_ctx = 131072
        else:
            new_ctx = 32768

        # Start new
        cmd = [
            str(LLAMA_SERVER),
            "-m", str(model_path),
            "-ngl", "99",
            "-c", str(new_ctx),
            "--cache-type-k", "q4_0",
            "--cache-type-v", "q4_0",
            "--flash-attn", "on",
            "--port", str(self.WORKER_PORT),
            "--host", "127.0.0.1",
        ]
        
        if profile.thinking == "none":
            pass  # No flags needed
        elif profile.thinking == "on":
            cmd += ["--reasoning-format", "deepseek"]
        # "optional" handled at request time via enable_thinking
        
        print(f"[ModelServer] Loading {profile.name} (ctx={new_ctx})...")
        self._process = subprocess.Popen(
            cmd, cwd=str(LLAMA_DIR),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # Wait for ready
        max_wait = 60 if profile.size_gb > 15 else 30
        for i in range(max_wait):
            try:
                r = requests.get(f"http://127.0.0.1:{self.WORKER_PORT}/health", timeout=2)
                if r.json().get("status") == "ok":
                    print(f"[ModelServer] {profile.name} READY ({i+1}s)")
                    self._current_model = model_key
                    return True
            except:
                pass
            time.sleep(1)
        
        print(f"[ModelServer] TIMEOUT loading {profile.name}")
        return False
    
    # Tokens de control de Gemma 4 que causan loops si no se filtran
    _GEMMA_CTRL_RE = re.compile(
        r'<\|channel\|>.*?<channel\|>|'
        r'<\|channel\|>.*?$|'
        r'<channel\|>|'
        r'<\|im_start\|>|<\|im_end\|>|'
        r'<start_of_turn>|<end_of_turn>|'
        r'<bos>|<eos>',
        re.DOTALL
    )

    def query(self, prompt: str, max_tokens: int = 100,
              use_thinking: bool = False) -> tuple[str, float]:
        """Query the current worker model."""
        profile = MODEL_REGISTRY.get(self._current_model)

        # Determine endpoint and payload based on model family
        if profile and profile.name.startswith("Gemma"):
            # --- Gemma 4: /completion + prompt manual ---
            # No usar /v1/chat/completions con Jinja: causa loop infinito
            # por re-inyección de reasoning_content y tokens de canal.
            endpoint = f"http://127.0.0.1:{self.WORKER_PORT}/completion"
            gemma_prompt = (
                f"<bos><start_of_turn>user\n"
                f"{prompt}"
                f"<end_of_turn>\n<start_of_turn>model\n"
            )
            payload = {
                "prompt":         gemma_prompt,
                "max_tokens":     max_tokens,
                "temperature":    0.35,
                "repeat_penalty": 1.1,   # evita colapso de probabilidad
                "stop":           ["<end_of_turn>", "<eos>", "<|im_end|>"],
            }

        elif profile and "GLM" in profile.name:
            endpoint = f"http://127.0.0.1:{self.WORKER_PORT}/completion"
            glm_prompt = f"<|system|>Answer concisely.<|user|>{prompt}<|assistant|>"
            payload = {
                "prompt":      glm_prompt,
                "max_tokens":  max_tokens,
                "temperature": 0.3,
                "stop":        ["<|user|>", "<|endoftext|>"],
            }

        else:
            # Qwen y familia: /v1/chat/completions con Jinja nativo
            endpoint = f"http://127.0.0.1:{self.WORKER_PORT}/v1/chat/completions"
            payload = {
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  max_tokens,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": use_thinking},
            }

        t0 = time.perf_counter()
        try:
            timeout = 120 if use_thinking else 60
            r = requests.post(endpoint, json=payload, timeout=timeout)
            elapsed = (time.perf_counter() - t0) * 1000

            data = r.json()
            # /completion → campo "content"; /v1/chat/completions → choices
            if "/v1/chat" not in endpoint:
                raw = data.get("content", "") or ""
            else:
                raw = (data["choices"][0]["message"].get("content") or "")

            # Limpiar thinking tags
            cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

            # Limpiar tokens de control de Gemma 4 si aplica
            if profile and profile.name.startswith("Gemma"):
                cleaned = self._GEMMA_CTRL_RE.sub('', cleaned).strip()

            return (cleaned or raw[:300]), elapsed

        except Exception as e:
            return f"[ERROR] {e}", -1
    
    @property
    def current(self) -> Optional[str]:
        return self._current_model


# ─── Model Selector ───────────────────────────────────────────────
class ModelSelector:
    """Selects best model based on domain, capabilities, and context."""
    
    def select(self, domains: list[str], needs_vision: bool = False,
               needs_reasoning: bool = False, use_fast: bool = False) -> str:
        """Select the best model for the task."""
        primary_domain = domains[0] if domains else "general"
        
        # Check vision first
        if needs_vision:
            return "qwen3.6-27b-omni"
        
        # Get domain route
        route = DOMAIN_ROUTES.get(primary_domain, DOMAIN_ROUTES["general"])
        
        if needs_reasoning:
            return route.get("thinking", route["primary"])
        if use_fast:
            return route.get("fast", route["primary"])
        
        return route["primary"]
    
    def needs_reasoning(self, query: str) -> bool:
        """Detect if query needs deep reasoning."""
        reasoning_triggers = ["prove", "analyze", "explain in detail", 
                             "compare and contrast", "why", "how does",
                             "implications", "philosophical", "ethical"]
        q = query.lower()
        return any(t in q for t in reasoning_triggers)
    
    def needs_vision(self, query: str) -> bool:
        """Detect if query needs vision/image understanding."""
        vision_triggers = ["image", "picture", "photo", "diagram", "graph",
                          "visual", "see", "look at this", "what is in",
                          "describe this", "screenshot"]
        q = query.lower()
        return any(t in q for t in vision_triggers)


# ─── Logging ──────────────────────────────────────────────────────
class OracleLogger:
    def __init__(self):
        self.entries = []
    
    def log(self, query: str, domains: list[str], model: str,
            latency: float, thinking: bool, error: str = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],
            "domains": domains,
            "model": model,
            "latency_ms": round(latency, 1),
            "thinking": thinking,
            "error": error,
        }
        self.entries.append(entry)
        
        # Write to JSONL
        log_file = LOG_DIR / f"oracle_{datetime.now():%Y%m%d}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def stats(self) -> dict:
        if not self.entries:
            return {}
        latencies = [e["latency_ms"] for e in self.entries if e["latency_ms"] > 0]
        models = {}
        for e in self.entries:
            models[e["model"]] = models.get(e["model"], 0) + 1
        return {
            "total": len(self.entries),
            "avg_latency_ms": round(sum(latencies)/len(latencies), 0) if latencies else 0,
            "models": models,
            "errors": sum(1 for e in self.entries if e["error"]),
        }


# ─── Main Pipeline ────────────────────────────────────────────────
class MARPPipeline:
    def __init__(self):
        self.router = Router()
        self.selector = ModelSelector()
        self.server = ModelServer()
        self.logger = OracleLogger()
    
    def process(self, query: str, needs_vision: bool = None,
                needs_reasoning: bool = None) -> str:
        """Process a query through the full MARP pipeline."""
        
        # 1. Route
        if not self.router.check():
            return "[ERROR] Router not available on 8082"
        
        domains, confidence, rt_latency = self.router.classify(query)
        
        # 2. Detect needs
        if needs_vision is None:
            needs_vision = self.selector.needs_vision(query)
        if needs_reasoning is None:
            needs_reasoning = self.selector.needs_reasoning(query)
        
        # 3. Select model
        model_key = self.selector.select(domains, needs_vision, needs_reasoning)
        profile = MODEL_REGISTRY[model_key]
        
        # 4. Determine thinking
        use_thinking = False
        if profile.thinking == "on":
            use_thinking = True
        elif profile.thinking == "optional" and needs_reasoning:
            use_thinking = True
        
        # 5. Load/switch model if needed
        if not self.server.start(model_key):
            # Fallback to first available
            for fallback in ["gemma-4-31b", "glm-4.7-flash"]:
                if self.server.start(fallback):
                    model_key = fallback
                    profile = MODEL_REGISTRY[fallback]
                    use_thinking = False
                    break
        
        # 6. Query worker
        answer, latency = self.server.query(
            prompt=query,
            max_tokens=200 if use_thinking else 100,
            use_thinking=use_thinking
        )
        
        # 7. Log
        self.logger.log(query, domains, profile.name, latency, use_thinking)
        
        # 8. Build response
        error = None
        if latency < 0:
            error = answer
            answer = f"[ERROR] Worker failed: {error}"
        
        return answer


# ─── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MARP ModelOrchestrator")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark")
    parser.add_argument("--stats", "-s", action="store_true", help="Show today stats")
    parser.add_argument("--vision", action="store_true", help="Force vision mode")
    parser.add_argument("--reasoning", action="store_true", help="Force reasoning mode")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()
    
    pipeline = MARPPipeline()
    
    if args.list_models:
        print(f"{'Model':30s} {'Type':10s} {'Thinking':12s} {'Latency':12s} {'Quality':8s}")
        print("-"*75)
        for key, prof in MODEL_REGISTRY.items():
            t = prof.thinking if prof.thinking != "none" else "No"
            lt = prof.no_thinking_latency if prof.thinking == "optional" else f"think:{prof.thinking_latency}"
            print(f"{prof.name:30s} {prof.type:10s} {t:12s} {lt:12s} {'*'*prof.quality:8s}")
        sys.exit(0)
    
    if args.stats:
        stats = pipeline.logger.stats()
        print(json.dumps(stats, indent=2))
        sys.exit(0)
    
    if args.benchmark:
        tests = [
            "What is the derivative of x squared?",
            "Write a Python async HTTP function",
            "Explain CRISPR gene editing",
            "How to deploy Docker to Kubernetes?",
            "Draft a non-disclosure agreement",
            "What are symptoms of diabetes?",
            "Calculate net present value",
            "Is free will compatible with determinism?",
            "Design a roguelike death mechanic",
            "What are the main parts of a suspension bridge?",
        ]
        print(f"MARP Orchestrator Benchmark: {len(tests)} queries")
        print("="*60)
        for q in tests:
            print(f"\nQuery: {q}")
            result = pipeline.process(q)
            print(f"  Router: {pipeline.router.classify(q)[0]}")
            print(f"  Model:  {pipeline.server.current}")
            print(f"  Answer: {result[:100]}")
        print(f"\nStats: {json.dumps(pipeline.logger.stats(), indent=2)}")
        sys.exit(0)
    
    if args.query:
        result = pipeline.process(args.query, args.vision, args.reasoning)
        print(result)
    
    elif args.interactive:
        print("MARP ModelOrchestrator — Interactive")
        print(f"Router: {'OK' if pipeline.router.check() else 'DOWN'}")
        print(f"Models: {len(MODEL_REGISTRY)} available")
        while True:
            try:
                q = input("\n> ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if not q: continue
                result = pipeline.process(q)
                print(f"\n{result}")
            except KeyboardInterrupt:
                break
        print(f"\nSession stats: {json.dumps(pipeline.logger.stats(), indent=2)}")
    
    else:
        print("MARP ModelOrchestrator v1.0")
        print(f"Router: {'OK' if pipeline.router.check() else 'DOWN'} on :8082")
        print(f"Models: {len(MODEL_REGISTRY)} registered")
        print(f"Logs:   {LOG_DIR}")
        print(f"Use:    --interactive | --benchmark | 'your query'")
