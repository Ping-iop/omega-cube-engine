#!/usr/bin/env python3
"""
MARP MCP Server — Model-Agnostic Routing Protocol via MCP.

Expone el ModelOrchestrator como herramientas MCP para que Hermes
las use automáticamente sin intervención del usuario.

Tools:
  - marp_query:     Clasifica + responde con el mejor modelo local
  - marp_switch:    Cambia manualmente de modelo worker
  - marp_status:    Estado actual del pipeline (router, worker, VRAM)
  - marp_vision:    Procesa imagen con el modelo con mmproj cargado

Siempre disponible cuando Hermes arranca (via cron + startup script).
"""

import sys, json, os, time, re, subprocess, signal, shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
import requests

# ─── Paths ───────────────────────────────────────────────────────
HOME = Path.home()
LLAMA_DIR = Path("C:/Users/GPAMD/Downloads/Llama.cpp Cuda/llama-b9045-bin-win-cuda-13.1-x64")
LLAMA_SERVER = LLAMA_DIR / "llama-server.exe"
MODELS_DIR = Path("J:/modelos_ia")
LOG_DIR = HOME / ".hermes" / "logs" / "marp_router"
LOG_DIR.mkdir(parents=True, exist_ok=True)

ROUTER_PORT = 8084
WORKER_PORT = 8082

# ─── Model Registry with mmproj ─────────────────────────────────
MODEL_REGISTRY = [
    {
        "key": "gemma-31b",
        "name": "Gemma 4 31B Q4",
        "file": "gemma-4-31B-it-Q4_K_M.gguf",
        "mmproj": "gemma-4-mmproj-BF16.gguf",
        "size_gb": 18,
        "type": "dense",
        "domains": ["math", "code", "science", "engineering", "law", "medical", "business", "philosophy", "gaming", "language", "general"],
        "thinking": "none",
        "quality": 5,
        "prerequisites": [],
    },
    {
        "key": "glm-flash",
        "name": "GLM-4.7-Flash Q4",
        "file": "GLM-4.7-Flash-Q4_0.gguf",
        "mmproj": None,
        "size_gb": 17,
        "type": "dense",
        "domains": ["math", "code", "science", "speed", "language"],
        "thinking": "none",
        "quality": 3,
        "prerequisites": [],
    },
    {
        "key": "qwen-moe",
        "name": "Qwen3.6 35B MoE (3.5B act)",
        "file": "Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf",
        "mmproj": "qwen3.6_35B_A3B_mmproj-BF16.gguf",
        "size_gb": 19,
        "type": "moe",
        "domains": ["math", "code", "science", "business", "engineering", "general"],
        "thinking": "none",
        "quality": 4,
        "prerequisites": [],
    },
    {
        "key": "qwen-omni",
        "name": "Qwen3.6 27B Omni v4 Q4",
        "file": "Qwen3.6-27B-Omni-v4-Q4_K_M.gguf",
        "mmproj": "qwen3.6_27b_mmproj-BF16.gguf",
        "size_gb": 16,
        "type": "dense",
        "domains": ["vision", "math", "code", "science", "general"],
        "thinking": "optional",
        "quality": 4,
        "prerequisites": [],
    },
    {
        "key": "qwen-reasoning",
        "name": "Qwen3.5 27B Reasoning Distilled",
        "file": "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.i1-Q4_K_M.gguf",
        "mmproj": None,
        "size_gb": 16,
        "type": "dense",
        "domains": ["reasoning", "math", "science", "philosophy", "analysis"],
        "thinking": "on",
        "quality": 5,
        "prerequisites": [],
    },
    {
        "key": "qwen-9b",
        "name": "Qwen3.5 9B Q8",
        "file": "Qwen3.5-9B-Q8_0.gguf",
        "mmproj": None,
        "size_gb": 9,
        "type": "dense",
        "domains": ["lightweight", "code", "math", "language"],
        "thinking": "none",
        "quality": 3,
        "prerequisites": [],
    },
]

# ─── Router (always-on, port 8084) ──────────────────────────────
ROUTER_PROMPT = """Map queries to EXACT domains from: math,code,science,engineering,language,law,medical,business,philosophy,gaming,general.
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
"bridge structural analysis" -> engineering
"What is in this image" -> general"""


class MARPServer:
    """Manages router + worker llama-server instances."""

    def __init__(self):
        self._current_worker = None
        self._worker_process: Optional[subprocess.Popen] = None
        self._thinking_mode = False

    # ── Router ────────────────────────────────────────────────
    def ensure_router(self) -> bool:
        """Ensure router (Qwen0.8B) is running on :8084."""
        if self._check_health(ROUTER_PORT):
            return True
        model_path = MODELS_DIR / "qwen3.5-0.8b-instruct-Q4_K_M.gguf"
        if not model_path.exists():
            return False
        subprocess.Popen(
            [str(LLAMA_SERVER), "-m", str(model_path), "-ngl", "99", "-c", "128",
             "--port", str(ROUTER_PORT), "--host", "127.0.0.1"],
            cwd=str(LLAMA_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        for _ in range(15):
            if self._check_health(ROUTER_PORT):
                return True
            time.sleep(1)
        return False

    def classify(self, query: str) -> dict:
        """Classify query via router. Returns {domains, confidence, latency_ms}."""
        t0 = time.perf_counter()
        try:
            r = requests.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions",
                json={
                    "messages": [{"role": "system", "content": ROUTER_PROMPT},
                                 {"role": "user", "content": query[:200]}],
                    "max_tokens": 10, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}
                }, timeout=10)
            elapsed = (time.perf_counter() - t0) * 1000
            raw = r.json()["choices"][0]["message"]["content"].strip().lower()
            valid = {"math","code","science","engineering","language","law",
                     "medical","business","philosophy","gaming","general"}
            domains = [d.strip() for d in raw.replace("domains:","").split(",")[:2]]
            domains = [d for d in domains if d in valid] or ["general"]
            return {"domains": domains, "confidence": 0.7 if len(domains)==1 else 0.5,
                    "latency_ms": round(elapsed, 1)}
        except Exception as e:
            return {"domains": ["general"], "confidence": 0.1, "latency_ms": -1, "error": str(e)}

    # ── Worker management ─────────────────────────────────────
    def _check_health(self, port: int) -> bool:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            return r.json().get("status") == "ok"
        except:
            return False

    def kill_worker(self):
        """Kill current worker process."""
        try:
            subprocess.run(["taskkill.exe", "//F", "//IM", "llama-server.exe"],
                          capture_output=True, timeout=5)
        except:
            pass
        time.sleep(2)
        self._current_worker = None
        self._worker_process = None

    def _find_model(self, query_domains: list, needs_vision: bool = False,
                    needs_reasoning: bool = False) -> dict:
        """Select the best model for the task."""
        domain = query_domains[0] if query_domains else "general"

        # Vision forces Omni
        if needs_vision:
            for m in MODEL_REGISTRY:
                if m["key"] == "qwen-omni":
                    return m

        # Reasoning forces reasoning model
        if needs_reasoning:
            for m in MODEL_REGISTRY:
                if m["key"] == "qwen-reasoning":
                    return m

        # Domain-based selection: pick highest quality that covers this domain
        best = None
        for m in MODEL_REGISTRY:
            if domain in m["domains"] or "general" in m["domains"]:
                if best is None or m["quality"] > best["quality"]:
                    best = m
        return best or MODEL_REGISTRY[0]

    def _detect_vision(self, query: str) -> bool:
        return any(w in query.lower() for w in
                  ["image", "picture", "photo", "see", "visual", "diagram",
                   "graph", "what is in", "describe this", "look at",
                   "screenshot", "vision", "mmproj"])

    def _detect_reasoning(self, query: str) -> bool:
        return any(w in query.lower() for w in
                  ["prove", "analyze in depth", "compare and contrast",
                   "philosophical", "why is", "how does", "implications",
                   "reason step by step"])

    def switch_worker(self, model_key: str, use_vision: bool = False) -> dict:
        """Kill current worker and load a new model. Returns status."""
        model = None
        for m in MODEL_REGISTRY:
            if m["key"] == model_key:
                model = m
                break
        if not model:
            return {"status": "error", "message": f"Unknown model: {model_key}"}

        model_path = MODELS_DIR / model["file"]
        if not model_path.exists():
            return {"status": "error", "message": f"File not found: {model['file']}"}

        self.kill_worker()
        time.sleep(2)

        # Build command
        cmd = [str(LLAMA_SERVER), "-m", str(model_path), "-ngl", "99", "-c", "2048",
               "--port", str(WORKER_PORT), "--host", "127.0.0.1"]

        # Add mmproj if vision requested and available
        if use_vision and model.get("mmproj"):
            mmproj_path = MODELS_DIR / model["mmproj"]
            if mmproj_path.exists():
                cmd += ["--mmproj", str(mmproj_path)]
                self._thinking_mode = False  # Vision models: thinking OFF
            else:
                return {"status": "error", "message": f"mmproj not found: {model['mmproj']}"}

        # Thinking config
        if model["thinking"] == "on":
            cmd += ["--reasoning-format", "deepseek"]
            self._thinking_mode = True
        else:
            self._thinking_mode = False

        # Start
        print(f"[MARP] Loading {model['name']}...")
        self._worker_process = subprocess.Popen(
            cmd, cwd=str(LLAMA_DIR),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Wait for ready
        max_wait = 60 if model["size_gb"] > 15 else 20
        for i in range(max_wait):
            if self._check_health(WORKER_PORT):
                self._current_worker = model_key
                vram = self._get_vram()
                return {"status": "ok", "model": model["name"],
                        "thinking": self._thinking_mode,
                        "vision": use_vision,
                        "load_time_s": i+1,
                        "vram_mb": vram}

            time.sleep(1)

        return {"status": "timeout", "message": f"Failed to load {model['name']} in {max_wait}s"}

    def auto_switch(self, query: str) -> dict:
        """Auto-detect needs and switch to best model."""
        needs_vision = self._detect_vision(query)
        needs_reasoning = self._detect_reasoning(query)

        # Classify
        classification = self.classify(query)
        domains = classification["domains"]

        # Select model
        model = self._find_model(domains, needs_vision, needs_reasoning)

        # Check if already loaded
        if self._current_worker == model["key"]:
            return {"status": "already_loaded", "model": model["name"],
                    "domains": domains, "classification": classification}

        # Switch
        result = self.switch_worker(model["key"], use_vision=needs_vision)
        result["domains"] = domains
        result["classification"] = classification
        return result

    def query_worker(self, prompt: str, system: str = "") -> dict:
        """Send a query to the current worker. Returns {response, latency_ms}."""
        if not self._check_health(WORKER_PORT):
            return {"response": "[ERROR] No worker loaded", "latency_ms": -1}

        current = None
        for m in MODEL_REGISTRY:
            if m["key"] == self._current_worker:
                current = m
                break

        model_name = current["name"] if current else "unknown"
        t0 = time.perf_counter()

        try:
            # Gemini format
            if current and "Gemma" in model_name:
                url = f"http://127.0.0.1:{WORKER_PORT}/completion"
                gemma_prompt = f"<bos><start_of_turn>user\n{system} {prompt}<end_of_turn>\n<start_of_turn>model\n"
                payload = {"prompt": gemma_prompt, "max_tokens": 200, "temperature": 0.3,
                          "stop": ["<end_of_turn>", "<start_of_turn>"]}
            # GLM format
            elif current and "GLM" in model_name:
                url = f"http://127.0.0.1:{WORKER_PORT}/completion"
                glm_prompt = f"<|system|>{system}<|user|>{prompt}<|assistant|>"
                payload = {"prompt": glm_prompt, "max_tokens": 200, "temperature": 0.3}
            # Qwen: chat completions
            else:
                url = f"http://127.0.0.1:{WORKER_PORT}/v1/chat/completions"
                messages = [{"role": "user", "content": f"{system} {prompt}".strip()}]
                if system:
                    messages.insert(0, {"role": "system", "content": system})
                payload = {"messages": messages, "max_tokens": 200, "temperature": 0.3}
                if not self._thinking_mode:
                    payload["chat_template_kwargs"] = {"enable_thinking": False}

            timeout = 120 if self._thinking_mode else 30
            r = requests.post(url, json=payload, timeout=timeout)
            elapsed = (time.perf_counter() - t0) * 1000

            if "completion" in url:
                raw = r.json()["content"]
            else:
                raw = r.json()["choices"][0]["message"]["content"]

            # Strip thinking tags
            cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if not cleaned:
                cleaned = raw[:300]

            return {"response": cleaned, "latency_ms": round(elapsed, 1)}

        except Exception as e:
            return {"response": f"[ERROR] Query failed: {e}", "latency_ms": -1}

    def status(self) -> dict:
        """Full pipeline status."""
        router_ok = self._check_health(ROUTER_PORT)
        worker_ok = self._check_health(WORKER_PORT)
        vram = self._get_vram()
        current_model = None
        for m in MODEL_REGISTRY:
            if m["key"] == self._current_worker:
                current_model = m["name"]
                break
        return {
            "router": {"port": ROUTER_PORT, "alive": router_ok},
            "worker": {"port": WORKER_PORT, "alive": worker_ok,
                      "model": current_model or "none",
                      "thinking": self._thinking_mode},
            "vram_mb": vram,
            "available_models": [m["key"] for m in MODEL_REGISTRY],
        }

    def _get_vram(self) -> int:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5)
            return int(r.stdout.strip().split()[0])
        except:
            return -1


# ─── MCP Server ─────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MARP Orchestrator", dependencies=["requests"])

server = MARPServer()


@mcp.tool(description="[MARP] Process a query: router classifies + auto-selects best model + answers")
def marp_query(query: str, system: str = "") -> str:
    """
    Process any query through the MARP pipeline.
    
    1. Router (Qwen0.8B) classifies the domain
    2. Auto-selects best model for that domain (quality-first)
    3. Loads model with correct mmproj/thinking config
    4. Queries worker and returns answer
    
    Args:
        query: The user's question or request
        system: Optional system prompt context
    """
    # 1. Ensure router
    if not server.ensure_router():
        return "[ERROR] Router failed to start on port 8084"

    # 2. Auto-switch to best model
    switch_result = server.auto_switch(query)
    
    # 3. If error switching, report
    if switch_result.get("status") == "error":
        return f"[ERROR] {switch_result.get('message')}"
    if switch_result.get("status") == "timeout":
        return f"[ERROR] {switch_result.get('message')}"

    # 4. Query worker
    if system:
        system_prompt = f"Answer as a {','.join(switch_result.get('domains', ['general']))} expert. {system}"
    else:
        system_prompt = f"Answer as a {','.join(switch_result.get('domains', ['general']))} expert."

    result = server.query_worker(query, system=system_prompt)

    if result["latency_ms"] < 0:
        return f"[ERROR] Worker failed: {result['response']}"

    # 5. Build response
    model_name = switch_result.get("model", "unknown")
    domains = switch_result.get("domains", ["general"])
    thinking = " [thinking ON]" if server._thinking_mode else ""
    
    return f"[{model_name}{thinking} | {', '.join(domains)} | {result['latency_ms']:.0f}ms]\n{result['response']}"


@mcp.tool(description="[MARP] Switch the worker model manually")
def marp_switch(model_key: str, vision: bool = False) -> str:
    """
    Manually switch to a specific model.
    
    Args:
        model_key: One of: gemma-31b, glm-flash, qwen-moe, qwen-omni, qwen-reasoning, qwen-9b
        vision: Load with mmproj for vision tasks (only for models that support it)
    """
    result = server.switch_worker(model_key, use_vision=vision)
    if result["status"] == "ok":
        return f"Loaded {result['model']} (thinking={'ON' if result.get('thinking') else 'OFF'}, vision={'ON' if vision else 'OFF'}, {result.get('load_time_s','?')}s)"
    return f"Error: {result.get('message', 'Unknown error')}"


@mcp.tool(description="[MARP] Current pipeline status")
def marp_status() -> str:
    """Router health, current worker model, VRAM usage, available models."""
    s = server.status()
    lines = [
        f"MARP Pipeline Status",
        f"  Router (:8084): {'✅ Alive' if s['router']['alive'] else '❌ Dead'}",
        f"  Worker (:8082): {'✅ Alive' if s['worker']['alive'] else '❌ Dead'}",
        f"  Model:          {s['worker']['model']}",
        f"  Thinking:       {'ON' if s['worker']['thinking'] else 'OFF'}",
        f"  VRAM:           {s['vram_mb']} MB / 24576 MB ({s['vram_mb']*100//24576}%)",
        f"  Available:      {', '.join(s['available_models'])}",
    ]
    return "\n".join(lines)


@mcp.tool(description="[MARP] Process an image with vision model")
def marp_vision(image_path: str, prompt: str = "Describe this image") -> str:
    """
    Process an image using the Qwen Omni vision model.
    Auto-loads the model with mmproj if not already loaded.
    
    Args:
        image_path: Path to the image file
        prompt: Question about the image
    """
    # Auto-switch to vision mode
    result = server.switch_worker("qwen-omni", use_vision=True)
    if result["status"] != "ok":
        return f"[ERROR] Failed to load vision model: {result.get('message')}"
    
    # Query with image
    p = Path(image_path)
    if not p.exists():
        return f"[ERROR] Image not found: {image_path}"
    
    return server.query_worker(
        f"[Image: {str(p)}] {prompt}",
        system="You are a vision expert. Describe what you see."
    )["response"]


@mcp.tool(description="[MARP] List available models with specs")
def marp_models() -> str:
    """List all available worker models, their domains, thinking support, and quality."""
    lines = []
    lines.append(f"{'Key':20s} {'Name':30s} {'Type':8s} {'Thinking':10s} {'Quality':8s} {'Domains':30s}")
    lines.append("-" * 110)
    for m in MODEL_REGISTRY:
        thinking = m["thinking"]
        doms = ", ".join(m["domains"][:4])
        if len(m["domains"]) > 4:
            doms += "..."
        lines.append(f"{m['key']:20s} {m['name']:30s} {m['type']:8s} {thinking:10s} {'★'*m['quality']:8s} {doms:30s}")
    return "\n".join(lines)


# ─── Startup helper ─────────────────────────────────────────────
def startup():
    """Called on boot. Ensures router is running."""
    print("[MARP] Starting router...")
    s = MARPServer()
    if s.ensure_router():
        print(f"[MARP] Router OK on :{ROUTER_PORT}")
    else:
        print(f"[MARP] Router FAILED")


# ─── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MARP MCP Server")
    parser.add_argument("--startup", action="store_true", help="Start router only (for boot)")
    parser.add_argument("--query", type=str, help="Quick query without MCP")
    parser.add_argument("--status", action="store_true", help="Show status")
    args = parser.parse_args()

    if args.startup:
        startup()
    elif args.query:
        # Standalone mode
        s = MARPServer()
        s.ensure_router()
        switch = s.auto_switch(args.query)
        if switch["status"] in ("ok", "already_loaded"):
            doms = switch.get("domains", ["general"])
            result = s.query_worker(args.query, system=f"Answer as a {','.join(doms)} expert.")
            print(result["response"])
        else:
            print(f"Error: {switch}")
    elif args.status:
        s = MARPServer()
        print(json.dumps(s.status(), indent=2))
    else:
        # MCP mode: register and run
        print("[MARP] Starting MCP server...")
        s = MARPServer()
        s.ensure_router()
        mcp.run()
