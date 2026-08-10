#!/usr/bin/env python3
"""
MARP MCP Server — Simplified version for port 8084 (Worker) and 8091 (Support).

Decoupled worker routing: The user starts whatever model they want on port 8084 manually.
This server provides support tools (search/index/tag) using the 0.8B support model on port 8091.
"""

import sys, json, os, time, re, subprocess, shutil
from pathlib import Path
from typing import Optional
import requests

# ─── Paths ───────────────────────────────────────────────────────
HOME = Path.home()
LLAMA_DIR = Path("C:/Users/GPAMD/Downloads/Llama.cpp Cuda/llama-b9045-bin-win-cuda-13.1-x64")
LLAMA_SERVER = LLAMA_DIR / "llama-server.exe"
SUPPORT_MODEL_PATH = Path("P:/AI_INFRA/custom_models/Qwen/Qwen3.5-0.8B-Q6_K.gguf")

SUPPORT_PORT = 8091
WORKER_PORT = 8084

# ─── Prompts for Support Model (0.8B) ───────────────────────────
SUPPORT_CLASSIFY_PROMPT = """Clasifica el siguiente texto en categorias internas de Omega Cube.
Categories: knowledge, task, idea, code, reference, personal, project, research, other.
Output ONLY: category. One word."""

SUPPORT_TAG_PROMPT = """Extrae hasta 5 tags clave del siguiente texto.
Output ONLY: tag1, tag2, tag3 (comma-separated, lowercase, no explanations)."""

SUPPORT_SUMMARIZE_PROMPT = """Resume el siguiente texto en una sola oracion concisa (maximo 20 palabras).
Output ONLY the summary sentence, nothing else."""

class MARPServer:
    """Manages support model (8091) instance and checks worker (8084) status."""

    def __init__(self):
        pass

    def _check_health(self, port: int) -> bool:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            return r.json().get("status") == "ok"
        except:
            return False

    def ensure_support(self) -> bool:
        """Ensure support model (Qwen3.5-0.8B) is running on :8091.
        Used for internal Axioma-Omega tasks.
        """
        if self._check_health(SUPPORT_PORT):
            return True
        if not SUPPORT_MODEL_PATH.exists():
            return False
        subprocess.Popen(
            [str(LLAMA_SERVER), "-m", str(SUPPORT_MODEL_PATH),
             "-ngl", "99", "-c", "512", "--port", str(SUPPORT_PORT),
             "--host", "127.0.0.1", "--log-disable"],
            cwd=str(LLAMA_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        for _ in range(20):
            if self._check_health(SUPPORT_PORT):
                return True
            time.sleep(1)
        return False

    def _support_query(self, system: str, user_text: str, max_tokens: int = 20) -> str:
        """Internal query to the support model. Returns raw text."""
        if not self.ensure_support():
            return ""
        try:
            r = requests.post(
                f"http://127.0.0.1:{SUPPORT_PORT}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_text[:500]}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}
                }, timeout=15
            )
            return r.json()["choices"][0]["message"]["content"].strip()
        except:
            return ""

    def classify_document(self, text: str) -> dict:
        """Classify a document/note via support model for internal indexing."""
        t0 = time.perf_counter()
        category = self._support_query(SUPPORT_CLASSIFY_PROMPT, text, max_tokens=5) or "other"
        tags_raw = self._support_query(SUPPORT_TAG_PROMPT, text, max_tokens=30)
        summary = self._support_query(SUPPORT_SUMMARIZE_PROMPT, text, max_tokens=30)
        elapsed = (time.perf_counter() - t0) * 1000
        tags = [t.strip().lower() for t in tags_raw.split(",") if t.strip()][:5]
        return {
            "category": category.lower().strip(),
            "tags": tags,
            "summary": summary,
            "latency_ms": round(elapsed, 1)
        }

    def _get_vram(self) -> int:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5)
            return int(r.stdout.strip().split()[0])
        except:
            return -1

    def status(self) -> dict:
        support_ok = self._check_health(SUPPORT_PORT)
        worker_ok = self._check_health(WORKER_PORT)
        vram = self._get_vram()
        
        # Detect active model name from health endpoint of 8084 if possible
        model_name = "unknown"
        if worker_ok:
            try:
                # Intenta obtener el alias o nombre cargado en el worker
                r = requests.get(f"http://127.0.0.1:{WORKER_PORT}/slots", timeout=2)
                slots = r.json()
                if isinstance(slots, list) and len(slots) > 0:
                    model_name = slots[0].get("model", "llama-cpp-model")
            except:
                model_name = "active-model"

        return {
            "support": {"port": SUPPORT_PORT, "alive": support_ok, "model": "Qwen3.5-0.8B"},
            "worker": {"port": WORKER_PORT, "alive": worker_ok, "model": model_name},
            "vram_mb": vram
        }

# ─── MCP Server ─────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MARP Orchestrator", dependencies=["requests"])
server = MARPServer()

@mcp.tool(description="[MARP] Current pipeline status")
def marp_status() -> str:
    """Checks port status (8084 worker and 8091 support) and VRAM."""
    s = server.status()
    vram = s['vram_mb']
    vram_pct = vram * 100 // 24576 if vram > 0 else 0
    lines = [
        f"MARP Pipeline Status",
        f"  Support (:8091): {'✅ Alive' if s['support']['alive'] else '❌ Dead'} — Qwen3.5-0.8B (internal support)",
        f"  Worker  (:8084): {'✅ Alive' if s['worker']['alive'] else '❌ Dead'} — Model: {s['worker']['model']}",
        f"  VRAM:            {vram} MB / 24576 MB ({vram_pct}%)",
    ]
    return "\n".join(lines)

@mcp.tool(description="[Omega] Classify + tag + summarize a document/note for Omega Cube indexing")
def omega_index_document(text: str) -> str:
    """
    Uses Qwen3.5-0.8B (support model) to classify, tag, and summarize a text for indexing.
    Auto-starts the support model if not running.
    """
    result = server.classify_document(text)
    if not result["category"]:
        return "[ERROR] Support model unavailable. Start it with omega_start_support."
    return (
        f"Category: {result['category']}\n"
        f"Tags:     {', '.join(result['tags']) or 'none'}\n"
        f"Summary:  {result['summary']}\n"
        f"Latency:  {result['latency_ms']:.0f}ms"
    )

@mcp.tool(description="[Omega] Start/verify Qwen3.5-0.8B support model on port 8091")
def omega_start_support() -> str:
    """
    Ensures the Qwen3.5-0.8B support model is running on port 8091.
    Used for Axioma-Omega internal tasks (search, indexing, tagging).
    """
    if server.ensure_support():
        return "✅ Support model (Qwen3.5-0.8B) is running on port 8091."
    return f"❌ Support model failed to start. Check path: {str(SUPPORT_MODEL_PATH)}"

@mcp.tool(description="[Omega] Quick semantic search helper using support model")
def omega_search_classify(query: str) -> str:
    """
    Uses the support model to generate search tags and category for a query.
    """
    category = server._support_query(SUPPORT_CLASSIFY_PROMPT, query, max_tokens=5) or "other"
    tags_raw = server._support_query(SUPPORT_TAG_PROMPT, query, max_tokens=30)
    tags = [t.strip().lower() for t in tags_raw.split(",") if t.strip()][:5]
    return f"category: {category}\ntags: {', '.join(tags)}"

def startup():
    """Called on boot. Ensures support model is running."""
    print("[MARP] Starting support model (Qwen3.5-0.8B on :8091)...")
    s = MARPServer()
    if s.ensure_support():
        print(f"[MARP] Support model OK on :8091")
    else:
        print(f"[MARP] Support model FAILED — check path: {SUPPORT_MODEL_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MARP MCP Server")
    parser.add_argument("--startup", action="store_true", help="Start support model (for boot)")
    parser.add_argument("--status", action="store_true", help="Show status")
    args = parser.parse_args()

    if args.startup:
        startup()
    elif args.status:
        s = MARPServer()
        print(json.dumps(s.status(), indent=2))
    else:
        print("[MARP] Starting MCP server (support services only)...")
        s = MARPServer()
        s.ensure_support()
        mcp.run()


LLAMA_DIR = Path("C:/Users/GPAMD/Downloads/Llama.cpp Cuda/llama-b9045-bin-win-cuda-13.1-x64")
