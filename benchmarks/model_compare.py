"""Benchmark multiple worker models for MARP pipeline.
Tests: Gemma 4 31B, Qwen3.5 9B Q8, Qwen3.5 27B Reasoning, Qwen3.6 35B MoE.
Each model loaded on worker (8082), 3 test queries, timed.
Router (8084) stays running with Qwen0.8B throughout.
"""
import requests, time, re, subprocess, json, sys
from pathlib import Path

LLAMA_DIR = Path("C:/Users/GPAMD/Downloads/LLAMA~1.CPP/llama-b9045-bin-win-cuda-13.1-x64")
LLAMA_SERVER = LLAMA_DIR / "llama-server.exe"

MODELS = {
    "gemma-4-31B-it-Q4_K_M": {
        "path": "J:/modelos_ia/gemma-4-31B-it-Q4_K_M.gguf",
        "size_gb": 18, "desc": "Gemma 4 31B Q4 (no thinking)"
    },
    "Qwen3.5-9B-Q8_0": {
        "path": "J:/modelos_ia/Qwen3.5-9B-Q8_0.gguf",
        "size_gb": 9, "desc": "Qwen3.5 9B Q8 (dense, high quality)"
    },
    "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.i1-Q4_K_M": {
        "path": "J:/modelos_ia/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.i1-Q4_K_M.gguf",
        "size_gb": 16, "desc": "Qwen3.5 27B Reasoning Distilled (thinking?)"
    },
}

QUERIES = [
    ("What is the derivative of x squared?", "math"),
    ("Write a Python async HTTP function", "code"),
    ("Explain CRISPR gene editing", "science"),
]

def wait_for_server(port=8082, timeout=45):
    for i in range(timeout):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.json().get("status") == "ok":
                return True
        except:
            pass
        if i % 3 == 0:
            print(f"  Waiting... ({i}s)")
        time.sleep(1)
    return False

def kill_server():
    subprocess.run(["taskkill.exe", "//F", "//IM", "llama-server.exe"], 
                   capture_output=True, timeout=5)
    time.sleep(3)

def test_worker(name, model_path, desc, use_chat=False, use_reasoning=None):
    print(f"\n{'='*60}")
    print(f"  TESTING: {name}")
    print(f"  {desc}")
    print(f"  File: {Path(model_path).name} ({Path(model_path).stat().st_size/1e9:.1f}GB)")
    print(f"{'='*60}")
    
    # Start server
    cmd = [str(LLAMA_SERVER), "-m", model_path, "-ngl", "99", "-c", "1024", 
           "--port", "8082", "--host", "127.0.0.1"]
    if use_reasoning:
        cmd += ["--reasoning-format", use_reasoning]
    
    proc = subprocess.Popen(cmd, cwd=str(LLAMA_DIR), 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not wait_for_server():
        print("  ❌ FAILED TO LOAD")
        proc.kill()
        return None
    
    # Get VRAM
    try:
        vram = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except:
        vram = "N/A"
    
    time.sleep(2)  # settle
    
    # Test queries
    results = []
    total_time = 0
    for q, exp_domain in QUERIES[:2]:  # Just 2 queries per model
        t0 = time.perf_counter()
        try:
            if use_chat or "gemma" in name.lower():
                # Chat completions
                url = f"http://127.0.0.1:8082/v1/chat/completions"
                payload = {
                    "messages": [
                        {"role": "system", "content": f"Answer as a {exp_domain} expert."},
                        {"role": "user", "content": q}
                    ],
                    "max_tokens": 50, "temperature": 0,
                }
                if "gemma" not in name.lower():
                    payload["chat_template_kwargs"] = {"enable_thinking": False}
                r = requests.post(url, json=payload, timeout=60)
                content = r.json()["choices"][0]["message"]["content"]
            else:
                # Completion endpoint
                url = f"http://127.0.0.1:8082/completion"
                prompt = f"<|im_start|>system\nAnswer as a {exp_domain} expert.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
                if "gemma" in name.lower():
                    prompt = f"<start_of_turn>user\nAnswer as a {exp_domain} expert: {q}<end_of_turn>\n<start_of_turn>model\n"
                r = requests.post(url, json={"prompt": prompt, "max_tokens": 50, "temperature": 0}, timeout=60)
                content = r.json()["content"]
            
            elapsed = (time.perf_counter() - t0) * 1000
            cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if not cleaned:
                cleaned = content[:80]
            total_time += elapsed
            results.append((q, elapsed, cleaned[:80]))
            print(f"  {elapsed:6.0f}ms [{cleaned[:60]}]")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((q, -1, f"ERROR: {e}"))
    
    # VRAM after
    try:
        vram2 = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except:
        vram2 = "N/A"
    
    # Kill server
    proc.kill()
    proc.wait()
    time.sleep(2)
    
    # Wait for VRAM release
    try:
        vram_after = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except:
        vram_after = "N/A"
    
    avg = total_time / max(len([r for r in results if r[1] > 0]), 1)
    
    print(f"  Avg: {avg:.0f}ms | VRAM: {vram} → {vram_after}")
    
    return {
        "name": name,
        "desc": desc,
        "avg_latency_ms": round(avg, 0),
        "vram_before": vram,
        "vram_after": vram_after,
        "results": results,
    }


if __name__ == "__main__":
    print("MARP MODEL COMPARISON BENCHMARK")
    print("Router: Qwen0.8B (8084, stays running)")
    print("Testing workers on 8082...")
    print()
    
    all_results = {}
    
    for name, model_info in MODELS.items():
        result = test_worker(
            name=name,
            model_path=model_info["path"],
            desc=model_info["desc"],
        )
        if result:
            all_results[name] = result
    
    print(f"\n\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Modelo':40s} {'Latencia':>10s} {'VRAM':>12s}")
    print(f"{'-'*62}")
    for name, result in sorted(all_results.items(), key=lambda x: x[1]["avg_latency_ms"]):
        print(f"{result['desc']:40s} {result['avg_latency_ms']:>6.0f}ms {result['vram_after']:>12s}")
