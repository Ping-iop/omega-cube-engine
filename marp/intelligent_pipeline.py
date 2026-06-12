#!/usr/bin/env python3
"""
MARP Intelligent Pipeline v2 — Entry point.
Router + Worker con thinking dinámico por dominio.
"""
import requests, time, re, json, sys
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────
RURL = "http://127.0.0.1:8084/v1/chat/completions"
WURL = "http://127.0.0.1:8082/v1/chat/completions"

REASONING_DOMAINS = {"math", "science", "philosophy", "analysis", "engineering", "code"}
SPEED_DOMAINS = {"language", "business", "general", "gaming", "law", "medical"}

# Queries cortas/simples → nunca thinking aunque el dominio lo pida
SIMPLE_PATTERNS = re.compile(
    r'^(what is\s+\d+\s*[+\-*/]\s*\d+|'
    r'hi|hello|hey|thanks|thank you|good morning|good night|'
    r'who are you|what can you do|'
    r'translate\s+\w+\s+to\s+\w+|'
    r'define\s+\w+|'
    r'what\'?s?\s+the\s+(capital|population|weather|time|date)|'
    r'[?]\s*$|'
    r'^\w+\s+\w+\s*$)',
    re.IGNORECASE
)

ROUTER_PROMPT_HEAD = """Classify each query into EXACTLY one domain.
Domains: math, code, science, engineering, language, law, medical, business, philosophy, gaming

Examples:
query: derivative of x squared
domain: math
query: python async function
domain: code
query: quantum entanglement explained
domain: science
query: bridge structural design
domain: engineering
query: translate hello to spanish
domain: language
query: NDA contract terms
domain: law
query: diabetes symptoms treatment
domain: medical
query: NPV calculation startup
domain: business
query: free will determinism philosophy
domain: philosophy
query: roguelike game design
domain: gaming

Now classify this query and output ONLY the domain name:"""

def query_router(query: str) -> tuple[str, float]:
    """Query Qwen0.8B router → returns (domain, latency_ms)."""
    full_prompt = ROUTER_PROMPT_HEAD + f"\nquery: {query}\ndomain:"
    t0 = time.perf_counter()
    r = requests.post(RURL, json={
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": 5, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}
    }, timeout=15)
    lat = (time.perf_counter() - t0) * 1000
    raw = (r.json()["choices"][0]["message"]["content"] or "").strip().lower().split()[0] if r.json()["choices"][0]["message"]["content"] else "general"
    valid = REASONING_DOMAINS | SPEED_DOMAINS
    return raw if raw in valid else "general", lat

def query_worker(prompt: str, domain: str, system: str = "", max_tokens: int = 200) -> tuple[str, float, dict]:
    # Detectar queries simples → nunca thinking
    is_simple = bool(SIMPLE_PATTERNS.search(prompt.strip()))
    needs_thinking = domain in REASONING_DOMAINS and not is_simple
    system = system or f"Eres un experto en {domain}. Responde de forma {'concisa y directa' if not needs_thinking else 'detallada paso a paso.'}"
    
    payload = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    if needs_thinking:
        payload["max_tokens"] = min(max_tokens * 3, 4096)
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        payload["max_tokens"] = max_tokens
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    t0 = time.perf_counter()
    r = requests.post(WURL, json=payload, timeout=120)
    lat = (time.perf_counter() - t0) * 1000
    data = r.json()
    msg = data["choices"][0]["message"]
    finish = data["choices"][0]["finish_reason"]

    raw_content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""

    if needs_thinking:
        if not raw_content.strip() and finish == "length":
            # Thinking no terminó → fallback: re-query sin thinking
            fallback_payload = {**payload, "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": max_tokens}
            # No re-measured, just get answer
            try:
                fb = requests.post(WURL, json=fallback_payload, timeout=60)
                fb_data = fb.json()
                fb_msg = fb_data["choices"][0]["message"]
                clean = re.sub(r'<think>.*?</think>', '', (fb_msg.get("content") or ""), flags=re.DOTALL).strip()
                note = f"🔄 fallback (thinking={reasoning[:50]}…→direct)"
            except:
                clean = reasoning
                note = f"⚠️ thinking incomplete, fallback failed"
        else:
            clean = raw_content
            note = f"✅ thinking done ({data['usage']['total_tokens']} tok)"
    else:
        clean = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        note = f"⚡ fast ({data['usage']['total_tokens']} tok)"

    return clean, lat, {
        "domain": domain, "thinking": needs_thinking,
        "latency_ms": round(lat), "tokens": data["usage"]["total_tokens"],
        "finish": finish, "note": note, "model": data.get("model", "?"),
    }

def process_query(query: str, system: str = "", max_tokens: int = 200) -> dict:
    t0 = time.perf_counter()
    try:
        domain, rt = query_router(query)
    except Exception as e:
        return {"error": f"Router: {e}", "query": query}
    try:
        answer, wt, meta = query_worker(query, domain, system, max_tokens)
    except Exception as e:
        return {"error": f"Worker: {e}", "domain": domain, "query": query}
    total = (time.perf_counter() - t0) * 1000
    return {
        "query": query, "domain": domain, "thinking": meta["thinking"],
        "answer": answer, "router_ms": round(rt), "worker_ms": round(wt),
        "total_ms": round(total), "tokens": meta["tokens"],
        "note": meta["note"], "model": meta["model"], "finish": meta["finish"],
    }

def print_result(r: dict):
    if "error" in r:
        print(f"  ❌ {r['error']}"); return
    icon = "🧠" if r["thinking"] else "⚡"
    print(f"  📡 Router: {r['domain']:<12s} ({r['router_ms']}ms)")
    print(f"  {icon} Worker:  {r['worker_ms']}ms ({r['tokens']} tok) {r['note']}")
    print(f"  💬 {r['answer'][:250]}")
    if len(r['answer']) > 250: print(f"     ...({len(r['answer'])} chars)")
    print(f"  ⏱  Total: {r['total_ms']}ms [{r['model']}]")

# ─── Tests ────────────────────────────────────────────────────────
TETS = [
    ("derivative of sin(x)*cos(x) step by step", "math"),
    ("Python async function with error handling", "code"),
    ("explain quantum entanglement", "science"),
    ("suspension bridge key structural elements", "engineering"),
    ("translate good morning to Japanese", "language"),
    ("key elements of an NDA agreement", "law"),
    ("common symptoms of type 2 diabetes", "medical"),
    ("calculate NPV for a startup 5yr projection", "business"),
    ("is free will compatible with determinism?", "philosophy"),
    ("design a roguelike death penalty system", "gaming"),
]

def cmd_test():
    print(f"\n  ⚡ Quick test — 3 queries\n")
    for q in ["What is 2+2?", "Derivative of x^2 using chain rule", "Translate hello to spanish"]:
        r = process_query(q, max_tokens=50)
        icon = "🧠" if r.get("thinking") else "⚡"
        print(f"  {icon} {q[:35]:35s} → {r.get('domain','?'):<10s} {r.get('total_ms',0):>5}ms  \"{r.get('answer','')[:40]}\"")
    print()

def cmd_benchmark():
    print(f"\n{'='*65}")
    print(f"  BENCHMARK: {len(TETS)} queries — thinking ON for {sorted(REASONING_DOMAINS)}")
    print(f"{'='*65}\n")
    results = []
    for i, (q, exp) in enumerate(TETS):
        print(f"  [{i+1}/{len(TETS)}] {q[:45]:45s}", end=" ", flush=True)
        r = process_query(q, max_tokens=150)
        if "error" in r:
            print(f"❌ {r['error']}")
        else:
            ok = "✅" if r['domain'] == exp else "❌"
            icon = "🧠" if r['thinking'] else "⚡"
            print(f"{ok} {r['domain']:<10s} {icon} {r['total_ms']:>5}ms {r['tokens']:>4}tok")
        results.append(r)
    correct = sum(1 for i, r in enumerate(results) if "error" not in r and r['domain'] == TETS[i][1])
    think_n = sum(1 for r in results if "error" not in r and r['thinking'])
    valid = [r for r in results if "error" not in r]
    avg_ms = sum(r['total_ms'] for r in valid) / len(valid) if valid else 0
    avg_tok = sum(r['tokens'] for r in valid) / len(valid) if valid else 0
    print(f"\n{'─'*65}")
    print(f"  📊 Router acc: {correct}/{len(TETS)} ({correct*100//len(TETS)}%)")
    print(f"  🧠 Thinking:   {think_n}/{len(valid)} queries")
    print(f"  ⚡ Avg:        {avg_ms:.0f}ms, {avg_tok:.0f} tok")
    print(f"{'─'*65}\n")

# ─── Interactive ──────────────────────────────────────────────────
def cmd_interactive():
    print("\n" + "=" * 65)
    print("  MARP Intelligent Pipeline v2")
    print("  Router: Qwen0.8B  |  Worker: Qwen 27B Omni")
    print(f"  🧠 Thinking ON:  {', '.join(sorted(REASONING_DOMAINS))}")
    print(f"  ⚡ Thinking OFF: {', '.join(sorted(SPEED_DOMAINS))}")
    print("=" * 65)
    print("  Commands: /quit  /test  /benchmark")
    print()
    while True:
        try:
            q = input("  >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!"); break
        if not q: continue
        if q in ("/quit", "/exit", "q"): break
        if q == "/test": cmd_test(); continue
        if q == "/benchmark": cmd_benchmark(); continue
        print()
        print_result(process_query(q, max_tokens=200))
        print()

# ─── CLI Dispatch ─────────────────────────────────────────────────
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--benchmark" or (__import__('shutil').which('true') and False)]
    
    if "--benchmark" in sys.argv or "-b" in sys.argv:
        cmd_benchmark()
    elif "--test" in sys.argv or "-t" in sys.argv:
        cmd_test()
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        r = process_query(" ".join(sys.argv[1:]))
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        cmd_interactive()
