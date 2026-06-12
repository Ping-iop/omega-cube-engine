#!/usr/bin/env python3
"""
MARP + Omega-Cube + Qwen 3.6 27B Omni INTEGRATION TEST
Prueba el pipeline completo con Qwen Omni como worker.
"""
import requests, time, re, json, sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes/axioma-omega-protocol"))
from omega_cube.engine import OmegaCubeEngine

RURL = "http://127.0.0.1:8084/v1/chat/completions"
WURL = "http://127.0.0.1:8082/v1/chat/completions"  # Qwen uses chat completions!

ROUTER_PROMPT = """Map queries to EXACT domains from: math,code,science,engineering,language,law,medical,business,philosophy,gaming,general.
Output ONLY: domain or domain1,domain2."""

print("=" * 72)
print("  MARP + OMEGA-CUBE + QWEN 3.6 27B OMNI")
print("  Router: Qwen0.8B (:8084)  |  Worker: Qwen 27B Omni (:8082)")
print("=" * 72)

e = OmegaCubeEngine()
domain_knowledge = {
    "math": ["calculus derivatives", "linear algebra matrices"],
    "code": ["python async await", "docker container deploy"],
    "science": ["quantum entanglement", "crispr gene editing"],
    "engineering": ["bridge structural load", "pid controller"],
    "law": ["nda confidentiality", "patent intellectual property"],
    "medical": ["diabetes diagnosis", "cancer immunotherapy"],
    "business": ["npv net present value", "startup funding"],
    "philosophy": ["free will determinism", "kant ethics"],
    "gaming": ["rpg game design", "mmorpg balance"],
    "language": ["spanish translation", "linguistics phonetics"],
}
for domain, topics in domain_knowledge.items():
    for topic in topics:
        e.add_node(content=topic, hierarchies=[f"{domain}.{topic[:12]}"], tensor_position=[0.85, 0.5])

print(f"\n📦 Omega-Cube: {e.stats()['total_nodes']} nodos\n")

tests = [
    ("derivative of x squared",     "math",        "Explain the power rule for derivatives briefly"),
    ("Python async function",        "code",        "Give an example of Python async/await"),
    ("quantum entanglement",        "science",     "Explain quantum entanglement simply"),
    ("Docker deploy to K8s",        "code",        "How to deploy a container to Kubernetes"),
    ("NDA agreement draft",         "law",         "Key elements of an NDA"),
    ("diabetes type 2 symptoms",    "medical",     "Common diabetes symptoms"),
    ("NPV calculation startup",     "business",    "How to calculate net present value"),
    ("free will and determinism",   "philosophy",  "Is free will compatible with determinism?"),
    ("roguelike death mechanic",    "gaming",      "Design a death penalty for a roguelike"),
    ("translate hello to spanish",  "language",    "Translate: hello, how are you?"),
    ("suspension bridge design",    "engineering", "Key structural elements of a suspension bridge"),
]

corr = 0; rtimes, wtimes, ptimes = [], [], []

for i, (query, expected_domain, followup) in enumerate(tests):
    print(f"\n─── Test {i+1}/{len(tests)}: \"{query}\" ───")
    t0 = time.perf_counter()

    # 1. ROUTER
    try:
        r1 = requests.post(RURL, json={
            "messages": [{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": query}],
            "max_tokens": 10, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}
        }, timeout=10)
        rt = (time.perf_counter() - t0) * 1000
        raw = r1.json()["choices"][0]["message"]["content"].strip().lower()
        doms = [d.strip() for d in raw.replace("domains:", "").replace("->", ",").split(",")[:2]]
        valid = {"math","code","science","engineering","language","law","medical","business","philosophy","gaming","general"}
        doms = [d for d in doms if d in valid] or ["general"]
        hit = expected_domain in doms
        if hit: corr += 1
        rtimes.append(rt)
        print(f"  📡 Router: {str(doms):20s} {'✅' if hit else '❌'} ({rt:.0f}ms)")
    except Exception as ex:
        print(f"  📡 Router: ❌ {ex}")
        doms = ["general"]; rt = 0; rtimes.append(0)

    # 2. OMEGA-CUBE
    ctx_nodes = e.query(doms[0], mode="tensor", top_k=2)
    ct = (time.perf_counter() - t0) * 1000 - rt
    ctx_str = "; ".join([n.get("content", "")[:40] for n in ctx_nodes[:2]])
    print(f"  🧊 Omega-Cube: {len(ctx_nodes)} nodos ({ct:.0f}ms)")

    # 3. WORKER (Qwen 27B Omni via /v1/chat/completions)
    try:
        system = f"Eres un experto en {doms[0]}. Responde de forma concisa.\nContexto: {ctx_str[:200]}"
        r2 = requests.post(WURL, json={
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": followup}],
            "max_tokens": 80, "temperature": 0.3,
            "chat_template_kwargs": {"enable_thinking": False}
        }, timeout=30)
        wt = (time.perf_counter() - t0) * 1000 - rt - ct
        ans = r2.json()["choices"][0]["message"]["content"]
        wtimes.append(wt)
        print(f"  ⚡ Qwen Omni: {wt:.0f}ms → \"{ans[:60].strip()}{'...' if len(ans)>60 else ''}\"")
    except Exception as ex:
        print(f"  ⚡ Qwen Omni: ❌ {ex}")
        wt = 0; wtimes.append(0)

    total = rt + ct + wt
    ptimes.append(total)
    print(f"  ⏱  Pipeline: {total:.0f}ms")

# Summary
print("\n" + "=" * 72)
print("  📊 RESULTADOS: MARP + OMEGA-CUBE + QWEN 3.6 27B OMNI")
print("=" * 72)
print(f"\n  🎯 Router Accuracy:  {corr}/{len(tests)} ({corr*100//len(tests)}%)")
print(f"  📡 Router avg:       {sum(rtimes)/len(rtimes):.0f}ms")
print(f"  ⚡ Qwen Omni avg:    {sum(wtimes)/len(wtimes):.0f}ms")
print(f"  ⏱  Pipeline avg:    {sum(ptimes)/len(ptimes):.0f}ms")
print(f"\n  📦 Omega-Cube:       {e.stats()['total_nodes']} nodos, {e.stats()['avg_retrieval_time_ms']:.1f}ms retrieval")
print(f"\n  🔧 Router:           Qwen0.8B (:8084) ~700MB VRAM")
print(f"  🔧 Worker:           Qwen 3.6 27B Omni (:8082) ~15GB VRAM")
print(f"  💾 VRAM total:       ~18/24.6 GB (73%)")
print("=" * 72)
