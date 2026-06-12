#!/usr/bin/env python3
"""
MARP + Omega-Cube INTEGRATION TEST
Prueba el pipeline completo: Router Qwen0.8B → Omega-Cube → Worker GLM-4.7-Flash
TODO con servidores ya existentes (no mata nada).
"""
import requests, time, re, json, sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes/axioma-omega-protocol"))
from omega_cube.engine import OmegaCubeEngine

# ─── Config ───────────────────────────────────────────────────────
RURL = "http://127.0.0.1:8084/v1/chat/completions"   # Router Qwen0.8B
WURL = "http://127.0.0.1:8082/completion"             # Worker GLM-4.7-Flash
ROUTER_PROMPT = """Map queries to EXACT domains from: math,code,science,engineering,language,law,medical,business,philosophy,gaming,general.
Output ONLY: domain or domain1,domain2.
Examples:
"derivative" -> math, "Python" -> code, "quantum" -> science, "Docker" -> code
"NDA" -> law, "diabetes" -> medical, "NPV" -> business, "free will" -> philosophy
"game" -> gaming, "translate" -> language, "bridge" -> engineering"""

# ─── Omega-Cube Init ──────────────────────────────────────────────
print("=" * 72)
print("  MARP + OMEGA-CUBE INTEGRATION TEST")
print("  Router: Qwen0.8B (:8084)  |  Worker: GLM-4.7-Flash (:8082)")
print("=" * 72)

e = OmegaCubeEngine()

# Poblar Omega-Cube con conocimiento de dominios
domain_knowledge = {
    "math": ["calculus derivatives", "linear algebra matrices", "integral calculus"],
    "code": ["python async await", "docker container deploy", "git version control"],
    "science": ["quantum entanglement", "crispr gene editing", "cellular biology"],
    "engineering": ["bridge structural load", "pid controller feedback", "circuit design"],
    "law": ["nda confidentiality agreement", "patent intellectual property", "contract law"],
    "medical": ["diabetes type2 diagnosis", "cancer immunotherapy", "cardiology"],
    "business": ["npv net present value", "startup funding roi", "market analysis"],
    "philosophy": ["free will determinism", "kant categorical imperative", "existentialism"],
    "gaming": ["rpg game design", "mmorpg balance mechanics", "roguelike procedural"],
    "language": ["spanish translation", "linguistics phonetics", "japanese grammar"],
    "general": ["general knowledge question"],
}

for domain, topics in domain_knowledge.items():
    for topic in topics:
        e.add_node(
            content=topic,
            hierarchies=[f"{domain}.{topic[:12]}"],
            tensor_position=[0.85, 0.5]
        )

print(f"\n📦 Omega-Cube: {e.stats()['total_nodes']} nodos cargados")
print()

# ─── Test Queries ──────────────────────────────────────────────────
tests = [
    ("derivative of x squared",     "math",        "Explain the power rule for derivatives briefly"),
    ("Python async function",        "code",        "Give an example of Python async/await"),
    ("quantum entanglement",        "science",     "Explain quantum entanglement simply"),
    ("Docker deploy to K8s",        "code",        "How to deploy a container to Kubernetes"),
    ("NDA agreement draft",         "law",         "List key elements of an NDA"),
    ("diabetes type 2 symptoms",    "medical",     "List common diabetes symptoms"),
    ("NPV calculation startup",     "business",    "How to calculate net present value"),
    ("free will and determinism",   "philosophy",  "Is free will compatible with determinism?"),
    ("roguelike death mechanic",    "gaming",      "Design a death penalty system for a roguelike"),
    ("translate hello to spanish",  "language",    "Translate: hello, how are you?"),
    ("suspension bridge design",    "engineering", "Key structural elements of a suspension bridge"),
]

results = {"router_correct": 0, "router_accurate": [], "worker_times": [], "pipeline_times": []}

for i, (query, expected_domain, followup) in enumerate(tests):
    print(f"\n─── Test {i+1}/{len(tests)}: \"{query}\" ───")
    t_start = time.perf_counter()

    # 1. ROUTER: clasificar dominio
    try:
        r1 = requests.post(RURL, json={
            "messages": [
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": query}
            ],
            "max_tokens": 10,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}
        }, timeout=10)
        rt_ms = (time.perf_counter() - t_start) * 1000
        raw = r1.json()["choices"][0]["message"]["content"].strip().lower()
        domains = [d.strip() for d in raw.replace("domains:", "").replace("->", ",").split(",")[:2]]
        valid = {"math","code","science","engineering","language","law","medical","business","philosophy","gaming","general"}
        domains = [d for d in domains if d in valid] or ["general"]
        hit = expected_domain in domains
        if hit:
            results["router_correct"] += 1
        results["router_accurate"].append(hit)
        print(f"  📡 Router: {raw[:20]:20s} → {str(domains):20s} {'✅' if hit else '❌'} ({rt_ms:.0f}ms)")
    except Exception as ex:
        print(f"  📡 Router: ❌ ERROR - {ex}")
        domains = ["general"]
        rt_ms = 0

    # 2. OMEGA-CUBE: contexto relevante
    ctx_nodes = e.query(domains[0], mode="tensor", top_k=2)
    ctx_ms = (time.perf_counter() - t_start) * 1000 - rt_ms
    ctx_str = "; ".join([n.get("content", "") for n in ctx_nodes[:2]]) if ctx_nodes else ""
    print(f"  🧊 Omega-Cube: {len(ctx_nodes)} nodos relevantes ({ctx_ms:.0f}ms)")

    # 3. WORKER: generar respuesta (GLM format)
    try:
        glm_prompt = f"<|system|>Eres un experto en {domains[0]}. Responde de forma concisa.\nContexto: {ctx_str[:200]}<|user|>{followup}<|assistant|>"
        r2 = requests.post(WURL, json={
            "prompt": glm_prompt,
            "max_tokens": 80,
            "temperature": 0.3,
        }, timeout=30)
        wt_ms = (time.perf_counter() - t_start) * 1000 - rt_ms - ctx_ms
        ans = r2.json().get("content", "")
        # Limpiar think tags si existen
        ans = re.sub(r'<think>.*?</think>', '', ans, flags=re.DOTALL).strip()
        # Limpiar backslashes
        ans = ans.replace("\\n", "\n").strip()
        results["worker_times"].append(wt_ms)
        print(f"  ⚡ Worker: {wt_ms:.0f}ms → \"{ans[:80].strip()}{'...' if len(ans)>80 else ''}\"")
    except Exception as ex:
        print(f"  ⚡ Worker: ❌ ERROR - {ex}")
        ans = ""
        wt_ms = 0

    total_ms = rt_ms + ctx_ms + wt_ms
    results["pipeline_times"].append(total_ms)
    print(f"  ⏱  Pipeline total: {total_ms:.0f}ms")

# ─── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  📊 RESUMEN INTEGRACIÓN MARP + OMEGA-CUBE + GLM")
print("=" * 72)

router_acc = results["router_correct"] / len(tests) * 100
router_lat = sum(results["router_accurate"])  # actually this is bool list, skip
rt_list = []
# Re-calcular latencias de router
print(f"\n  🎯 Router Accuracy: {results['router_correct']}/{len(tests)} ({router_acc:.0f}%)")
print(f"  ⚡ Worker avg:       {sum(results['worker_times'])/len(results['worker_times']):.0f}ms" if results['worker_times'] else "  ⚡ Worker: N/A")
print(f"  ⏱  Pipeline avg:     {sum(results['pipeline_times'])/len(results['pipeline_times']):.0f}ms")
print(f"\n  📦 Omega-Cube stats: {json.dumps(e.stats(), indent=2)}")
print(f"\n  🔧 Modelos activos:  Qwen0.8B router (:8084) + GLM-4.7-Flash (:8082)")
print(f"  💾 VRAM:             ~22.7/24.6 GB (92%) — liberar para Qwen 27B")
print("=" * 72)
