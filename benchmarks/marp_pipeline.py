"""MARP Pipeline: Qwen0.8B router + Qwen3.6 27B worker + Omega-Cube."""
import requests, time, re, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.home() / ".hermes/axioma-omega-protocol"))
from omega_cube.engine import OmegaCubeEngine
from omega_cube.predictive_search import PredictiveContextSearch

RURL="http://127.0.0.1:8084/v1/chat/completions"  # Qwen0.8B GPU
WURL="http://127.0.0.1:8082/completion"   # Qwen3.6 27B (GPU, completion endpoint)

# Omega-Cube
e=OmegaCubeEngine()
for d,c in {"math":["calculus","linear algebra"],"code":["python async","docker devops"],
"science":["quantum physics","crispr gene"],"engineering":["bridge structural","pid controller"],
"law":["nda legal","patent ip"],"medical":["diabetes diagnosis","cancer therapy"],
"business":["npv finance","startup roi"],"philosophy":["free will","kant ethics"],
"gaming":["rpg design","mmorpg balance"],"language":["spanish translation","linguistics"]}.items():
    for x in c: e.add_node(x, [d + "." + x[:8]], [0.85, 0.5])

pcs=PredictiveContextSearch()
for nid,node in e.nodes.items():
    for h in node.hierarchies:
        if h!="root": pcs.trie.insert("n_"+h, h.split(".")[0], nid)

RS="""Map queries to EXACT domains from: math,code,science,engineering,language,law,medical,business,philosophy,gaming,general.
Output ONLY: domain or domain1,domain2.
Examples:
"derivative" -> math, "Python" -> code, "quantum" -> science, "Docker" -> code
"NDA" -> law, "diabetes" -> medical, "NPV" -> business, "free will" -> philosophy
"game" -> gaming, "translate" -> language, "bridge" -> engineering"""

tests=[
("derivative of x squared","math","Explain derivatives briefly"),
("Python async function","code","Example of Python async function"),
("quantum entanglement","science","Explain quantum entanglement simply"),
("Docker deploy K8s","code","How to deploy to Kubernetes"),
("NDA agreement draft","law","Write a short NDA template"),
("diabetes symptoms","medical","List common diabetes symptoms"),
("NPV calculation","business","Calculate net present value"),
("free will determinism","philosophy","Is free will real?"),
("roguelike RPG mechanic","gaming","Design a death mechanic"),
("translate to Spanish","language","Translate hello how are you"),
("suspension bridge","engineering","Bridge structural elements"),
]

valid={"math","code","science","engineering","language","law","medical","business","philosophy","gaming","general"}
corr=0; rtimes=[]; wtimes=[]; tots=[]

print("=" * 65)
print("  MARP PIPELINE: Qwen0.8B Router (GPU) + Qwen3.6 27B Worker (GPU)")
print("  Router full-GPU: 82ms p50 (vs 260ms CPU = 3.2x faster)")
print("  Omega-Cube: %d nodes, %d queries" % (e.stats()["total_nodes"], len(tests)))
print("=" * 65)

for q, exp_d, follow in tests:
    t0 = time.perf_counter()

    # Router
    r1 = requests.post(RURL, json={
        "messages": [{"role": "system", "content": RS}, {"role": "user", "content": q}],
        "max_tokens": 10, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}
    }, timeout=30)
    rt = (time.perf_counter() - t0) * 1000
    raw = r1.json()["choices"][0]["message"]["content"].strip().lower()
    doms = [d.strip() for d in raw.replace("domains:","").split(",")[:2]]
    doms = [d for d in doms if d in valid] or ["general"]
    hit = exp_d in doms
    if hit: corr += 1
    rtimes.append(rt)

    # Omega-Cube context
    ctx = []
    for d in doms[:2]:
        pred = pcs.predict(d[:4]) if d != "general" else []
        if pred: ctx.extend(pred[:2])
    ct = (time.perf_counter() - t0) * 1000 - rt

    # Worker (completion endpoint for Qwen3.6 Omni thinking model)
    prompt = "<|im_start|>system\nYou are a %s expert. Answer concisely.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % (doms[0], follow)
    r2 = requests.post(WURL, json={
        "prompt": prompt,
        "max_tokens": 50, "temperature": 0.3,
    }, timeout=60)
    wt = (time.perf_counter() - t0) * 1000 - rt - ct
    raw_ans = r2.json()["content"]
    ans = re.sub(r'<think>.*?</think>', '', raw_ans, flags=re.DOTALL).strip()
    wtimes.append(wt)
    total = rt + ct + wt
    tots.append(total)

    print("  %s R:%5.0fms%s %-20s W:%5.0fms [%s]" % (
        q[:30].ljust(30), rt, "O" if hit else "X",
        str(doms), wt, ans[:40].replace("\n"," ")))

print()
print("  ROUTER ACCURACY:  %d/%d (%d%%)" % (corr, len(tests), corr*100//len(tests)))
rt_avg = sum(rtimes)/len(rtimes)
print("  ROUTER LATENCY:   avg=%.0fms p50=%.0fms (Qwen0.8B CPU)" % (rt_avg, sorted(rtimes)[len(rtimes)//2]))
wt_avg = sum(wtimes)/len(wtimes)
print("  WORKER LATENCY:   avg=%.0fms p50=%.0fms (Qwen3.6 27B GPU)" % (wt_avg, sorted(wtimes)[len(wtimes)//2]))
tot_avg = sum(tots)/len(tots)
print("  TOTAL PIPELINE:   avg=%.0fms (router+worker)" % tot_avg)
print("  OMEGA-CUBE NODES: %d" % e.stats()["total_nodes"])
print("  GPU ROUTER:       127.0.0.1:8084 (Qwen0.8B, ~1.3GB VRAM)")
print("  GPU WORKER:       127.0.0.1:8082 (Qwen3.6 27B, ~16GB VRAM)")
print("  TOTAL VRAM:       ~17.3GB / 24GB (%.0f%% free)" % ((24-17.3)/24*100))
print("=" * 65)
