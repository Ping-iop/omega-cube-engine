"""
FINAL BENCHMARK: MARP Router + H-Bit Spectrum (local execution).
Runs what actually works here, generates data for PDF charts.
"""
import json, time, sys
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "C:/Users/GPAMD/Documents/GEMINI/DESARROLLO_APPS/H-Bit/src")

@dataclass
class R: name:str; metric:str; value:float; unit:str; trials:int; source:str="LOCAL"

results = []

# === 1. MARP Router ===
from omega_cube.marp import MARPRouter
from omega_cube.marp.protocol import ShardConfig, MARPMode

router = MARPRouter()
shards = [ShardConfig(name=f'{d}_v1',domains=[d],mode=MARPMode.WRAPPER,
          base_model='gemma-4-31b',gpu_memory_mb=4000)
          for d in ['math','code','science','engineering','language','law','medical','business','philosophy','gaming']]

# Latency
qs = [f"Query {i} about math and science topics" for i in range(100)]
ts = []
for q in qs:
    t0 = time.perf_counter()
    router.route(q, shards)
    ts.append((time.perf_counter()-t0)*1000)
ts.sort()

results.append(R("MARP Router","avg_latency_ms",round(np.mean(ts),3),"ms",100))
results.append(R("MARP Router","p50_latency_ms",round(ts[50],3),"ms",100))
results.append(R("MARP Router","p99_latency_ms",round(ts[99],3),"ms",100))
results.append(R("MARP Router","queries_per_sec",round(1000/np.mean(ts),0),"q/s",100))

# Domain accuracy
test_qs = [
    ("Prove the Riemann hypothesis","math"),("Write a Python async scraper","code"),
    ("Explain quantum entanglement","science"),("Design a bridge truss","engineering"),
    ("Draft an NDA clause","law"),("Diagnose pneumonia symptoms","medical"),
    ("Calculate NPV with discount rate","business"),("Design roguelike death mechanic","gaming"),
    ("Compare Kant vs utilitarianism","philosophy"),("Translate poem to Spanish","language"),
    ("Explain cross-entropy in ML","math"),("Optimize PostgreSQL query","code"),
    ("What causes seasons","science"),("Calculate beam deflection","engineering"),
    ("Write non-compete clause","law"),("Symptoms of diabetes","medical"),
    ("ROI calculation SaaS","business"),("Free will vs determinism","philosophy"),
    ("Best programming language 2026","code"),("Derivative of x squared","math"),
    ("How CRISPR works","science"),("Patent filing","law"),
    ("Startup valuation","business"),("Explain photosynthesis","science"),
    ("Write Rust function","code"),
]
hits = 0
sv = []
for q,exp in test_qs:
    d = router.route(q, shards)
    if exp in d.ticket.active_domains: hits += 1
    sv.append(d.token_savings_estimate)
acc = hits/len(test_qs)
results.append(R("MARP Router","domain_accuracy_25q",round(acc,3),"ratio",25))
results.append(R("MARP Router","avg_token_savings",round(np.mean(sv),3),"ratio",25))

# === 2. H-Bit Spectrum ===
try:
    from hbit.core.crypto import generate_key_pair
    from hbit.universal import UniversalEncoder
    from hbit.analysis.spectrum import SpectrumVerifier
    from hbit.formats.base import MediaRegistry

    rng=np.random.default_rng(42)
    img=Image.fromarray(rng.integers(0,256,(512,512,3),dtype=np.uint8))
    tp=Path("/tmp/hb_orig.png"); img.save(tp)
    kp=generate_key_pair()
    sp=Path("/tmp/hb_signed.png")
    UniversalEncoder(use_kdf=False).encode(tp,kp,sp)

    MediaRegistry.reset()
    vf=SpectrumVerifier()

    t0=time.perf_counter()
    fr=vf.analyze(sp)
    st=(time.perf_counter()-t0)*1000

    results.append(R("H-Bit Spectrum","full_confidence",round(fr.confidence,3),"ratio",1))
    results.append(R("H-Bit Spectrum","full_tiles",float(fr.payloads_valid),"tiles",1))
    results.append(R("H-Bit Spectrum","analysis_time_ms",round(st,1),"ms",1))

    idata=np.array(Image.open(sp))
    for pct in [0.25,0.12,0.06,0.03]:
        rows=max(1,int(idata.shape[0]*pct))
        cd=idata[:rows,:,:]
        cp=Path(f"/tmp/hb_c{int(pct*100)}.png")
        Image.fromarray(cd).save(cp)
        MediaRegistry.reset()
        r=vf.analyze(cp)
        if r.has_evidence:
            results.append(R("H-Bit Spectrum",f"crop_{int(pct*100)}pct_conf",round(r.confidence,3),"ratio",1))
            results.append(R("H-Bit Spectrum",f"crop_{int(pct*100)}pct_tiles",float(r.payloads_valid),"tiles",1))
except Exception as e:
    print(f"H-Bit: {e}")

# === SAVE ===
out = Path(__file__).parent / "final_benchmark_data.json"
out.write_text(json.dumps([asdict(r) for r in results], indent=2))
print(f"Done: {len(results)} measurements")
for r in results:
    print(f"  {r.name:30s} {r.metric:30s} = {r.value:>10} {r.unit}")
print(f"Saved: {out}")
