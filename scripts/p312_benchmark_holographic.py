"""P3.12 — Micro-benchmark del binding holografico (convolucion circular).

Criterio predefinido (MEJORAS.md P3.12): similitud coseno > 0.95 al deshacer
el enganche, es decir unbind(bind(a, b), a) ~= b.

Metrica principal: encoder.similarity() (mapeo (cos+1)/2 -> [0,1]), la misma
que usa produccion via partial_match(). Se reporta tambien el coseno puro
para trazabilidad (umbral 0.95 mapeado == 0.90 puro).

Escenarios:
  A. Vectores gaussianos unitarios (caso ideal de Plate, cota superior).
  B. Vectores reales de encode_node() (feature hashing, textos tecnicos ES/EN,
     jerarquias reales del grafo) -- condicion de produccion (vectores dispersos).
  C. Firma completa: partial_match(padre, firma) positivo + control negativo
     con vector no relacionado (informativo).

Dims: 256 (default vieja) y 2048 (actual post-F8). 200 pruebas por celda.
Salida: tabla en stdout + scripts/p312_results.json para archivo.
"""

import json
import statistics
import sys
import time

import numpy as np

DEPLOY = r"C:\Users\GPAMD\.hermes\axioma-omega-protocol"
sys.path.insert(0, DEPLOY)

from omega_cube.holographic import HolographicEncoder

TRIALS = 200
SEED = 7

VOCAB = (
    "engine cache embedding holographic binding convolution vector graph "
    "node axiom protocol router worker scheduler orchestrator gate validity "
    "motor caché incrustación holográfico convolución grafo nodo axioma "
    "protocolo enrutador planificador orquestador validez cadena color tono "
    "saturación profundidad dominio física energía conservación diseño sistema "
    "retrieval semantic fallback degraded orphan hierarchy domain tools dev "
    "evony alliance hermes skill cron monitor latency throughput benchmark"
).split()

HIERARCHIES = [
    "DEV.TOOLS", "DEV.ARCHITECTURE", "EVONY.ALLIANCE.MGMT", "HERMES.SKILLS",
    "PHYSICS.MECH", "MATH.LINEAR_ALGEBRA", "PROYECTO.AXIOMA", "CHEM.THERMO",
]


def rand_text(rng, nmin=6, nmax=14):
    n = int(rng.integers(nmin, nmax + 1))
    idx = rng.integers(0, len(VOCAB), size=n)
    return " ".join(VOCAB[i] for i in idx)


def raw_cos(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    p05 = xs[max(0, int(0.05 * n) - 1)]
    return {
        "mean": round(statistics.fmean(xs), 4),
        "p05": round(p05, 4),
        "min": round(xs[0], 4),
        "max": round(xs[-1], 4),
    }


def run_dim(dim, trials=TRIALS, seed=SEED):
    enc = HolographicEncoder(dimension=dim, seed=seed)
    rng = np.random.default_rng(seed)

    A_raw, A_map = [], []
    B_raw, B_map = [], []
    C_pos, C_neg = [], []
    t_bind = []
    t_unbind = []

    for _ in range(trials):
        # --- Escenario A: gaussianas unitarias ---
        a = rng.standard_normal(dim)
        a /= np.linalg.norm(a)
        b = rng.standard_normal(dim)
        b /= np.linalg.norm(b)

        t0 = time.perf_counter()
        bound = enc.bind(a.tolist(), b.tolist())
        t_bind.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        rec = enc.unbind(bound, a.tolist())
        t_unbind.append((time.perf_counter() - t0) * 1000)

        A_raw.append(raw_cos(rec, b))
        A_map.append(enc.similarity(rec, b))

        # --- Escenario B: vectores reales encode_node ---
        ha, hb = str(rng.choice(HIERARCHIES)), str(rng.choice(HIERARCHIES))
        ta, tb = rand_text(rng), rand_text(rng)
        va = enc.encode_node(ta, ha)
        vb = enc.encode_node(tb, hb)

        bound_b = enc.bind(va, vb)
        rec_b = enc.unbind(bound_b, va)
        B_raw.append(raw_cos(rec_b, vb))
        B_map.append(enc.similarity(rec_b, vb))

        # --- Escenario C: firma completa (informativo) ---
        kids = [(rand_text(rng), str(rng.choice(HIERARCHIES))) for _ in range(3)]
        nbrs = [(rand_text(rng), str(rng.choice(HIERARCHIES))) for _ in range(2)]
        sig = enc.encode_holographic_signature(
            ta, ha, parent_content=tb, parent_hierarchy=hb,
            children=kids, neighbors=nbrs,
        )
        C_pos.append(enc.similarity(va, sig))          # propio nodo dentro de su firma
        unrelated = enc.encode_node(rand_text(rng), str(rng.choice(HIERARCHIES)))
        C_neg.append(enc.similarity(unrelated, sig))   # control negativo

    return {
        "dim": dim,
        "trials": trials,
        "A_ideal_bind_unbind": {"raw_cos": stats(A_raw), "mapped_similarity": stats(A_map)},
        "B_real_encode_node": {"raw_cos": stats(B_raw), "mapped_similarity": stats(B_map)},
        "C_partial_match_vs_firma": {
            "positivo_nodo_propio": stats(C_pos),
            "control_negativo": stats(C_neg),
        },
        "timing_ms": {
            "bind_mean": round(statistics.fmean(t_bind), 3),
            "unbind_mean": round(statistics.fmean(t_unbind), 3),
        },
        "_series": {
            "A_mapped": A_map, "B_mapped": B_map,
            "C_pos": C_pos, "C_neg": C_neg,
        },
    }


def main():
    results = [run_dim(256), run_dim(2048)]
    series = {r.pop("_series")[0]: r for r in []}  # placeholder no-op
    out_path = DEPLOY + r"\scripts\p312_results.json"

    payload = []
    series_store = {}
    for r in results:
        series_store[r["dim"]] = r.pop("_series")
        payload.append(r)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"date": "2026-08-25", "criterion": "mapped similarity > 0.95",
                   "results": payload, "series": series_store},
                  f, indent=1)

    print("=== P3.12 Micro-benchmark binding holografico ===")
    print(f"criterio: encoder.similarity() > 0.95 (= coseno puro > 0.90)")
    print(f"pruebas por celda: {TRIALS}\n")

    def line(r, scen_key, scen_name):
        s_raw = r[scen_key]["raw_cos"]
        s_map = r[scen_key]["mapped_similarity"]
        verdict = "PASS" if s_map["mean"] > 0.95 else "FAIL"
        print(f"[dim {r['dim']:>4}] {scen_name}")
        print(f"    mapped : mean={s_map['mean']:.4f}  p05={s_map['p05']:.4f}  min={s_map['min']:.4f}  max={s_map['max']:.4f}")
        print(f"    raw cos: mean={s_raw['mean']:.4f}  p05={s_raw['p05']:.4f}  min={s_raw['min']:.4f}")
        print(f"    veredicto media: {verdict}")

    for r in results:
        print("-" * 56)
        line(r, "A_ideal_bind_unbind", "A. ideal (gaussianas unitarias)")
        line(r, "B_real_encode_node", "B. real (encode_node feature hashing)")
        cp = r["C_partial_match_vs_firma"]["positivo_nodo_propio"]
        cn = r["C_partial_match_vs_firma"]["control_negativo"]
        print(f"[dim {r['dim']:>4}] C. partial_match nodo propio vs firma: mean={cp['mean']:.4f} min={cp['min']:.4f}")
        print(f"[dim {r['dim']:>4}]    control negativo (no relacionado):     mean={cn['mean']:.4f} max={cn['max']:.4f}")
        tm = r["timing_ms"]
        print(f"[dim {r['dim']:>4}] timing: bind={tm['bind_mean']}ms  unbind={tm['unbind_mean']}ms")

    overall_B = all(
        r["B_real_encode_node"]["mapped_similarity"]["mean"] > 0.95 for r in results
    )
    print("\nDECISION (criterio sobre escenario B real, dim 2048 produccion):")
    r2048 = next(r for r in results if r["dim"] == 2048)
    mB = r2048["B_real_encode_node"]["mapped_similarity"]["mean"]
    print(f"  B@2048 mapped mean = {mB:.4f} -> {'CUMPLE (>0.95)' if mB > 0.95 else 'NO CUMPLE (<=0.95)'}")
    print(f"resultados archivados en: {out_path}")


if __name__ == "__main__":
    main()
