"""
F8-ablación — Descomponer el rendimiento de cube_move en sus fases.

Brazos:
  plano-coseno   : referencia (mismo embedding holográfico)
  cm-2D-solo     : fase 2D únicamente (tau=0.05, sin giros 3D)
  cm-full        : cube_move completo 2D+3D (tau=0.05)
  cm-full-tau0   : cube_move completo sin umbral (tau=0.0)

Lecturas:
  cm-2D-solo ~= plano  -> fase 2D sana; el daño viene de los giros 3D.
  cm-full-tau0 > cm-full -> parte del daño es el umbral tau sobre cosenos hash.

Uso: python f8_ablation.py [--dim 2048]
"""
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f8_battle as fb

from omega_cube.cube_move import CubeMover          # noqa: E402
from omega_cube.engine_v2 import OmegaCubeEngineV2   # noqa: E402


def main() -> None:
    dim = int(sys.argv[sys.argv.index("--dim") + 1]) if "--dim" in sys.argv else 2048
    rng = random.Random(fb.SEED)
    np.random.seed(fb.SEED)

    print(f"[1/3] Corpus sintético (dim={dim})...", flush=True)
    specs, edges = fb.build_corpus(rng)
    questions = fb.build_questions(rng)

    engine = OmegaCubeEngineV2(auto_load=False, holographic_dim=dim)
    lid_map = {}
    for lid, content, hier in specs:
        node = engine.add_node(content, [hier])
        lid_map[lid] = node.node_id
    seen_edges = set()
    for a, b in edges:
        key = frozenset((a, b))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        engine.associate(lid_map[a], lid_map[b])
    print(f"      nodos={len(engine.nodes)}  aristas={len(seen_edges)}  "
          f"preguntas={len(questions)}", flush=True)

    mover = CubeMover(engine)
    flat = fb.arm_flat(engine, {})

    def cm(query, k, tau, expand3d):
        r = mover.cube_move(query, k=k, tau=tau, expand_3d=expand3d)
        return [(n["node_id"], n.get("score", 0))
                for n in (r.nodes_2d + r.nodes_3d)]

    arms = {
        "plano-coseno": lambda q: flat(q, fb.K),
        "cm-2D-solo":   lambda q: cm(q, fb.K, 0.05, False),
        "cm-full":      lambda q: cm(q, fb.K, 0.05, True),
        "cm-full-tau0": lambda q: cm(q, fb.K, 0.0, True),
    }

    print(f"[2/3] Ablación ({len(questions)} preguntas x {len(arms)} brazos)...",
          flush=True)
    acc = {name: defaultdict(list) for name in arms}
    for qi, qq in enumerate(questions):
        target_nid = lid_map[qq["target"]]
        for name, fn in arms.items():
            t = time.time()
            ranked = fn(qq["query"])
            dt = (time.time() - t) * 1000
            p, r, f1, mrr = fb.prf(ranked, target_nid)
            cell = acc[name]
            cell["p"].append(p); cell["r"].append(r); cell["f1"].append(f1)
            cell["mrr"].append(mrr); cell["ms"].append(dt)
            cell[f"type:{qq['type']}"].append(f1)
        if (qi + 1) % 60 == 0:
            print(f"      {qi+1}/{len(questions)}...", flush=True)

    summary = {}
    for name, cell in acc.items():
        mean = lambda key: sum(cell[key]) / len(cell[key])  # noqa: E731
        summary[name] = {
            "precision_at_k": round(mean("p"), 4),
            "recall_at_k": round(mean("r"), 4),
            "f1_at_k": round(mean("f1"), 4),
            "mrr": round(mean("mrr"), 4),
            "latency_ms_mean": round(mean("ms"), 1),
            "f1_by_type": {k.split(":", 1)[1]: round(sum(v) / len(v), 4)
                           for k, v in cell.items() if k.startswith("type:")},
        }

    print("[3/3] Resultados:", flush=True)
    for name, s in summary.items():
        print(f"  {name:14s} F1@5={s['f1_at_k']:.3f}  R@5={s['recall_at_k']:.3f}  "
              f"MRR={s['mrr']:.3f}  {s['latency_ms_mean']:.1f} ms/q")
        for t, f1v in s["f1_by_type"].items():
            print(f"      {t:22s} F1={f1v:.3f}")

    out_dir = fb.ROOT / "docs" / "pruebas"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")
    path = out_dir / f"f8_ablation_{stamp}_d{dim}.json"
    path.write_text(json_dumps(summary, dim), encoding="utf-8")
    print(f"OK -> {path.name}", flush=True)


def json_dumps(summary, dim):
    import json
    return json.dumps({"date": time.strftime("%Y-%m-%d"), "seed": fb.SEED,
                       "k": fb.K, "dim": dim, "summary": summary},
                      ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
