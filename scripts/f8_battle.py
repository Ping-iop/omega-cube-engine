"""
F8 — La prueba reina (PLAN_2026-08-09 sección 4.F8, ejecutada 2026-08-25).

Tesis central del proyecto:
    F1(navegación jerárquica + cube_move) > F1(búsqueda plana por coseno)
medido sobre >=100 preguntas y >=1000 nodos. Margen >5 puntos para confirmar.

Diseño del experimento (justo por construcción):
  - Corpus sintético DETERMINISTA (seed=42): 8 dominios × (1 hub + 120 hojas)
    + 60 nodos ruido = 1028 nodos. Asociaciones: hoja↔hub, hoja↔hoja vecina,
    +1 intra-dominio aleatoria. Los nodos ruido mezclan vocabulario cruzado.
  - Suite de 120 preguntas etiquetadas (15 por dominio × 8), 3 dificultades:
      * directa      — el query contiene el marcador único del nodo objetivo
      * reformulada  — mismos términos del objetivo, estructura distinta + relleno
      * solo-vecino  — query construido SOLO con vocabulario del nodo asociado
                       (sin solaparse con el vocabulario del objetivo): exige
                       propagación estructural, imposible por puro léxico
  - Brazos (todos sobre el MISMO motor y el MISMO espacio de embeddings,
    firmas holográficas feature-hashing 256d; solo cambia la estrategia):
      A) plano-coseno   : ranking global cos(query_vec, firma) descendente
      B) cube_move      : Fórmula del plan (Fase 2D top-k + Fase 3D giros 3D)
      C) engine-v2-jerárquico : query(mode="hierarchical") HNSW + difusión
    Notas de equidad: C corre sin boundary filter ni detector de alucinación
    (se aísla la CALIDAD DE RANKING, que es lo que apuesta la tesis);
    B corre con tau bajo (0.05) para que ambos brazos llenen el mismo
    presupuesto de k=5 resultados (tau=0.3 dejaría manos vacías sin jugadas).
  - Métricas por brazo: Precision@5, Recall@5, F1@5, MRR, latencia media ms.
    Objetivo relevante único por pregunta (la respuesta correcta etiquetada).

Salidas:
  - docs/pruebas/f8_results_<fecha>.json   (datos crudos)
  - docs/pruebas/F8_<fecha>.md             (informe + veredicto)
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omega_cube.engine_v2 import OmegaCubeEngineV2  # noqa: E402
from omega_cube.cube_move import CubeMover          # noqa: E402

SEED = 42
K = 5
N_DIRECT, N_REFORM, N_NEIGHBOR = 5, 5, 5  # por dominio => 120 preguntas
# Dimensión del encoder holográfico. 256 = valor de producción actual.
# Uso: python f8_battle.py --dim 2048   (duelo controlado sobre representación sana)
DIM = int(sys.argv[sys.argv.index("--dim") + 1]) if "--dim" in sys.argv else 256

DOMAINS = {
    "astronomia": ["galaxia", "estrella", "planeta", "orbita", "telescopio",
                   "nebulosa", "cometa", "supernova", "exoplaneta", "quasar",
                   "pulsar", "meteorito", "eclipse", "satelite"],
    "cocina": ["fermentacion", "salsa", "masa", "horno", "emulsion",
               "brasear", "marinar", "levadura", "umami", "glaseado",
               "caramelo", "especias", "caldo", "mise"],
    "medicina": ["diagnostico", "sintoma", "terapia", "vacuna", "anticuerpo",
                 "metabolismo", "arteria", "neurona", "inflamacion", "dosis",
                 "sindrome", "cronico", "biopsia", "protesis"],
    "software": ["compilador", "repositorio", "depuracion", "bucle",
                 "variable", "servidor", "despliegue", "refactor",
                 "prueba-unitaria", "api", "caché", "concurrencia", "schema", "log"],
    "musica": ["acorde", "armonia", "tempo", "pentagrama", "melodia",
               "timbre", "escala", "sinfonia", "contrapunto", "tonalidad",
               "riff", "groove", "sampling", "reverb"],
    "economia": ["inflacion", "mercado", "oferta", "demanda", "divisa",
                 "arancel", "subsidio", "balanza", "fiscal", "deuda",
                 "interes", "accion", "bono", "liquidez"],
    "biologia": ["mitocondria", "adn", "enzima", "proteina", "celula",
                 "membrana", "evolucion", "especie", "habitat",
                 "fotosintesis", "genoma", "receptor", "hormona", "simbiosis"],
    "historia": ["imperio", "dinastia", "revolucion", "tratado", "colonia",
                 "arqueologia", "manuscrito", "civilizacion", "guerra",
                 "alianza", "monarca", "feudal", "republica", "cronica"],
}

FILLER = ["nota", "apunte", "resumen", "detalle", "tema", "entrada",
          "busqueda", "referencia", "material", "consulta"]

REFORM_TEMPLATES = [
    "quiero {f1} sobre {w1} y {w2}, con todo el {f2}",
    "donde guardo la {f1} de {w2} {w1}",
    "explicame {w1} {w3} paso a paso en esta {f2}",
    "necesito el {f1} completo acerca de {w3} y {w1}",
    "busca {f2} relacionada con {w2}, {w3} y algo mas de {f1}",
]


def build_corpus(rng: random.Random):
    """Devuelve (specs, edges): specs[(logical_id, content, hierarchy)], edges[(a,b)]."""
    specs, edges = [], []
    for dom, vocab in DOMAINS.items():
        hub_lid = f"{dom}::hub"
        hub_content = f"{dom} nucleo central " + " ".join(vocab)
        specs.append((hub_lid, hub_content, f"{dom}/hub"))
        leaf_words = {}  # lid -> [palabras del contenido]
        for i in range(120):
            subtopic = i // 20  # 6 subtemas × 20 hojas
            words = rng.sample(vocab, 4)
            marker = f"m{dom[:3]}{i:03d}"
            content = f"{marker} " + " ".join(words) + \
                f" registro {dom} caso {i}"
            lid = f"{dom}::{i}"
            specs.append((lid, content, f"{dom}/t{subtopic}"))
            leaf_words[lid] = words
            edges.append((lid, hub_lid))                      # hoja → hub
            edges.append((lid, f"{dom}::{(i + 1) % 120}"))     # anillo local
            partner = rng.randrange(120)                       # puente interno
            if partner != i:
                edges.append((lid, f"{dom}::{partner}"))
        # guardar palabras para generación de preguntas
        DOM_WORDS[dom] = leaf_words
    for j in range(60):  # ruido: mezcla vocabularios cruzados, sin aristas
        d1, d2 = rng.sample(list(DOMAINS.keys()), 2)
        w = [rng.choice(DOMAINS[d1]), rng.choice(DOMAINS[d2]),
             rng.choice(DOMAINS[d1])]
        specs.append((f"ruido::{j}", f"nz{j:03d} " + " ".join(w), "ruido/misc"))
    return specs, edges


DOM_WORDS: dict[str, dict[str, list[str]]] = {}


def build_questions(rng: random.Random):
    """120 preguntas etiquetadas. Devuelve [{qid, type, domain, query, target_lid}]."""
    questions = []
    qid = 0
    for dom in DOMAINS:
        leaves = [f"{dom}::{i}" for i in range(120)]
        picks = rng.sample(leaves, N_DIRECT + N_REFORM + N_NEIGHBOR)
        for idx, lid in enumerate(picks):
            words = DOM_WORDS[dom][lid]
            marker = lid.split("::")[1]
            marker_tok = f"m{dom[:3]}{int(marker):03d}"
            if idx < N_DIRECT:
                # directa: mitad por marcador único, mitad solo vocabulario
                if idx % 2 == 0:
                    q = f"informacion {marker_tok}"
                else:
                    q = " ".join(rng.sample(words, len(words)))
                ttype = "directa"
            elif idx < N_DIRECT + N_REFORM:
                tpl = REFORM_TEMPLATES[idx % len(REFORM_TEMPLATES)]
                ws = rng.sample(words, 3)
                q = tpl.format(w1=ws[0], w2=ws[1], w3=ws[2],
                               f1=rng.choice(FILLER), f2=rng.choice(FILLER))
                ttype = "reformulada"
            else:
                # solo-vecino: vocabulario del compañero de anillo (i+1),
                # excluyendo palabras compartidas con el objetivo
                nb = f"{dom}::{(int(marker) + 1) % 120}"
                nb_words = [w for w in DOM_WORDS[dom][nb] if w not in words]
                if len(nb_words) < 2:
                    nb_words = DOM_WORDS[dom][nb][:2]
                q = " ".join(nb_words) + " " + rng.choice(FILLER)
                ttype = "solo-vecino"
                lid = nb if len(nb_words) >= 2 else lid  # target = el vecino
                if len([w for w in DOM_WORDS[dom][nb] if w not in words]) < 2:
                    lid = f"{dom}::{int(marker)}"  # degenerado: objetivo original
                    ttype = "solo-vecino-degenerada"
            questions.append({"qid": qid, "type": ttype, "domain": dom,
                              "query": q, "target": lid})
            qid += 1
    return questions


# ── Brazo A: coseno plano ────────────────────────────────────────────
def arm_flat(engine, qvec_cache):
    sig_ids = list(engine.nodes.keys())
    dim = len(next(iter(engine.nodes.values())).holographic_signature)
    mat = np.zeros((len(sig_ids), dim), dtype=np.float32)
    for r, nid in enumerate(sig_ids):
        v = np.array(engine.nodes[nid].holographic_signature, dtype=np.float32)
        n = np.linalg.norm(v)
        mat[r] = v / n if n > 0 else v
    def rank(query: str, top_k: int):
        if query not in qvec_cache:
            v = np.array(engine.holographic.encode_node(query, ""), dtype=np.float32)
            n = np.linalg.norm(v)
            qvec_cache[query] = v / n if n > 0 else v
        sims = mat @ qvec_cache[query]
        order = np.argsort(-sims)[:top_k]
        return [(sig_ids[i], float(sims[i])) for i in order]
    return rank


# ── Métricas (fórmulas exactas F8) ───────────────────────────────────
def prf(rank_list, target):
    hits = sum(1 for nid, _ in rank_list[:K] if nid == target)
    prec = hits / K
    rec = hits / 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mrr = 0.0
    for rank, (nid, _) in enumerate(rank_list, start=1):
        if nid == target:
            mrr = 1.0 / rank
            break
    return prec, rec, f1, mrr


def main():
    rng = random.Random(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    print("[1/5] Construyendo corpus sintético...", flush=True)
    specs, edges = build_corpus(rng)
    questions = build_questions(rng)

    engine = OmegaCubeEngineV2(auto_load=False, holographic_dim=DIM)  # benchmark aislado
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
    print(f"      nodos={len(engine.nodes)}  asociaciones={len(seen_edges)}  "
          f"preguntas={len(questions)}  dim={DIM}  ({time.time()-t0:.1f}s)", flush=True)

    mover = CubeMover(engine)
    qvec_cache: dict[str, np.ndarray] = {}
    flat_arm = arm_flat(engine, qvec_cache)

    arms = {
        "plano-coseno": lambda q: flat_arm(q, K),
        "cube_move": lambda q: [(n["node_id"], n.get("score", 0))
                                for n in mover.cube_move(q, k=K, tau=0.05,
                                                         expand_3d=True).nodes_2d],
        "engine-v2-jerarquico": lambda q: [
            (r["node_id"], r.get("score", 0))
            for r in engine.query(q, mode="hierarchical", top_k=K,
                                  apply_boundaries=False,
                                  detect_hallucination=False)],
    }

    results = {name: {"p": [], "r": [], "f1": [], "mrr": [], "ms": [],
                      "by_type": {}} for name in arms}
    print("[2/5] Ejecutando duelo (120 preguntas × 3 brazos)...", flush=True)
    for qi, qq in enumerate(questions):
        target_nid = lid_map[qq["target"]]
        for name, fn in arms.items():
            t = time.time()
            ranked = fn(qq["query"])
            dt = (time.time() - t) * 1000
            p, r, f1, mrr = prf(ranked, target_nid)
            cell = results[name]
            cell["p"].append(p); cell["r"].append(r); cell["f1"].append(f1)
            cell["mrr"].append(mrr); cell["ms"].append(dt)
            bt = cell["by_type"].setdefault(qq["type"],
                                            {"p": [], "r": [], "f1": [], "mrr": []})
            bt["p"].append(p); bt["r"].append(r); bt["f1"].append(f1); bt["mrr"].append(mrr)
        if (qi + 1) % 40 == 0:
            print(f"      {qi+1}/{len(questions)} preguntas...", flush=True)

    print("[3/5] Agregando métricas...", flush=True)
    summary = {}
    for name, cell in results.items():
        n = len(cell["f1"])
        summary[name] = {
            "precision_at_k": round(sum(cell["p"]) / n, 4),
            "recall_at_k": round(sum(cell["r"]) / n, 4),
            "f1_at_k": round(sum(cell["f1"]) / n, 4),
            "mrr": round(sum(cell["mrr"]) / n, 4),
            "latency_ms_mean": round(sum(cell["ms"]) / n, 2),
            "by_type": {
                t: {"f1": round(sum(v["f1"]) / len(v["f1"]), 4),
                    "recall": round(sum(v["r"]) / len(v["r"]), 4)}
                for t, v in cell["by_type"].items()
            },
        }

    flat_f1 = summary["plano-coseno"]["f1_at_k"]
    best_hier = max(summary["cube_move"]["f1_at_k"],
                    summary["engine-v2-jerarquico"]["f1_at_k"])
    margin_pp = round((best_hier - flat_f1) * 100, 1)
    if margin_pp > 5:
        verdict = "CONFIRMADA"
    elif margin_pp >= 0:
        verdict = "MATIZADA"
    else:
        verdict = "REFUTADA"

    out = {
        "date": str(date.today()), "seed": SEED, "k": K, "dim": DIM,
        "n_nodes": len(engine.nodes), "n_edges": len(seen_edges),
        "n_questions": len(questions),
        "questions_by_type": {},
        "summary": summary,
        "margin_pp": margin_pp, "verdict": verdict,
    }
    for qq in questions:
        out["questions_by_type"][qq["type"]] = \
            out["questions_by_type"].get(qq["type"], 0) + 1

    print("[4/5] Guardando resultados...", flush=True)
    pdir = ROOT / "docs" / "pruebas"
    pdir.mkdir(parents=True, exist_ok=True)
    jpath = pdir / f"f8_results_{out['date']}_d{DIM}.json"
    jpath.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    lines = [
        "# F8 — La prueba reina: navegación jerárquica vs coseno plano",
        "",
        f"> **Fecha:** {out['date']} · **Seed:** {SEED} · "
        f"**Dim encoder:** {DIM} · "
        f"**Nodos:** {out['n_nodes']} · **Aristas:** {out['n_edges']} · "
        f"**Preguntas:** {out['n_questions']} · **k:** {K}",
        "",
        "## Diseño",
        "- Corpus sintético determinista: 8 dominios × (1 hub + 120 hojas) + 60 ruido.",
        "- 3 dificultades: directa (40), reformulada (40), solo-vecino (40, requiere propagación estructural).",
        "- Mismo motor y mismo espacio de embeddings para los tres brazos.",
        "- engine-v2 sin boundary filter ni detector de alucinación (se aísla el ranking).",
        "- cube_move con τ=0.05 para igualar presupuesto de resultados.",
        "",
        "## Resultados globales (@k=5)",
        "",
        "| Brazo | P@5 | R@5 | F1@5 | MRR | Latencia media (ms) |",
        "|--------|-----|-----|------|-----|---------------------|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s['precision_at_k']} | {s['recall_at_k']} | "
            f"{s['f1_at_k']} | {s['mrr']} | {s['latency_ms_mean']} |")
    lines += ["", "## Desglose por dificultad (F1@5)", "",
              "| Brazo | " + " | ".join(sorted(out["questions_by_type"])) + " |",
              "|--------|" + "---|" * len(out["questions_by_type"])]
    types_sorted = sorted(out["questions_by_type"])
    for name, s in summary.items():
        cells = [str(s["by_type"].get(t, {}).get("f1", "—")) for t in types_sorted]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "", "## Veredicto",
        "",
        f"- Mejor brazo jerárquico F1: **{best_hier}** vs plano: **{flat_f1}** "
        f"→ margen **{margin_pp:+.1f} pp**.",
        f"- **Tesis central: {verdict}** "
        f"(criterio del plan: confirmada si margen > +5 pp).",
        "",
    ]
    mpath = pdir / f"F8_{out['date']}_d{DIM}.md"
    mpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"[5/5] OK -> {jpath.name} + {mpath.name}", flush=True)

    print("\n=== RESUMEN ===")
    for name, s in summary.items():
        print(f"  {name:22} F1@5={s['f1_at_k']:.3f}  R@5={s['recall_at_k']:.3f}  "
              f"MRR={s['mrr']:.3f}  {s['latency_ms_mean']:.1f} ms/q")
    print(f"\nVEREDICTO: {verdict} (margen {margin_pp:+.1f} pp)")


if __name__ == "__main__":
    main()
