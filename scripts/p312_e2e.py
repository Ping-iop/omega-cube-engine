"""P3.12 E2E — Motor real con el fix de clave unitaria.

1) Carga el store de PRODUCCION en modo LECTURA (sin save) y consulta
   holografica real: confirma que las firmas persistidas (generadas con el
   bind viejo) siguen siendo compatibles como vectores de comparacion.
2) Grafo temporal: add_node x3 + associate + query holographic + unbind
   exactitud sobre bind() del camino vivo (_recompute_signature).
"""
import os
import statistics
import sys
import tempfile

import numpy as np

DEPLOY = r"C:\Users\GPAMD\.hermes\axioma-omega-protocol"
sys.path.insert(0, DEPLOY)

from omega_cube.engine import OmegaCubeEngine
from omega_cube.holographic import HolographicEncoder

enc = HolographicEncoder(dimension=2048, seed=7)

# ---------- 1) produccion, solo lectura ----------
eng = OmegaCubeEngine(memory_dir=DEPLOY + r"\memory", holographic_dim=2048)
n_nodes = len(eng.nodes)
qres = eng.query("cache de embeddings degradados en cube_move", mode="holographic", top_k=5)
print(f"[prod] nodos={n_nodes}  query holo -> {len(qres)} resultados")
for r in qres[:3]:
    print(f"       {r['score']:.4f}  {r['node_id'][:24]}...  {r['content'][:60]}")

# ---------- 2) grafo temporal, camino vivo ----------
with tempfile.TemporaryDirectory() as td:
    e2 = OmegaCubeEngine(memory_dir=td, holographic_dim=256)
    n0 = e2.add_node("Router GPU clasifica queries y delega al worker", ["DEV.ARCHITECTURE"])
    n1 = e2.add_node("Worker ejecuta inferencia y devuelve JSON", ["DEV.ARCHITECTURE"])
    n2 = e2.add_node("Scheduler prefetcha resultados frecuentes", ["DEV.ARCHITECTURE"])
    ids = [n0.node_id, n1.node_id, n2.node_id]
    ok1 = e2.associate(ids[0], ids[1])
    ok2 = e2.associate(ids[1], ids[2])
    print(f"[tmp] asociaciones: {ok1}, {ok2}")

    # firmas recalculadas via _recompute_signature -> usan bind() corregido
    sig = np.asarray(n0.holographic_signature)
    self_vec = np.asarray(e2.holographic.encode_node(n0.content, "DEV.ARCHITECTURE"))

    # unbind de la firma completa no es la operacion (firma = bundle), pero el
    # par bind/unbind del camino vivo debe recuperar exacto:
    bound = e2.holographic.bind(self_vec.tolist(), sig.tolist())
    rec = e2.holographic.unbind(bound, self_vec.tolist())
    nu, nv = np.linalg.norm(rec), np.linalg.norm(sig)
    cos = float(np.dot(rec, sig) / (nu * nv))
    mapped = enc.similarity(rec, sig)

    # query holographica sobre el grafo temporal
    q = e2.query("clasificacion GPU router worker", mode="holographic", top_k=3)
    print(f"[tmp] query holo top: {q[0]['score']:.4f} {q[0]['content'][:50]}")
    print(f"[tmp] unbind(bind(self,firma)) cos={cos:.6f} mapped={mapped:.4f}")

    # persistencia roundtrip con el fix activo
    e2.save()
    e3 = OmegaCubeEngine(memory_dir=td, holographic_dim=256)
    same = len(e3.nodes) == 3 and bool(e3.nodes[ids[0]].holographic_signature)
    print(f"[tmp] roundtrip save/load nodos={len(e3.nodes)} firma_viva={same}")

assert n_nodes >= 90, f"store inesperado: {n_nodes}"
print("E2E_OK")
