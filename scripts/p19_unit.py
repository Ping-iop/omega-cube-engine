"""Smoke P1.9 camino sano: embedder con Ollama stubbeado (sin tocar prod cache)."""
import sys, tempfile, os
sys.path.insert(0, '.')

import numpy as np
import omega_cube.embeddings as emb_mod

# Stub de _embed_api: devuelve vectores 8-dim deterministas por texto
calls = {"n": 0}
def fake_embed_api(text: str):
    calls["n"] += 1
    rng = np.random.RandomState(abs(hash(text)) % (2**32))
    v = rng.rand(8).astype(np.float32)
    return v

emb_mod._embed_api = fake_embed_api

from omega_cube.embeddings import SemanticEmbedder

tmpdir = tempfile.mkdtemp()
e = SemanticEmbedder(tmpdir)

# Caso 1: embed_nodes marca generador y fuente ollama
from types import SimpleNamespace
nodes = {"n1": SimpleNamespace(node_id="n1", content="derivadas en calculus", primary_hierarchy=None),
         "n2": SimpleNamespace(node_id="n2", content="torneos de evony", primary_hierarchy=None)}
vecs = e.embed_nodes(nodes)
print("caso1 vecs:", sorted(vecs.keys()), "| last_source:", e.last_source)
gen_n1 = e.cache["n1"]["gen"]
print("cache gen:", gen_n1)

# Caso 2: mezcla de dims detectada por query_compatible_with_cache
e.cache["n1"]["gen"] = "holographic_dim999"   # simular vector viejo otra dim
q = e.embed_query("calcular derivada")
print("caso2 dim-mismatch bloqueado ->", e.query_compatible_with_cache(q) == False)

# Caso 3: mismo generador -> compatible
e.cache["n1"]["gen"] = gen_n1
q2 = e.embed_query("calcular derivada")
print("caso3 generador compatible ->", e.query_compatible_with_cache(q2) == True)

print("llamadas a api:", calls["n"])
print("P1.9 SANO OK")
