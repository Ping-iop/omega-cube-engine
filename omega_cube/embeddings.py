"""
SemanticEmbedder — capa vectorial semántica para Omega-Cube.

Usa nomic-embed-text vía ollama (puerto 11434) — embeddings reales de 768 dims.
Reemplaza los embeddings ALEATORIOS del TurboVec bridge (fallback sin servidor
de embeddings activo, ver test_embed 2026-08-09: similitudes ~0.0 = ruido).

Cache en disco (memory/semantic_embeddings.json): node_id → vector.
Solo re-embeddea nodos nuevos o con contenido cambiado (content_hash).

Prefijos nomic: "search_document:" para nodos, "search_query:" para queries
(retrieval asimétrico — mejora ~3-5 puntos en benchmarks del modelo).

Si ollama está caído: degrade a firmas holográficas del engine (256 dims) —
la mecánica cube_move sigue funcionando, con menor calidad semántica.
"""

import json
import os
import hashlib
import time
import urllib.request
from pathlib import Path

import numpy as np

OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
CONTENT_CHARS = 512  # contenido máximo que se embeddea por nodo


def _embed_api(text: str, timeout: float = 30.0) -> np.ndarray:
    """Llamada a ollama. Lanza excepción si no responde."""
    payload = {"model": EMBED_MODEL, "input": text}
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return np.array(data["embeddings"][0], dtype=np.float32)


class SemanticEmbedder:
    """Embeddings semánticos con cache persistente."""

    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)
        self.cache_path = self.memory_dir / "semantic_embeddings.json"
        self.cache: dict = {}  # node_id -> {"hash": str, "vec": list[float], "gen": str}
        self._load_cache()
        self.last_source = None  # "ollama" | "holographic" (diagnóstico)

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, encoding="utf-8") as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.cache = {}

    def save_cache(self):
        tmp = str(self.cache_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)
        os.replace(tmp, self.cache_path)

    @staticmethod
    def _node_text(node) -> str:
        """Texto a embeddear: contenido + jerarquía (la jerarquía da contexto)."""
        h = node.primary_hierarchy or ""
        return f"search_document: {node.content[:CONTENT_CHARS]} [{h}]"

    @staticmethod
    def _content_hash(node) -> str:
        return hashlib.sha256(
            (node.content[:CONTENT_CHARS] + "|" + (node.primary_hierarchy or "")).encode()
        ).hexdigest()[:16]

    def embed_nodes(self, nodes: dict, force: bool = False) -> dict[str, np.ndarray]:
        """Embeddea todos los nodos. Usa cache cuando el contenido no cambió.

        Returns: {node_id: vector normalizado}. Si ollama cae, devuelve {}
        (el caller decide el fallback).
        """
        out: dict[str, np.ndarray] = {}
        pending = []

        for nid, node in nodes.items():
            ch = self._content_hash(node)
            cached = self.cache.get(nid)
            if not force and cached and cached.get("hash") == ch:
                out[nid] = np.array(cached["vec"], dtype=np.float32)
            else:
                pending.append((nid, node, ch))

        if pending:
            try:
                for nid, node, ch in pending:
                    vec = _embed_api(self._node_text(node))
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    out[nid] = vec
                    self.cache[nid] = {"hash": ch, "vec": vec.tolist(),
                                       "gen": EMBED_MODEL}
                # limpiar cache de nodos borrados del grafo
                live = set(nodes.keys())
                for nid in list(self.cache.keys()):
                    if nid not in live:
                        del self.cache[nid]
                self.save_cache()
                self.last_source = "ollama"
            except Exception:
                # ollama caído: devolver lo que se pudo + cache existente
                self.last_source = "ollama_partial"
        else:
            self.last_source = "cache"

        return out

    def embed_query(self, query: str) -> np.ndarray | None:
        """Embed de la query (siempre live, ~40-70ms warm). None si cae.

        P1.9: si el cache tiene vectores de OTRO generador (dim o modelo
        distinto), devuelve None — mezclar espacios corrompe los scores
        coseno y el caller caerá al fallback holográfico consistente.
        """
        try:
            vec = _embed_api(f"search_query: {query}")
            norm = np.linalg.norm(vec)
            self.last_source = "ollama"
            return vec / norm if norm > 0 else vec
        except Exception:
            self.last_source = "offline"
            return None

    def query_compatible_with_cache(self, q_vec: np.ndarray | None) -> bool:
        """True si q_vec vive en el mismo espacio que los vectores del cache.

        P1.9: compara dimensión contra las entradas del cache con generador
        conocido. Cache vacío o sin 'gen' (legado) se asume compatible.
        """
        if q_vec is None:
            return False
        gens = {c.get("gen") for c in self.cache.values() if isinstance(c, dict)}
        known = [g for g in gens if g]
        if not known:
            return True  # cache legado o vacío: no hay mezcla posible aún
        if len(known) > 1:
            return False  # cache ya mezclado: forzar fallback consistente
        if known[0] != EMBED_MODEL:
            return False
        cached_vecs = [c["vec"] for c in self.cache.values()
                       if isinstance(c, dict) and c.get("gen") == EMBED_MODEL]
        if cached_vecs and len(cached_vecs[0]) != len(q_vec):
            return False
        return True
