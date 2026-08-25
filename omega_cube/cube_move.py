"""
cube_move.py — Fase 2D + Fase 3D del puzzle de cubos (PLAN 2026-08-09).

Metáfora: dado un query, TODOS los cubos del tablero se mueven a la vez hacia
él (Fase 2D, vectorial); las K fichas más relevantes caen en el hueco
(presupuesto de contexto). Luego, giros de cara del Rubik (Fase 3D) siguen
asociaciones cruzadas entre dominios para traer conceptos de temas vecinos.

Implementación exacta de las fórmulas del plan (sección 4c):

F1 — Fase 2D:
    v̂ = v / ||v||
    s_i = v̂_i · q̂            (UNA operación matricial: scores = V_norm @ q_norm)
    S_2D = { i : s_i ≥ τ } ordenado descendente, máx k nodos
    τ inicial = 0.3 (calibrar con batería 1.1/1.2)

F2 — Fase 3D:
    S_3D = S_2D ∪ { j : ∃ i ∈ S_2D, (i,j) ∈ A ∧ dominio(j) ≠ dominio(i) }
    |S_3D| ≤ k; si desborda, ordenar por peso(i,j)·s_i y cortar en k
    (peso de asociación: 1/posición en la lista de asociaciones = fuerza)

Validación del ensamblado: cada nodo seleccionado debe tener linaje verificable
hasta un axioma del registro (se integra con color_chain en Fase 2/3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CubeMoveResult:
    """Resultado completo de una jugada de cube_move."""
    query: str
    nodes_2d: list[dict] = field(default_factory=list)   # fichas que cayeron al hueco
    nodes_3d: list[dict] = field(default_factory=list)   # traídas por giros de cara
    scores: dict[str, float] = field(default_factory=dict)
    verdict: str = "FLAGGED"                              # APPROVED | VETOED | FLAGGED
    verdict_reason: str = ""
    tau: float = 0.3
    k: int = 20
    elapsed_ms: float = 0.0
    embed_source: str = ""  # diagnóstico: ollama | cache | offline
    fallback_used: bool = False
    hue_filtered_out: int = 0  # nodos descartados por filtro de cadena de tono
    gate: object | None = None  # GateVerdict (Fase 3) cuando hay color_chain

    def all_node_ids(self) -> list[str]:
        return [n["node_id"] for n in self.nodes_2d] + [n["node_id"] for n in self.nodes_3d]

    def summary(self) -> str:
        lines = [
            f"cube_move('{self.query}') k={self.k} τ={self.tau} [{self.elapsed_ms:.1f}ms] "
            f"embed={self.embed_source}",
            f"Fase 2D: {len(self.nodes_2d)} nodos | Fase 3D: +{len(self.nodes_3d)} nodos",
            f"Veredicto: {self.verdict} ({self.verdict_reason})",
        ]
        for n in self.nodes_2d:
            lines.append(f"  [2D] {self.scores[n['node_id']]:+.3f}  {n['node_id'][:40]:40} "
                         f"{n['primary_hierarchy']}  \"{n['content'][:60]}\"")
        for n in self.nodes_3d:
            lines.append(f"  [3D] {self.scores.get(n['node_id'], 0):+.3f}  {n['node_id'][:40]:40} "
                         f"{n['primary_hierarchy']}  via {n.get('via', '?')}")
        return "\n".join(lines)


class CubeMover:
    """Mecánica del cubo sobre un OmegaCubeEngine cargado."""

    def __init__(self, engine, embedder=None, color_chain=None):
        """
        engine: OmegaCubeEngine con nodos cargados
        embedder: SemanticEmbedder (opcional). Sin él, fallback a firmas
                  holográficas (256 dims) — mecánica intacta, semántica reducida.
        color_chain: ColorChain (opcional, Fase 3). Si se pasa, el veredicto del
                  ensamblado usa linaje de color (ValidityGate); si no, fallback
                  jerárquico por dominio (Fase 1).
        """
        self.engine = engine
        self.embedder = embedder
        self.color_chain = color_chain

    # ── Fase 2D ──────────────────────────────────────────────────

    def _embed_all_nodes(self, force: bool = False) -> dict[str, np.ndarray]:
        """Vectors de todos los nodos (cache) o fallback holográfico."""
        if self.embedder is not None:
            vecs = self.embedder.embed_nodes(self.engine.nodes, force=force)
            if vecs and len(vecs) == len(self.engine.nodes):
                # P1.9: embeddings semánticos frescos → limpiar flags viejos
                # (un nodo degradado en una jugada anterior no debe manchar
                # el veredicto de una jugada con embeddings reales)
                for node in self.engine.nodes.values():
                    if getattr(node, "_degraded_embedding", False):
                        del node._degraded_embedding
                return vecs
        # Fallback: firmas holográficas normalizadas (256 dims, deterministas)
        vecs = {}
        for nid, node in self.engine.nodes.items():
            sig = node.holographic_signature or []
            if sig:
                v = np.array(sig, dtype=np.float32)
                norm = np.linalg.norm(v)
                vecs[nid] = v / norm if norm > 0 else v
                # P1.9: etiqueta RUNTIME del nodo con embedding degradado.
                # No persiste (to_dict lista campos explícitos); el gate la
                # reporta para que la degradación nunca sea silenciosa.
                setattr(node, "_degraded_embedding", True)
        return vecs

    def _fallback_query_vec(self, query: str) -> np.ndarray:
        """Vector de query en espacio holográfico (fallback offline).

        NOTA F8 (2026-08-25): se usa hier="" (mismo tratamiento que los nodos).
        El marcador "QUERY" entraba al hash como un token más y degradaba el
        recall del fallback medido en docs/pruebas/f8_ablation_2026-08-25_d2048.json.
        """
        v = np.array(
            self.engine.holographic.encode_node(query, ""), dtype=np.float32
        )
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def cube_move(
        self,
        query: str,
        k: int = 20,
        tau: float = 0.3,
        # F8 (2026-08-25): fase 3D DESACTIVADA por defecto. La ablación midió
        # que los giros 3D restan rendimiento (F1@5: 0.231 plano+2D vs 0.158
        # con 3D activo) y son ciegos a aristas intra-dominio. Disponible para
        # experimentos vía expand_3d=True explícito.
        expand_3d: bool = False,
        hue_filter: float | None = None,
        hue_toleranceance: float = 10.0,
    ) -> CubeMoveResult:
        """
        Jugada completa: Fase 2D (tablero se ordena) + Fase 3D (giros de cara).

        Args:
            query: texto de búsqueda
            k: presupuesto de contexto ("el hueco")
            tau: umbral mínimo de similitud coseno (F1)
            expand_3d: activar Fase 3D (asociaciones cruzadas)
            hue_filter: si se pasa, la respuesta sigue SOLO esa cadena de tono
                (Fase 2): los nodos con tono a más de hue_toleranceance grados
                (distancia circular) quedan fuera del ensamblado.
            hue_toleranceance: tolerancia angular del filtro (default 10°)
        """
        t0 = time.perf_counter()
        res = CubeMoveResult(query=query, k=k, tau=tau)

        # --- Fase 2D: todos los cubos se mueven a la vez (operación matricial) ---
        node_vecs = self._embed_all_nodes()
        if not node_vecs:
            res.verdict = "FLAGGED"
            res.verdict_reason = "sin vectores de nodo disponibles"
            res.elapsed_ms = (time.perf_counter() - t0) * 1000
            return res

        if self.embedder is not None:
            q_vec = self.embedder.embed_query(query)
            res.embed_source = self.embedder.last_source
            # P1.9: bloquear mezcla de espacios vectoriales — si el cache
            # fue generado por otro modelo/dim, la query nomic no es
            # comparable y corrompería los scores coseno.
            if q_vec is not None and not self.embedder.query_compatible_with_cache(q_vec):
                q_vec = None
        else:
            q_vec = None
        if q_vec is None:
            q_vec = self._fallback_query_vec(query)
            res.fallback_used = True
            res.embed_source = "holographic_fallback"

        ids = list(node_vecs.keys())
        dim = len(node_vecs[ids[0]])
        V = np.empty((len(ids), dim), dtype=np.float32)
        for i, nid in enumerate(ids):
            v = node_vecs[nid]
            # tolerancia de dimensión (mix cache 768 / fallback 256)
            if len(v) != dim:
                v = np.resize(v, dim)
                n = np.linalg.norm(v)
                v = v / n if n > 0 else v
            V[i] = v
        # F1: UNA operación matricial (ya normalizados → producto punto = coseno)
        scores = V @ q_vec

        ranked = sorted(zip(ids, scores), key=lambda x: -float(x[1]))
        selected = [(nid, float(s)) for nid, s in ranked if s >= tau]
        res.scores = {nid: s for nid, s in selected}

        # --- Fase 3D: giros de cara — asociaciones cruzadas, 1 nivel ---
        # F2: los giros compiten con las fichas 2D por el hueco (|S_3D| ≤ k total).
        # Cada giro se pondera peso(i,j)·s_i; el ensamblado final es el top-k
        # de la unión. Si un giro supera a una ficha 2D débil, la desplaza.
        selected_ids = {nid for nid, _ in selected}
        turn_candidates = []  # (peso, assoc_id, via_nid)
        if expand_3d and selected_ids:
            for nid, s in selected:
                node = self.engine.nodes[nid]
                dom_i = node.primary_hierarchy.split(".")[0] if node.primary_hierarchy else "?"
                for pos, assoc_id in enumerate(node.associations):
                    target = self.engine.nodes.get(assoc_id)
                    if target is None or assoc_id in selected_ids:
                        continue
                    dom_j = target.primary_hierarchy.split(".")[0] if target.primary_hierarchy else "?"
                    # F8 (2026-08-25): los giros ya no ignoran aristas intra-dominio.
                    # Antes: `if dom_j == dom_i: continue` dejaba ciego al difusor
                    # justo en el caso vecino->objetivo del mismo dominio.
                    # Ahora todo participa; el cruce de dominios lleva bono x2.
                    bono_cruce = 2.0 if dom_j != dom_i else 1.0
                    peso = (1.0 / (pos + 1)) * s * bono_cruce
                    turn_candidates.append((peso, assoc_id, nid))

        # Unión: fichas 2D + giros 3D, top-k global (los giros pueden desplazar)
        assembly = [(s, nid, None) for nid, s in selected]
        assembly += [(w, aid, via) for w, aid, via in turn_candidates]
        assembly.sort(key=lambda x: -x[0])

        # Filtro por cadena de tono (Fase 2): la query sigue solo su color
        if hue_filter is not None:
            def _in_chain(nid):
                node = self.engine.nodes.get(nid)
                if node is None or node.hue is None:
                    return False
                d = abs((node.hue - hue_filter + 180) % 360 - 180)
                return d <= hue_toleranceance
            kept = [item for item in assembly if _in_chain(item[1])]
            res.hue_filtered_out = len(assembly) - len(kept)
            assembly = kept

        for s, nid, via in assembly[:k]:
            node = self.engine.nodes[nid]
            entry = {
                "node_id": nid,
                "primary_hierarchy": node.primary_hierarchy,
                "node_type": node.node_type,
                "content": node.content,
                "score": s,
            }
            if via is None:
                res.nodes_2d.append(entry)
            else:
                entry["via"] = via
                entry["assoc_weight"] = s
                res.nodes_3d.append(entry)
            res.scores[nid] = s

        # --- Validación del ensamblado contra el registro de axiomas ---
        res.verdict, res.verdict_reason = self._validate_assembly(res)

        # P1.9 (2026-08-25): la degradación de embeddings NUNCA es silenciosa.
        # Cubre ambas ramas del validador (gate y fallback jerárquico):
        # cuenta nodos con embedding degradado, lo reporta y baja APPROVED a
        # FLAGGED para que nadie confíe en calidad semántica que no existe.
        all_nodes = res.nodes_2d + res.nodes_3d
        n_degraded = sum(
            1 for n in all_nodes
            if getattr(self.engine.nodes.get(n["node_id"]),
                       "_degraded_embedding", False)
        )
        if n_degraded:
            nota = (f" | ⚠ {n_degraded}/{len(all_nodes)} embeddings degradados "
                    f"(ollama caído → firmas holográficas)")
            g = getattr(res, "gate", None)
            if g is not None:
                g.degraded_embeddings = n_degraded
                g.reason += nota
                if g.verdict == "APPROVED":
                    g.verdict = "FLAGGED"
                res.verdict, res.verdict_reason = g.verdict, g.reason
            else:
                res.verdict_reason += nota
                if res.verdict == "APPROVED":
                    res.verdict = "FLAGGED"

        res.elapsed_ms = (time.perf_counter() - t0) * 1000
        return res

    def _validate_assembly(self, res: CubeMoveResult) -> tuple[str, str]:
        """
        Ensamblado válido ⟺ todos los nodos seleccionados tienen linaje
        verificable hasta un axioma del registro.

        Fase 3: si hay color_chain, usa ValidityGate (linaje de color real).
        Fase 1 (fallback): linaje jerárquico por dominio.
        """
        if not res.nodes_2d and not res.nodes_3d:
            return "VETOED", "ensamblado vacío: ningún nodo superó τ"

        if self.color_chain is not None:
            from .validity_gate import ValidityGate
            gate = ValidityGate(self.engine, self.color_chain)
            v = gate.validate_response(res.all_node_ids())
            res.gate = v
            return v.verdict, v.reason

        axiom_ids = {a.node_id for a in self.engine.axioms}
        axiom_domains = {
            a.primary_hierarchy.split(".")[0] for a in self.engine.axioms
            if a.primary_hierarchy
        }

        verified, unverified = 0, []
        for n in res.nodes_2d + res.nodes_3d:
            nid = n["node_id"]
            dom = n["primary_hierarchy"].split(".")[0] if n["primary_hierarchy"] else "?"
            if nid in axiom_ids or dom in axiom_domains:
                verified += 1
            else:
                unverified.append(nid)

        total = verified + len(unverified)
        if verified == total:
            return "APPROVED", f"linaje verificable en {verified}/{total} nodos"
        if verified == 0:
            return "VETOED", f"0/{total} nodos con linaje a axiomas"
        return "FLAGGED", f"linaje parcial: {verified}/{total} nodos verificados"
