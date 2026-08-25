"""
color_chain.py — Cadena de color (degradado por tono, NO escala de grises).
Fase 2 del PLAN 2026-08-09. Implementa fórmulas F3–F6.

Metáfora Matrioshka: cubos anidados, centro = verdad inmutable. Cada axioma
tiene un TONO (hue) único; la información derivada degrada la SATURACIÓN del
mismo tono hacia afuera:
    AXIOMA   depth 0 → sat 1.000
    CONCEPTO depth 1 → sat 0.700
    INSTANCIA depth 2 → sat 0.490
    EMERGENTE depth 3 → sat 0.343
El TONO = identidad de la cadena (de qué axioma viene).
La SATURACIÓN = profundidad derivacional / certeza. Los ejes son ortogonales.

F3 — hue(axioma) = int(sha256(node_id)[:8], 16) mod 360  (determinista)
F4 — sat(n) = λ^depth(n), λ = 0.7
F5 — tono como vector unitario u(h)=(cos,sin); media circular por atan2.
     PROHIBIDO promediar grados como escalares (error del círculo).
F6 — APPROVED ⟺ depth ≤ D_max ∧ axioma_origen ∈ registro ∧ sat ≥ sat_min

Extiende (no reemplaza) Gray-Scale Validation: el gris dice "qué tan
verdadero", el tono dice "verdadero respecto a QUÉ axioma".
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

LAMBDA = 0.7      # F4: decaimiento de saturación por nivel
D_MAX = 4         # F6: profundidad derivacional máxima aceptable
SAT_MIN = 0.24    # F6: ≈ λ³, saturación mínima para aprobar
MIX_CONCENTRATION = 0.9   # F5: >0.9 cadena pura; ≤0.9 posible mezcla
DETECT_CONCENTRATION = 0.5  # F5: <0.5 → MEZCLA declarada


def hue_unit(h_deg: float) -> tuple[float, float]:
    """F5: u(h) = (cos(h·π/180), sin(h·π/180))."""
    r = math.radians(h_deg % 360)
    return (math.cos(r), math.sin(r))


def vec_to_hue(vec: tuple[float, float]) -> float:
    """atan2(R_y, R_x) en grados [0, 360)."""
    return math.degrees(math.atan2(vec[1], vec[0])) % 360


def circular_mean(hues_deg: list[float]) -> tuple[float, float]:
    """F5: media circular correcta + concentración.

    Returns: (hue_medio_grados, concentración ∈ [0,1])
    concentración ≈ 1 → todos los aportes del mismo tono (cadena pura)
    concentración baja → MEZCLA
    """
    if not hues_deg:
        raise ValueError("circular_mean: lista vacía")
    rx = sum(math.cos(math.radians(h)) for h in hues_deg)
    ry = sum(math.sin(math.radians(h)) for h in hues_deg)
    n = len(hues_deg)
    conc = math.hypot(rx, ry) / n
    return vec_to_hue((rx, ry)), conc


class ColorChain:
    """Asigna, propaga, detecta mezcla y verifica linaje de color sobre un engine."""

    def __init__(self, engine, lambda_: float = LAMBDA, d_max: int = D_MAX,
                 sat_min: float = SAT_MIN):
        self.engine = engine
        self.lambda_ = lambda_
        self.d_max = d_max
        self.sat_min = sat_min

    # ── F3: asignación de tono ────────────────────────────────────

    @staticmethod
    def _sha_angle(node_id: str) -> float:
        """Ángulo determinista desde sha256 — la base 'dirección de origen' de F3."""
        h = hashlib.sha256(node_id.encode("utf-8")).hexdigest()
        return int(h[:8], 16) % 360

    def assign_axiom_hues(self) -> dict[str, float]:
        """Asigna tono a cada AXIOMA del registro. Devuelve {axiom_id: hue}.

        REFINAMIENTO DE F3 (necesario para que la prueba 2.4 del plan sea
        posible): el tono es identidad de cadena, y una identidad solo sirve si
        es DISTINGUIBLE. sha256 mod 360 con pocos axiomas puede agruparlos
        (observado: 193/220/232 en un arco de 39°, lo que hace ambigua la
        detección de mezcla). Se asignan tonos espaciados uniformemente
        (separación mínima = 360/n), con un offset base derivado de sha256
        (conserva el espíritu de 'dirección de origen' de F3) y orden estable
        por node_id (determinista entre procesos).
        """
        axs = sorted(self.engine.axioms, key=lambda a: a.node_id)
        n = len(axs)
        if n == 0:
            return {}
        base = self._sha_angle(axs[0].node_id) if n > 1 else self._sha_angle(axs[0].node_id)
        step = 360.0 / n if n > 1 else 0.0
        hues = {}
        for i, ax in enumerate(axs):
            hue = (base + i * step) % 360 if n > 1 else base
            ax.hue = round(hue, 2)
            ax.saturation = 1.0          # depth 0 → λ⁰ = 1.0
            ax.lineage = [ax.node_id]
            ax.hue_origin = ax.node_id
            ax.hue_vector = list(hue_unit(ax.hue))
            hues[ax.node_id] = ax.hue
        return hues

    def axiom_hues(self) -> dict[str, float]:
        """Mapa de tonos del registro: hue asignado, o ángulo sha256 si no."""
        return {ax.node_id: (ax.hue if ax.hue is not None else self._sha_angle(ax.node_id))
                for ax in self.engine.axioms}

    # ── F4: propagación del degradado ─────────────────────────────

    def _axiom_for_domain(self, domain: str):
        for ax in self.engine.axioms:
            if ax.primary_hierarchy.split(".")[0] == domain:
                return ax
        return None

    def propagate(self) -> dict[str, list]:
        """Propaga tono+saturación desde cada axioma hacia abajo.

        Regla de asignación (grafo sin aristas de derivación): cada nodo hereda
        el tono del axioma de su DOMINIO (prefijo de jerarquía). La profundidad
        derivacional se estima por distancia jerárquica al axioma:
            depth = max(0, len(nivel_nodo) − len(nivel_axioma))
        sat = λ^depth, truncada en sat_min de F6.

        Returns: {"colored": [nid...], "orphans": [nid...]}
        """
        result = {"colored": [], "orphans": []}
        for nid, node in self.engine.nodes.items():
            if node.node_type == "AXIOM":
                continue  # los axiomas ya se colorearon en assign_axiom_hues
            dom = node.primary_hierarchy.split(".")[0] if node.primary_hierarchy else None
            ax = self._axiom_for_domain(dom) if dom else None
            if ax is None or ax.hue is None:
                node.hue = None
                node.saturation = None
                node.lineage = []
                node.hue_origin = None
                node.hue_vector = None
                result["orphans"].append(nid)
                continue
            depth = max(0, len(node.primary_hierarchy.split("."))
                        - len(ax.primary_hierarchy.split(".")))
            node.hue = ax.hue
            node.saturation = self.lambda_ ** depth
            node.lineage = [ax.node_id, nid]
            node.hue_origin = ax.node_id
            node.hue_vector = list(hue_unit(ax.hue))
            result["colored"].append(nid)
        return result

    # ── Derivación explícita (construye cadenas profundas) ────────

    def attach_derived(self, parent_id: str, child_id: str) -> bool:
        """Declara child como derivado de parent (1 nivel más profundo).

        child hereda el tono exacto del origen de parent; su saturación decae
        un factor λ respecto a la del padre. Crea arista de linaje real,
        independiente de la jerarquía (para cadenas multi-nivel y pruebas).
        """
        parent = self.engine.nodes.get(parent_id)
        child = self.engine.nodes.get(child_id)
        if parent is None or child is None:
            return False
        if parent.hue is None:
            return False
        child.hue = parent.hue
        child.saturation = (parent.saturation or 1.0) * self.lambda_
        child.lineage = list(parent.lineage) + [child_id]
        child.hue_origin = parent.hue_origin
        child.hue_vector = list(hue_unit(parent.hue))
        return True

    def create_hybrid(self, parent_axiom_ids: list[str], hybrid_node_id: str) -> dict | None:
        """F5: crea un nodo híbrido desde varios axiomas (mezcla de cadenas).

        El híbrido guarda el vector resultante de sumar los vectores unitarios
        de los tonos padre. Su hue = ángulo del resultante; su saturación =
        concentración (un híbrido de dos cadenas distintas NUNCA es 100% puro).
        """
        hybrid = self.engine.nodes.get(hybrid_node_id)
        if hybrid is None:
            return None
        axes = []
        hues = []
        for aid in parent_axiom_ids:
            ax = self.engine.nodes.get(aid)
            if ax is None or ax.node_type != "AXIOM":
                return None
            hue = ax.hue if ax.hue is not None else self._sha_angle(ax.node_id)
            hues.append(hue)
            axes.append(hue_unit(hue))

        rx = sum(v[0] for v in axes)
        ry = sum(v[1] for v in axes)
        conc = math.hypot(rx, ry) / len(axes)
        hybrid.hue_vector = [rx, ry]
        hybrid.hue = vec_to_hue((rx, ry))
        hybrid.saturation = conc
        hybrid.lineage = sorted(parent_axiom_ids) + [hybrid_node_id]
        hybrid.hue_origin = None  # híbrido: sin origen único
        return {"hue": hybrid.hue, "concentration": conc, "parents": list(parent_axiom_ids)}

    # ── F5: detección de mezcla ───────────────────────────────────

    def detect_mixture(self, node_id: str) -> dict:
        """Clasifica el color de un nodo y, si es mezcla, identifica los padres.

        F5 — proyección_a = R · u(h_a); los axiomas con mayor proyección
        positiva = candidatos a padres. Geometría primero; el linaje almacenado
        desempata cuando el tono de la mezcla colisiona con el tono de otro
        axioma (ambigüedad geométrica real: con axiomas uniformemente espaciados,
        la mezcla de un par a distancia 2 cae EXACTAMENTE sobre un tercer axioma).

        Caso disperso: dos tonos muy separados (≥120°) dan concentración baja
        (resultante corta); la geometría sola no atribuye padres → linaje.
        """
        node = self.engine.nodes.get(node_id)
        if node is None or node.hue_vector is None:
            return {"status": "UNCOLORED", "node_id": node_id}

        rx, ry = node.hue_vector
        norm = math.hypot(rx, ry)
        # concentración normalizada (F5: ||R||/n). create_hybrid la guarda en
        # saturation; fallback a la norma bruta acotada si no existe.
        conc = node.saturation if node.saturation is not None else min(1.0, norm)

        # nodo AXIOM puro
        if node.node_type == "AXIOM":
            return {"status": "PURE", "node_id": node_id, "hue": node.hue,
                    "concentration": 1.0, "parents": [node_id]}

        registry = self.axiom_hues()
        stored = [x for x in (node.lineage or []) if x in registry]

        # MEZCLA DISPERSA: concentración baja (tonos separados ≥120°).
        # La resultante es corta y la geometría no atribuye padres con confianza.
        if conc <= DETECT_CONCENTRATION:
            if len(stored) >= 2:
                return {"status": "MIXTURE", "node_id": node_id, "hue": node.hue,
                        "concentration": conc, "parents": stored[:2],
                        "lineage_assisted": True,
                        "note": "mezcla dispersa: padres por linaje almacenado"}
            return {"status": "MIXTURE", "node_id": node_id, "hue": node.hue,
                    "concentration": conc, "parents": [],
                    "note": "mezcla dispersa sin linaje: padres indeterminables"}

        # proyección sobre el tono de cada axioma del registro
        projections = []
        for aid, hue in registry.items():
            ux, uy = hue_unit(hue)
            proj = rx * ux + ry * uy
            projections.append((proj, aid, hue))
        projections.sort(key=lambda x: -x[0])

        # cadena pura: resultante alineada con UN solo axioma
        if projections and projections[0][0] > 0.9 * norm:
            # ¿el segundo también proyecta fuerte y está separado? → mezcla
            if len(projections) > 1 and projections[1][0] > 0.25 * norm:
                _, aid1, h1 = projections[0]
                _, aid2, h2 = projections[1]
                sep = abs((h1 - h2 + 180) % 360 - 180)
                if sep > 15:
                    geo = [aid1, aid2]
                    if len(stored) >= 2 and set(stored[:2]) != set(geo):
                        return self._lineage_override(node, stored, conc, projections)
                    return {"status": "MIXTURE", "node_id": node_id, "hue": node.hue,
                            "concentration": conc, "parents": geo,
                            "projections": [(round(p, 4), aid) for p, aid, _ in projections]}
            return {"status": "PURE", "node_id": node_id, "hue": node.hue,
                    "concentration": conc, "parents": [projections[0][1]],
                    "origin": node.hue_origin}

        # resultante entre dos tonos sin alinearse con ninguno → mezcla (F5)
        positives = [p for p in projections if p[0] > 0]
        if len(positives) >= 2:
            geo = [positives[0][1], positives[1][1]]
            # colisión de tono: la mezcla cae sobre un axioma que NO es padre.
            # El linaje almacenado (si existe y proyecta positivo) desempata.
            if len(stored) >= 2 and set(stored[:2]) != set(geo):
                proj_map = {aid: p for p, aid, _ in projections}
                if all(proj_map.get(s, -1.0) > 0 for s in stored[:2]):
                    return self._lineage_override(node, stored, conc, projections)
            return {"status": "MIXTURE", "node_id": node_id, "hue": node.hue,
                    "concentration": conc, "parents": geo,
                    "projections": [(round(p, 4), aid) for p, aid, _ in projections]}

        return {"status": "PURE", "node_id": node_id, "hue": node.hue,
                "concentration": conc,
                "parents": [positives[0][1]] if positives else [],
                "origin": node.hue_origin}

    def _lineage_override(self, node, stored: list, conc: float, projections) -> dict:
        """Resolución por linaje cuando la geometría es ambigua."""
        return {"status": "MIXTURE", "node_id": node.node_id, "hue": node.hue,
                "concentration": conc, "parents": stored[:2],
                "lineage_assisted": True,
                "note": "tono de mezcla colisiona con otro axioma; "
                        "padres por linaje almacenado (proyecciones positivas)",
                "projections": [(round(p, 4), aid) for p, aid, _ in projections]}

    # ── F6: verificación de linaje ────────────────────────────────

    def verify_lineage(self, node_id: str) -> tuple[str, str]:
        """F6: APPROVED ⟺ depth ≤ D_max ∧ axioma_origen ∈ registro ∧ sat ≥ sat_min.

        Returns: (veredicto, explicación) con la ruta de linaje real.
        """
        node = self.engine.nodes.get(node_id)
        if node is None:
            return "FLAGGED", f"nodo {node_id} no existe"

        if not node.lineage:
            return "FLAGGED", "sin linaje (nodo huérfano, nunca coloreado)"

        origin = node.lineage[0]
        registry = {ax.node_id for ax in self.engine.axioms}
        if origin not in registry:
            return "FLAGGED", f"linaje roto: origen {origin[:16]} NO está en el registro de axiomas"

        # todos los ids del linaje deben existir en el grafo
        missing = [x for x in node.lineage if x not in self.engine.nodes]
        if missing:
            return "FLAGGED", f"linaje roto: {len(missing)} ids del camino no existen"

        depth = len(node.lineage) - 1  # camino completo al centro
        if depth > self.d_max:
            return "FLAGGED", f"depth {depth} > D_max={self.d_max}"

        sat = node.saturation if node.saturation is not None else self.lambda_ ** depth
        if sat < self.sat_min:
            return "FLAGGED", f"sat {sat:.3f} < sat_min={self.sat_min} (depth {depth})"

        chain = " → ".join(x[:12] for x in node.lineage)
        return "APPROVED", f"linaje verificable depth={depth} sat={sat:.3f}: {chain}"
