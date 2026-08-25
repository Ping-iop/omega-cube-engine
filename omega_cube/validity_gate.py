"""
validity_gate.py — Gate de validez del ensamblado (Fase 3, PLAN 2026-08-09).

Una respuesta es APPROVED si TODOS sus nodos tienen linaje de color verificable
hasta un axioma del registro. FLAGGED si el linaje es parcial o se rompe.
VETOED si ningún nodo tiene linaje verificable (o el ensamblado está vacío).

EXTIENDE (no reemplaza) Gray-Scale Validation: cada nodo conserva su perfil
gris (qué tan verdadero); el gate añade la dimensión de PROCEDENCIA (verdadero
respecto a QUÉ axioma). El veredicto final reporta ambos ejes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateVerdict:
    verdict: str                 # APPROVED | VETOED | FLAGGED
    reason: str
    node_verdicts: dict = field(default_factory=dict)  # nid -> (verdict, detail)
    axiom_chains: list = field(default_factory=list)   # cadenas verificadas
    gray_scale_min: float = 0.0  # peor gray-scale composite del ensamblado
    gray_scale_avg: float = 0.0
    degraded_embeddings: int = 0  # P1.9: nodos con embedding fallback (no semántico)

    def summary(self) -> str:
        lines = [f"GATE: {self.verdict} — {self.reason}",
                 f"  gray-scale: min={self.gray_scale_min:.0f}% avg={self.gray_scale_avg:.0f}%"]
        if self.degraded_embeddings:
            lines.append(f"  ⚠ embeddings degradados (fallback holográfico): {self.degraded_embeddings}")
        for nid, (v, detail) in self.node_verdicts.items():
            lines.append(f"  [{v:8}] {nid[:36]:36} {detail[:90]}")
        return "\n".join(lines)


class ValidityGate:
    """Verifica el ensamblado de una respuesta contra el registro de axiomas."""

    def __init__(self, engine, color_chain):
        self.engine = engine
        self.chain = color_chain

    def validate_response(self, node_ids: list[str]) -> GateVerdict:
        """
        Regla (F6 extendida):
            APPROVED ⟺ todos los nodos: depth ≤ D_max ∧ origen ∈ registro
                       ∧ sat ≥ sat_min (verificación por linaje de color real)
            VETOED   ⟺ 0 nodos verificables
            FLAGGED  ⟺ linaje parcial
        """
        if not node_ids:
            return GateVerdict("VETOED", "ensamblado vacío")

        approved, flagged, broken = [], [], []
        for nid in node_ids:
            verdict, detail = self.chain.verify_lineage(nid)
            (approved if verdict == "APPROVED" else flagged).append((nid, verdict, detail))

        # Gray-Scale del ensamblado (eje ortogonal, existente)
        gs_scores = []
        for nid in node_ids:
            node = self.engine.nodes.get(nid)
            if node is not None and node.gray_scale:
                gs_scores.append(self.engine.gray_validator.composite_score(node.gray_scale))
        gs_min = min(gs_scores) if gs_scores else 0.0
        gs_avg = sum(gs_scores) / len(gs_scores) if gs_scores else 0.0

        verdict = GateVerdict("", "")
        verdict.node_verdicts = {nid: (v, d) for nid, v, d in approved + flagged}
        verdict.axiom_chains = [d for _, v, d in approved]
        verdict.gray_scale_min = gs_min
        verdict.gray_scale_avg = gs_avg

        if len(approved) == len(node_ids):
            verdict.verdict = "APPROVED"
            verdict.reason = (f"linaje completo en {len(approved)}/{len(node_ids)} nodos "
                              f"(origen verificable en el registro de axiomas)")
        elif len(approved) == 0:
            verdict.verdict = "VETOED"
            verdict.reason = f"0/{len(node_ids)} nodos con linaje verificable"
        else:
            verdict.verdict = "FLAGGED"
            verdict.reason = (f"linaje parcial: {len(approved)}/{len(node_ids)} verificados; "
                              f"rotura en: {[nid[:16] for nid, _, _ in flagged]}")
        return verdict
