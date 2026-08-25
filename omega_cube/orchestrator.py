"""
orchestrator.py — Fase 4 (PLAN 2026-08-09): orquestador que consulta Axioma
antes de lanzar subagentes y valida su salida por linaje de color.

Flujo:
  0. MARP router (Qwen0.8B, puerto 8082) clasifica el dominio de la tarea
  1. cube_move(task) → nodos relevantes (2D + giros 3D)
  2. identificar la(s) cadena(s) de tono dominante(s) del ensamblado
  3. componer el brief enriquecido: axiomas del dominio + nodos verificados
  4. (el subagente ejecuta)
  5. ValidityGate sobre los nodos que el subagente afirma usar → APPROVED/FLAGGED

Extiende axion_brief_enricher.py (que usaba query plana) con cube_move + gate.
"""

from __future__ import annotations


class AxiomaOrchestrator:
    def __init__(self, engine, mover, color_chain, gate, marp_router=None):
        self.engine = engine
        self.mover = mover
        self.chain = color_chain
        self.gate = gate
        self.marp = marp_router  # QwenGPURouter opcional (puerto 8082)

    def build_brief(self, task: str, k: int = 12, tau: float = 0.6) -> dict:
        """Compone el brief enriquecido para un subagente.

        Returns dict con: brief (str), node_ids, verdict, chains, marp (clasificación).
        """
        # Paso 0: MARP router clasifica el dominio de la tarea (~250ms GPU,
        # 0.1ms si cae en keyword pre-filter). Informativo + diagnóstico.
        marp_info = None
        if self.marp is not None:
            try:
                domains, conf = self.marp.classify(task)
                marp_info = {"domains": domains, "confidence": conf}
            except Exception as e:
                marp_info = {"error": str(e)}

        res = self.mover.cube_move(task, k=k, tau=tau)
        node_ids = res.all_node_ids()

        # cadenas de tono presentes en el ensamblado, PONDERADAS por score
        # (contar nodos por cadena metía ruido: cadenas con 1 nodo débil
        # arrastraban su axioma al brief — observado en MASTERFLUX 2026-08-10)
        chains: dict[str, float] = {}
        for nid in node_ids:
            node = self.engine.nodes.get(nid)
            if node is not None and node.hue_origin:
                chains[node.hue_origin] = chains.get(node.hue_origin, 0.0) + \
                    res.scores.get(nid, 0.0)

        # axiomas relevantes: por peso de cadena descendente, solo cadenas con
        # masa de relevancia ≥ 0.6 (≈ 1 nodo sólido) + axiomas en el ensamblado
        axiom_ids = [aid for aid, w in sorted(chains.items(), key=lambda x: -x[1])
                     if w >= 0.6]
        for nid in node_ids:
            node = self.engine.nodes.get(nid)
            if node is not None and node.node_type == "AXIOM" and nid not in axiom_ids:
                axiom_ids.append(nid)

        # verificación del ensamblado ANTES de inyectar
        v = self.gate.validate_response(node_ids) if node_ids else None
        approved_ids = [nid for nid, (vv, _) in v.node_verdicts.items()
                        if vv == "APPROVED"] if v else []

        # composición del brief: axiomas primero (verdades ancla), luego nodos
        lines = [f"## CONTEXTO AXIOMÁTICO (Axioma-Omega) para la tarea: {task}",
                 ""]
        lines.append("### Verdades ancla del dominio (axiomas verificados):")
        for aid in axiom_ids:
            ax = self.engine.nodes.get(aid)
            if ax:
                lines.append(f"- [{ax.primary_hierarchy}] {ax.content}")
        lines.append("")
        lines.append("### Conocimiento derivado verificado (linaje completo):")
        for nid in approved_ids:
            if nid in axiom_ids:
                continue
            node = self.engine.nodes.get(nid)
            if node:
                sat = f"sat={node.saturation:.2f}" if node.saturation is not None else ""
                lines.append(f"- [{node.primary_hierarchy}] {node.content[:400]} ({sat})")
        lines.append("")
        lines.append(f"### Verificación del ensamblado: {v.verdict if v else 'N/A'} — "
                     f"{v.reason if v else ''}")
        lines.append("Regla: si tu respuesta usa hechos fuera de este contexto, "
                     "indícalo explícitamente como NO VERIFICADO.")

        return {
            "brief": "\n".join(lines),
            "node_ids": node_ids,
            "approved_ids": approved_ids,
            "axiom_ids": axiom_ids,
            "verdict": v.verdict if v else "VETOED",
            "chains": dict(chains),
            "marp": marp_info,
            "cube_move_result": res,
        }

    def validate_subagent_response(self, claimed_node_ids: list[str]) -> dict:
        """Valida la salida del subagente: ¿los nodos que afirma haber usado
        tienen linaje verificable?"""
        v = self.gate.validate_response(claimed_node_ids)
        return {
            "verdict": v.verdict,
            "reason": v.reason,
            "summary": v.summary(),
        }
