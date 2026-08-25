"""Axion Brief Enricher — Capa 1 del protocolo automático (2026-08-10).

Antes de lanzar un subagente/cron, enriquece el brief con memoria verificada:
    MARP router (clasifica dominio) → cube_move (τ=0.60, 2D+3D) →
    cadena de color (linaje) → gate de validez (APPROVED/FLAGGED)

Modo asesor: enriquece y marca; NUNCA bloquea (Capa 3, veto duro, está
diferida hasta que el grafo tenga ≥1000 nodos y la tasa de falsos FLAGGED
sea medible — decisión del usuario 2026-08-10).

API (estable):
    enriched, used = enrich_brief(topic, task_brief)
    - enriched: brief original + bloque de contexto verificado
    - used:     jerarquías inyectadas (para mark_used tras integración)

Fallback: si el motor omega falla, degrada al enriquecedor simple
(consulta directa a la memoria axiomática), nunca rompe la tarea.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from memory_engine import AxiomaticMemoryEngine

MEMORY_PATH = Path(PROJECT_ROOT) / "memory" / "unified_memory.json"

_ICONS = {"AXIOM": "📜", "CONCEPT": "💡", "INSTANCE": "📌", "SESSION": "💬"}

# singleton diferido del orquestador (carga el grafo una sola vez por proceso)
_ORCH = None


def _get_orchestrator():
    """Cablea el stack completo una sola vez por proceso (igual que ab_gen)."""
    global _ORCH
    if _ORCH is None:
        from omega_cube.engine import OmegaCubeEngine
        from omega_cube.embeddings import SemanticEmbedder
        from omega_cube.cube_move import CubeMover
        from omega_cube.color_chain import ColorChain
        from omega_cube.validity_gate import ValidityGate
        from omega_cube.orchestrator import AxiomaOrchestrator

        engine = OmegaCubeEngine()
        chain = ColorChain(engine)
        chain.assign_axiom_hues()
        chain.propagate()
        gate = ValidityGate(engine, chain)
        embedder = SemanticEmbedder(engine.memory_dir)
        embedder.embed_nodes(engine.nodes)
        mover = CubeMover(engine, embedder, color_chain=chain)

        marp = None
        try:
            from omega_cube.marp.gpu_router import QwenGPURouter
            marp = QwenGPURouter()
        except Exception:
            pass  # MARP opcional: el pipeline funciona sin él

        _ORCH = AxiomaOrchestrator(engine, mover, chain, gate, marp_router=marp)
    return _ORCH


def _get_engine() -> AxiomaticMemoryEngine:
    engine = AxiomaticMemoryEngine()
    if MEMORY_PATH.exists():
        engine.load(str(MEMORY_PATH))
    return engine


# ---------------------------------------------------------------- fallback
def _retrieve_simple(topic: str, top_k: int = 8) -> list[dict]:
    """Enriquecedor simple (pre-Fase 4): consulta directa a Axioma."""
    engine = _get_engine()
    return engine.query(topic)[:top_k]


def _format_simple(results: list[dict], char_budget: int = 1500) -> str:
    if not results:
        return ""
    lines = ["## 🧠 Contexto de Axioma-Omega (memoria del usuario)", ""]
    used_chars = 0
    seen = set()
    for r in results:
        hier = r["hierarchy"]
        if hier in seen:
            continue
        seen.add(hier)
        icon = _ICONS.get(r["node_type"], "▸")
        entry = f"{icon} [{r['node_type']}] {hier}\n   {r['content']}"
        if used_chars + len(entry) > char_budget:
            break
        lines.append(entry)
        lines.append("")
        used_chars += len(entry)
    if len(lines) <= 2:
        return ""
    lines.append("_Contexto de la memoria verificada del usuario. Úsalo como base._")
    return "\n".join(lines)


# ---------------------------------------------------------------- capa 1
def _enrich_via_orchestrator(topic: str, task_brief: str, top_k: int, char_budget: int):
    """Pipeline completo. Devuelve (enriched, used) o None si debe degradar."""
    t0 = time.perf_counter()
    orch = _get_orchestrator()
    res = orch.build_brief(topic, k=top_k, tau=0.60)

    brief_block = res.get("brief", "")
    if not brief_block or not res.get("node_ids"):
        return None  # nada relevante → que el fallback simple decida

    # recortar si excede el presupuesto
    if len(brief_block) > char_budget:
        brief_block = brief_block[:char_budget] + "\n…(recortado por presupuesto)"

    verdict = res.get("verdict", "?")
    marp = res.get("marp") or {}
    if marp.get("domains"):
        marp_str = f"{marp['domains'][0]} ({marp.get('confidence', '?')})"
    elif marp.get("error"):
        marp_str = f"MARP caído ({marp['error'][:30]})"
    else:
        marp_str = "sin MARP"
    elapsed = (time.perf_counter() - t0) * 1000

    header = (
        f"## 🧠 Contexto verificado (protocolo Axioma-Omega)\n"
        f"_MARP: {marp_str} · gate: {verdict} · {len(res['node_ids'])} nodos · {elapsed:.0f}ms_"
    )
    if verdict == "FLAGGED":
        header += "\n_⚠ Parte del ensamblado no tiene linaje verificable; tratar como referencia, no como hecho._"

    enriched = f"{task_brief}\n\n{header}\n\n{brief_block}"

    # jerarquías inyectadas (para mark_used)
    used = []
    for nid in res["node_ids"]:
        node = orch.engine.nodes.get(nid)
        if node and node.hierarchies:
            used.append(node.hierarchies[0])
    return enriched, list(dict.fromkeys(used))


# ---------------------------------------------------------------- API
def enrich_brief(topic: str, task_brief: str, top_k: int = 8, char_budget: int = 1500):
    """API principal (Capa 1 automática): enriquecer un brief con memoria verificada.

    Returns:
        (enriched_brief, used_hierarchies)
    """
    try:
        out = _enrich_via_orchestrator(topic, task_brief, top_k, char_budget)
        if out is not None:
            return out
    except Exception as e:  # degradación segura: nunca romper la tarea
        print(f"[enricher] orquestador falló ({e}); degradando a enriquecedor simple", file=sys.stderr)

    results = _retrieve_simple(topic, top_k=top_k)
    block = _format_simple(results, char_budget=char_budget)
    if not block:
        return task_brief, []
    used = list(dict.fromkeys(r["hierarchy"] for r in results))
    return f"{task_brief}\n\n{block}", used


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "issue EB-001 de EvonyBot-Pro"
    brief = "Tarea: analiza el problema y propón el siguiente paso."
    enriched, used = enrich_brief(topic, brief)
    print(f"Tema: {topic}\nNodos inyectados: {len(used)}\n{'=' * 60}")
    print(enriched)
