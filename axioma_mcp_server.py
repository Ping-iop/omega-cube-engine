"""Axioma-Omega MCP Server - FastMCP implementation.

Expose axiomatic memory engine tools via MCP for Hermes Agent integration.
Supports: query, learn, associate, stats, tree, list axioms, hierarchy retrieval.
"""

import sys
from pathlib import Path

# Add project root to path so we can import memory_engine
PROJECT_ROOT = str(Path(__file__).parent.resolve())
sys.path.insert(0, PROJECT_ROOT)

from memory_engine import AxiomaticMemoryEngine
from session_context_engine import SessionContextEngine
from mcp.server.fastmcp import FastMCP


# Initialize MCP server
mcp = FastMCP("axioma-omega")

# Singleton engine instance - uses SessionContextEngine (extends AxiomaticMemoryEngine)
_engine_path = Path(__file__).parent / "memory" / "unified_memory.json"
engine = SessionContextEngine(str(Path(__file__).parent / "memory"))
if _engine_path.exists():
    engine.load(str(_engine_path))


@mcp.tool()
def axioma_query(query: str, max_depth: int = 5) -> str:
    """Búsqueda axiomática por navegación jerárquica.

    query: texto de búsqueda
    max_depth: profundidad máxima de navegación (default: 5)
    """
    results = engine.query(query, max_depth=max_depth)
    if not results:
        return "No se encontraron resultados. El motor está vacío o no hay coincidencias."
    
    lines = [f"🔍 Resultados para: '{query}'\n"]
    for r in results[:15]:
        icon = {"AXIOM": "📜", "CONCEPT": "💡", "INSTANCE": "📌"}.get(r["node_type"], "📄")
        lines.append(
            f"{icon} [{r['node_type']}] score={r['score']:.2f}\n"
            f"   Jerarquía: {r['hierarchy']}\n"
            f"   Contenido: {r['content'][:300]}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def axioma_learn(content: str, hierarchy: str, node_type: str = "CONCEPT", tags: str = "") -> str:
    """Aprender nuevo conocimiento (axioma/concepto/instancia).

    content: contenido del conocimiento
    hierarchy: ruta jerárquica (ej: FISICA.TERMODINAMICA.LEYES)
    node_type: AXIOM, CONCEPT, o INSTANCE (default: CONCEPT)
    tags: lista de tags separados por coma (opcional)
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    
    if node_type == "AXIOM":
        engine.add_axiom(content, hierarchy, tag_list)
    elif node_type == "INSTANCE":
        engine.add_instance(content, hierarchy, tag_list)
    else:
        engine.add_concept(content, hierarchy, tag_list)
    
    # Auto-save
    engine.save()
    
    return f"✅ Aprendido [{node_type}]: {hierarchy}\n   Contenido: {content[:100]}..."


@mcp.tool()
def axioma_associate(hierarchy1: str, hierarchy2: str) -> str:
    """Crear asociación entre dos nodos.

    hierarchy1: primer nodo (ej: VENTAS.SANDLER.FILOSOFIA)
    hierarchy2: segundo nodo (ej: PSICOLOGIA.DECISIONES.KAHNEMAN)
    """
    success = engine.associate(hierarchy1, hierarchy2)
    if success:
        engine.save()
        return f"✅ Asociados: {hierarchy1} ↔ {hierarchy2}"
    else:
        missing = []
        if hierarchy1 not in engine.nodes:
            missing.append(f"{hierarchy1} (no existe)")
        if hierarchy2 not in engine.nodes:
            missing.append(f"{hierarchy2} (no existe)")
        return f"❌ No se pudo asociar. Nodos faltantes: {', '.join(missing)}"


@mcp.tool()
def axioma_stats() -> str:
    """Estadísticas del grafo de memoria axiomática."""
    stats = engine.stats()
    lines = [
        "📊 Estadísticas de Memoria Axiomática",
        "=" * 40,
        f"Total nodos: {stats['total_nodos']}",
        f"📜 Axiomas (verdades absolutas): {stats['axiomas']}",
        f"💡 Conceptos (conocimiento estructurado): {stats['conceptos']}",
        f"📌 Instancias (datos específicos): {stats['instancias']}",
        f"💬 Sesiones indexadas: {stats.get('sesiones', 0)}",
        f"   Sesiones activas: {len(engine.list_sessions())}",
        f"📁 Directorio: {stats['memory_dir']}",
    ]
    return "\n".join(lines)


@mcp.tool()
def axioma_tree(domain: str = None) -> str:
    """Visualizar árbol jerárquico.

    domain: filtrar por dominio (opcional, ej: FOTOGRAFIA.CAMERA)
    """
    stats = engine.stats()
    if stats["total_nodos"] == 0:
        return "📭 Memoria vacía - no hay nodos para mostrar."
    
    tree_data = engine.tree(domain)
    
    lines = ["🌳 Árbol Jerárquico de Memoria Axiomática\n"]
    
    def render_tree(data, indent=0):
        for key, value in data.items():
            if key == "_node":
                continue
            node_info = value.get("_node", {})
            icon = {"AXIOM": "📜", "CONCEPT": "💡", "INSTANCE": "📌", "SESSION": "💬"}.get(
                node_info.get("type", ""), "📂"
            )
            prefix = "  " * indent
            lines.append(f"{prefix}{icon} {key}")
            preview = node_info.get("content_preview")
            if preview:
                lines.append(f"{prefix}   \"{preview}...\"")
            sub = {k: v for k, v in value.items() if k != "_node"}
            if sub:
                render_tree(sub, indent + 1)
    
    render_tree(tree_data)
    return "\n".join(lines[:100])  # Limit output


@mcp.tool()
def axioma_axioms(domain: str = None) -> str:
    """Listar todas las verdades absolutas (axiomas).

    domain: filtrar por dominio (opcional)
    """
    axioms = engine.retrieve_axioms(domain)
    if not axioms:
        return "📭 No hay axiomas registrados." + (f" en el dominio '{domain}'" if domain else "")
    
    lines = [f"📜 Axiomas ({len(axioms)}):"]
    for a in axioms[:20]:
        lines.append(
            f"  {a['hierarchy']}\n"
            f"    \"{a['content'][:150]}\"\n"
            f"    Tags: {', '.join(a.get('tags', []))}"
        )
    return "\n".join(lines)


@mcp.tool()
def axioma_hierarchy(hierarchy_prefix: str) -> str:
    """Recuperar todos los nodos bajo una ruta jerárquica.

    hierarchy_prefix: prefijo jerárquico (ej: FOTOGRAFIA.CAMERA.NIKON.Z6II)
    """
    nodes = engine.retrieve_by_hierarchy(hierarchy_prefix)
    if not nodes:
        return f"No se encontraron nodos bajo '{hierarchy_prefix}'"
    
    lines = [f"📂 Nodos bajo '{hierarchy_prefix}' ({len(nodes)}):\n"]
    for n in nodes[:20]:
        icon = {"AXIOM": "📜", "CONCEPT": "💡", "INSTANCE": "📌", "SESSION": "💬"}.get(
            n["node_type"], "📄"
        )
        lines.append(
            f"{icon} {n['hierarchy']} [{n['node_type']}]\n"
            f"   \"{n['content'][:200]}\"\n"
        )
    return "\n".join(lines)


# ─── Session Context Tools ───────────────────────────────────────────

@mcp.tool()
def session_index_turn(
    session_id: str,
    user_message: str,
    assistant_response: str,
    turn_number: int,
) -> str:
    """Indexar un turno de conversación en el grafo de sesión.

    Extrae decisiones, paths, comandos, configuraciones e insights 
    del turno y los almacena en la jerarquía SESSION.<session_id>.
    
    session_id: ID de la sesión (ej: 'abc123')
    user_message: mensaje del usuario
    assistant_response: respuesta del asistente
    turn_number: número del turno
    """
    stats = engine.index_turn(session_id, user_message, assistant_response, turn_number)
    engine.save()
    
    items = []
    if stats["decisions"]: items.append(f"📋 {stats['decisions']} decisiones")
    if stats["paths"]: items.append(f"📁 {stats['paths']} paths")
    if stats["commands"]: items.append(f"⚡ {stats['commands']} comandos")
    if stats["configs"]: items.append(f"⚙️ {stats['configs']} configs")
    if stats["insights"]: items.append(f"💡 {stats['insights']} insights")
    
    return f"✅ Turno {turn_number} indexado en sesión '{session_id}'\n" + \
           ("   " + ", ".join(items) if items else "   (sin entidades detectadas)")


@mcp.tool()
def session_retrieve_context(session_id: str, query: str, max_results: int = 10) -> str:
    """Recuperar contexto relevante de una sesión indexada.

    Busca en el subgrafo SESSION.<session_id> usando navegación jerárquica.
    Útil cuando el contexto lineal del chat ya no contiene información
    del principio de la conversación.

    session_id: ID de la sesión
    query: texto de búsqueda (lo que necesitás recordar)
    max_results: máximo de resultados (default: 10)
    """
    results = engine.retrieve_session_context(session_id, query, max_results)
    if not results:
        return f"🔍 No se encontró contexto para '{query}' en la sesión '{session_id}'"
    
    lines = [f"🔍 Contexto recuperado para: '{query}' ({len(results)} resultados)\n"]
    for r in results:
        cat = r["hierarchy"].split(".")[2] if len(r["hierarchy"].split(".")) > 2 else "?"
        icon = {
            "DECISIONS": "📋", "PATHS": "📁", "COMMANDS": "⚡",
            "CONFIG": "⚙️", "INSIGHTS": "💡", "STATE": "📌"
        }.get(cat, "📄")
        lines.append(
            f"{icon} score={r['score']:.2f} [{cat}]\n"
            f"   {r['content'][:250]}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def session_checkpoint(session_id: str, state_summary: str) -> str:
    """Guardar un checkpoint del estado actual de la conversación.

    Útil para retomar después de cambios de tema o sesiones largas.
    Solo mantiene el checkpoint más reciente (sobrescribe STATE).

    session_id: ID de la sesión
    state_summary: resumen del estado actual (qué se está haciendo, en qué punto)
    """
    ok = engine.checkpoint(session_id, state_summary)
    if ok:
        engine.save()
        return f"✅ Checkpoint guardado en sesión '{session_id}':\n   {state_summary[:200]}"
    return "❌ Error al guardar checkpoint"


@mcp.tool()
def session_summary(session_id: str) -> str:
    """Obtener un resumen completo de la sesión indexada.

    Devuelve conteos y listados de decisiones, paths, comandos,
    configuraciones e insights de la sesión, más el último checkpoint.

    session_id: ID de la sesión
    """
    s = engine.get_session_summary(session_id)
    if s["total_nodes"] == 0:
        return f"📭 Sesión '{session_id}' no encontrada o vacía"
    
    lines = [
        f"📊 Resumen de sesión: '{session_id}'",
        "=" * 50,
        f"Total nodos: {s['total_nodes']}",
    ]
    if s["turn_range"]:
        lines.append(f"Turnos indexados: {s['turn_range'][0]} → {s['turn_range'][1]}")
    
    if s["state"]:
        lines.append(f"\n📌 Último checkpoint:\n   {s['state'][:200]}")
    
    if s["decisions"]:
        lines.append(f"\n📋 Decisiones ({len(s['decisions'])}):")
        for d in s["decisions"][:5]:
            lines.append(f"   ▸ {d['content'][:150]}")
    
    if s["paths"]:
        lines.append(f"\n📁 Paths ({len(s['paths'])}):")
        for p in s["paths"][:5]:
            lines.append(f"   ▸ {p['content']}")
    
    if s["commands"]:
        lines.append(f"\n⚡ Comandos ({len(s['commands'])}):")
        for c in s["commands"][:5]:
            lines.append(f"   ▸ {c['content'][:100]}")
    
    if s["insights"]:
        lines.append(f"\n💡 Insights ({len(s['insights'])}):")
        for i in s["insights"][:5]:
            lines.append(f"   ▸ {i['content'][:150]}")
    
    return "\n".join(lines)


@mcp.tool()
def session_list() -> str:
    """Listar todas las sesiones activas en el grafo."""
    sessions = engine.list_sessions()
    if not sessions:
        return "📭 No hay sesiones indexadas"
    
    lines = [f"💬 Sesiones indexadas ({len(sessions)}):\n"]
    for sid in sessions:
        prefix = f"SESSION.{sid}."
        node_count = sum(1 for h in engine.nodes if h.startswith(prefix))
        # Check for state
        has_checkpoint = any(
            h.endswith(".STATE") and h.startswith(prefix)
            for h in engine.nodes
        )
        ck = " 📌" if has_checkpoint else ""
        lines.append(f"   ▸ {sid} ({node_count} nodos){ck}")
    return "\n".join(lines)


@mcp.tool()
def session_clear(session_id: str) -> str:
    """Eliminar todos los nodos de una sesión.

    session_id: ID de la sesión a eliminar
    """
    count = engine.clear_session(session_id)
    if count > 0:
        engine.save()
        return f"🗑️ Sesión '{session_id}' eliminada ({count} nodos)"
    return f"📭 Sesión '{session_id}' no encontrada"


@mcp.tool()
def axioma_telemetry() -> str:
    """Métrica de salud de la memoria: recalls vs usages.

    recalls = veces que se consultó la memoria (axioma_query)
    usages  = veces que el consumidor confirmó haber INTEGRADO lo recuperado
    ALERTA si recalls>0 con usages=0: memoria decorativa.
    """
    t = engine.telemetry()
    lines = [
        f"📈 Telemetría Axioma-Omega",
        f"   Recalls (consultas): {t['recalls']}",
        f"   Usages (integradas): {t['usages']}",
        f"   Salud: {t['health']}",
    ]
    if t["last_recalls"]:
        lines.append("\n   Últimos recalls:")
        for r in t["last_recalls"]:
            lines.append(f"     [{r['ts']}] \"{r['query']}\" -> {len(r['hits'])} hits")
    if t["last_usages"]:
        lines.append("\n   Últimos usages:")
        for u in t["last_usages"]:
            lines.append(f"     [{u['ts']}] {u['hierarchies'][:3]} — {u['context'][:60]}")
    return "\n".join(lines)


@mcp.tool()
def axioma_mark_used(hierarchies: str, context: str = "") -> str:
    """Confirmar que nodos recuperados fueron INTEGRADOS en el razonamiento.

    jerarchies: rutas separadas por coma (ej: FISICA.FUEGO,VENTAS.SANDLER)
    context: breve descripción de cómo se usó (opcional)

    Sin esta llamada, la telemetría mostrará recalls>0/usages=0 (memoria decorativa).
    """
    hier_list = [h.strip() for h in hierarchies.split(",") if h.strip()]
    if not hier_list:
        return "❌ Necesito al menos una jerarquía"
    engine.mark_used(hier_list, context)
    return f"✅ Usage registrado ({len(hier_list)} nodos): {hier_list[:3]}"


if __name__ == "__main__":
    print(f"🚀 Starting Axioma-Omega MCP server...")
    print(f"   Project root: {PROJECT_ROOT}")
    print(f"   Memory dir: {engine.memory_dir}")
    
    stats = engine.stats()
    print(f"   Loaded nodes: {stats['total_nodos']} "
          f"(Axioms: {stats['axiomas']}, Concepts: {stats['conceptos']})")
    
    mcp.run()
