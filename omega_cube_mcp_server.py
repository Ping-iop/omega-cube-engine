"""
Omega-Cube MCP Server — FastMCP implementation.

Exposes Omega-Cube Engine tools via MCP for plug-and-play 
integration with Hermes Agent, Claude, GPT, and any MCP-compatible agent.

Tools:
- omega_cube_query          — Multi-mode graph retrieval
- omega_cube_multi_topic    — Per-topic parallel retrieval
- omega_cube_patterns       — Cross-domain pattern detection
- omega_cube_learn          — Multi-dimensional knowledge ingestion
- omega_cube_associate      — Create associations between nodes
- omega_cube_stats          — Engine statistics
- omega_cube_diffuse        — Diffusion-based graph-to-text generation
- omega_cube_verify         — Gray-scale node verification
"""

import sys
import os
from pathlib import Path

# Ensure omega_cube package is importable regardless of launch directory
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent  # axioma-omega-protocol/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from omega_cube import OmegaCubeEngine
from mcp.server.fastmcp import FastMCP

# Initialize
mcp = FastMCP("omega-cube")

# Singleton engine
_engine_path = Path(__file__).parent / "memory" / "omega_cube_memory.json"
engine = OmegaCubeEngine()
if _engine_path.exists():
    engine.load(str(_engine_path))


@mcp.tool()
def omega_cube_query(
    query: str,
    mode: str = "diffusion",
    top_k: int = 10,
    temperature: float = 0.1,
) -> str:
    """Query Omega-Cube with multi-dimensional graph retrieval.

    query: search query text
    mode: retrieval mode - 'diffusion' (parallel, precise), 'holographic' (fast O(1)),
          'tensor' (spatial), 'combined' (best overall), 'annealing' (pattern discovery)
    top_k: number of results (default: 10)
    temperature: noise level 0.0-1.0 (lower = more deterministic, default: 0.1)
    """
    try:
        results = engine.query(query, mode=mode, top_k=top_k, temperature=temperature)
    except ValueError as e:
        return f"❌ Invalid mode: {e}. Valid modes: diffusion, holographic, tensor, combined, annealing"
    
    if not results:
        return f"🔍 No results for '{query}' (mode: {mode})"
    
    lines = [f"🔍 Omega-Cube: '{query}' ({mode} mode, {len(results)} results)\n"]
    for r in results:
        gs = r.get("gray_scale_composite", 50)
        gs_bar = "█" * int(gs / 10) + "░" * (10 - int(gs / 10))
        lines.append(
            f"📊 [{r['score']:.3f}] [{r['node_type']}] GS:{gs:.0f}% {gs_bar}\n"
            f"   Primary: {r['primary_hierarchy']}\n"
            f"   Dims: {', '.join(r['hierarchies'][:3])}\n"
            f"   {r['content'][:250]}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def omega_cube_multi_topic(
    query: str,
    topics: str,
    top_k_per_topic: int = 3,
) -> str:
    """Multi-topic parallel retrieval. Returns results organized by topic domain.

    query: search query text
    topics: comma-separated topic prefixes (e.g., 'COMFYUI,EVONY,HERMES')
    top_k_per_topic: results per topic (default: 3)
    """
    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
    if not topic_list:
        return "❌ No topics provided. Use comma-separated hierarchy prefixes."
    
    results = engine.query_multi_topic(query, topic_list, top_k_per_topic)
    
    lines = [f"🔍 Omega-Cube Multi-Topic: '{query}'\n"]
    for topic, hits in results.items():
        lines.append(f"\n## {topic} ({len(hits)} results)")
        for h in hits:
            lines.append(
                f"   [{h['score']:.3f}] {h['content'][:150]}"
            )
        if not hits:
            lines.append("   (no results for this topic)")
    
    return "\n".join(lines)


@mcp.tool()
def omega_cube_patterns(
    query: str,
    min_strength: float = 0.3,
) -> str:
    """Discover emergent cross-domain patterns via topology annealing.

    query: search query or topic of interest
    min_strength: minimum pattern strength 0.0-1.0 (default: 0.3)
    """
    patterns = engine.find_patterns(query, min_strength=min_strength)
    
    if not patterns:
        return f"🔮 No patterns detected for '{query}' (min_strength={min_strength})"
    
    lines = [f"🔮 Omega-Cube Pattern Emergence: '{query}' ({len(patterns)} patterns)\n"]
    for i, p in enumerate(patterns[:10]):
        lines.append(
            f"\n## Pattern {i+1}: {p.get('cube_topic', 'Unknown')} "
            f"(strength: {p.get('pattern_strength', 0):.3f})"
        )
        for ac in p.get("aligned_cubes", [])[:3]:
            lines.append(
                f"   ↔ {ac.get('cube_id', '?')} "
                f"(alignment: {ac.get('alignment', 0):.3f})"
            )
    
    return "\n".join(lines)


@mcp.tool()
def omega_cube_learn(
    content: str,
    hierarchies: str,
    node_type: str = "CONCEPT",
    confidence: float = 0.9,
    tags: str = "",
) -> str:
    """Ingest knowledge into Omega-Cube with multi-dimensional hierarchies.

    content: knowledge content
    hierarchies: hierarchy paths separated by | (e.g., 'DIM1.PATH|DIM2.PATH|DIM3.PATH')
    node_type: AXIOM, CONCEPT, or INSTANCE (default: CONCEPT)
    confidence: initial confidence 0.0-1.0 (default: 0.9)
    tags: comma-separated tags (optional)
    """
    hier_list = [h.strip() for h in hierarchies.split("|") if h.strip()]
    if not hier_list:
        return "❌ No hierarchies provided. Use | to separate hierarchy paths."
    
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    
    node = engine.add_node(
        content=content,
        hierarchies=hier_list,
        node_type=node_type.upper(),
        confidence=confidence,
        tags=tag_list,
    )
    
    engine.save()
    
    gs = node.gray_scale or {}
    gs_composite = engine.gray_validator.composite_score(gs)
    
    return (
        f"✅ Learned [{node_type.upper()}] in {len(hier_list)} dimensions\n"
        f"   ID: {node.node_id}\n"
        f"   Hierarchies: {', '.join(hier_list)}\n"
        f"   Gray-Scale: {gs_composite:.0f}%\n"
        f"   Content: {content[:150]}..."
    )


@mcp.tool()
def omega_cube_associate(node_id1: str, node_id2: str) -> str:
    """Create lateral association between two nodes.

    node_id1: first node ID
    node_id2: second node ID
    """
    success = engine.associate(node_id1, node_id2)
    if success:
        engine.save()
        return f"✅ Associated: {node_id1} ↔ {node_id2}"
    else:
        missing = []
        if node_id1 not in engine.nodes:
            missing.append(f"{node_id1} (not found)")
        if node_id2 not in engine.nodes:
            missing.append(f"{node_id2} (not found)")
        return f"❌ Cannot associate: {', '.join(missing)}"


@mcp.tool()
def omega_cube_stats() -> str:
    """Get Omega-Cube engine statistics."""
    s = engine.stats()
    lines = [
        "📊 Omega-Cube Engine Statistics",
        "=" * 45,
        f"Total nodes: {s['total_nodes']}",
        f"📜 Axioms: {s['axioms']}",
        f"💡 Concepts: {s['concepts']}",
        f"📌 Instances: {s['instances']}",
        f"💬 Sessions: {s['sessions']}",
        f"📐 Avg dimensions/node: {s['avg_dimensions_per_node']}",
        f"🔮 Holographic dim: {s['holographic_dim']}",
        f"🔍 Queries: {s['query_count']}",
        f"⚡ Avg retrieval: {s['avg_retrieval_time_ms']}ms",
        f"📁 Memory: {s['memory_dir']}",
    ]
    return "\n".join(lines)


@mcp.tool()
def omega_cube_verify(
    node_id: str,
    dimensions: str = "",
) -> str:
    """Verify a node's gray-scale truth profile (H-Bit inspired).

    node_id: node ID to verify
    dimensions: comma-separated dimensions to check (empty = all 6)
    """
    if node_id not in engine.nodes:
        return f"❌ Node '{node_id}' not found"
    
    node = engine.nodes[node_id]
    
    # Recompute gray-scale
    node.gray_scale = engine.gray_validator.evaluate_node(
        node, axioms=engine.axioms
    )
    
    dim_list = [d.strip() for d in dimensions.split(",") if d.strip()] if dimensions else []
    
    lines = [
        f"🔬 Gray-Scale Verification: {node_id}",
        f"   Content: {node.content[:120]}...",
        f"   Type: {node.node_type} | Confidence: {node.confidence:.2f}",
        f"",
    ]
    
    gs = node.gray_scale or {}
    all_dims = engine.gray_validator.DIMENSIONS
    
    for dim in all_dims:
        if dim_list and dim not in dim_list:
            continue
        score = gs.get(dim, 50)
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        lines.append(f"   {dim:<15} {bar} {score:.0f}%")
    
    composite = engine.gray_validator.composite_score(gs)
    lines.append(f"\n   {'COMPOSITE':<15} {'█' * int(composite / 10) + '░' * (10 - int(composite / 10))} {composite:.0f}%")
    
    # Partial evidence score (H-Bit principle)
    if dim_list:
        partial = engine.gray_validator.partial_evidence_score(gs, dim_list)
        lines.append(f"   {'PARTIAL (' + str(len(dim_list)) + ' dims)':<15} {'█' * int(partial / 10) + '░' * (10 - int(partial / 10))} {partial:.0f}%")
    
    return "\n".join(lines)


@mcp.tool()
def omega_cube_diffuse(
    query: str,
    steps: int = 20,
    guidance: float = 3.0,
) -> str:
    """Diffusion-based graph-to-text: generate response organized by hierarchy.

    Instead of sequential traversal, diffuses over all nodes simultaneously
    and converges to a hierarchically-organized result. Inspired by DiffusionGemma.

    query: input query
    steps: denoising steps (default: 20, more = more precise)
    guidance: guidance scale (default: 3.0, higher = more focused)
    """
    # Temporarily configure diffusion parameters
    orig_steps = engine.diffusion.num_steps
    orig_guidance = engine.diffusion.guidance_scale
    
    engine.diffusion.num_steps = steps
    engine.diffusion.guidance_scale = guidance
    
    results = engine.query(query, mode="diffusion", top_k=15, temperature=0.1)
    
    engine.diffusion.num_steps = orig_steps
    engine.diffusion.guidance_scale = orig_guidance
    
    if not results:
        return f"🔍 No diffusion results for '{query}'"
    
    # Organize results by hierarchy
    by_topic = {}
    for r in results:
        top_level = r["primary_hierarchy"].split(".")[0] if r["primary_hierarchy"] else "Other"
        if top_level not in by_topic:
            by_topic[top_level] = []
        by_topic[top_level].append(r)
    
    lines = [f"🌀 Omega-Cube Diffusion: '{query}' ({steps} steps, {len(results)} nodes)\n"]
    
    for topic, nodes in by_topic.items():
        lines.append(f"\n## {topic}")
        for n in nodes[:3]:
            gs = n.get("gray_scale_composite", 50)
            lines.append(f"   [{gs:.0f}%] {n['content'][:200]}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"🚀 Omega-Cube MCP Server starting...")
    s = engine.stats()
    print(f"   Nodes: {s['total_nodes']} | Dims avg: {s['avg_dimensions_per_node']}")
    print(f"   Memory: {s['memory_dir']}")
    mcp.run()
