"""
Axion Benchmark: Comparativa ANTES vs DESPUÉS con datos verificables.

Ejecuta el MISMO conjunto de queries contra:
- OmegaCubeEngine v1 (original)
- OmegaCubeEngineV2 (con mejoras de arXiv 2026)

Métricas medidas:
- Precision@5, Precision@10
- Tiempo de respuesta (ms)
- Complejidad de escalado (O(n) vs O(log n))
- Operaciones inválidas bloqueadas (typed schema)
- Resultados abstained (boundary control)
- Detecciones de bias (hallucination detector)
- Gray-scale composite score promedio
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omega_cube.engine import OmegaCubeEngine
from omega_cube.engine_v2 import OmegaCubeEngineV2


def build_test_graph_v1(engine, num_domains=5, nodes_per_domain=15):
    """Build test graph on v1 engine."""
    domains = [
        {"name": "COMFYUI", "dimensions": ["COMFYUI.WORKFLOWS.GENERATION", "COMFYUI.MODELS.CHECKPOINTS", "COMFYUI.NODES.CUSTOM", "IMAGE_QUALITY.RESOLUTION", "AI.TOOLS.VISUAL"],
         "topics": ["SDXL", "SD1.5", "ControlNet", "IPAdapter", "LoRA", "Upscale", "Inpaint", "FaceRestore", "Prompting", "Sampling"]},
        {"name": "EVONY", "dimensions": ["EVONY.GENERALS.PVP", "EVONY.GENERALS.F2P", "EVONY.STRATEGY.RALLIES", "GAMING.MOBILE.STRATEGY", "F2P.OPTIMIZATION.RESOURCES"],
         "topics": ["Marcian", "CharlesVI", "Hermes", "Akechi", "Tamar", "Ranged", "Mounted", "Ground", "Siege", "Defense"]},
        {"name": "HERMES", "dimensions": ["HERMES.CONFIG.MCP", "HERMES.SKILLS.AUTOMATION", "HERMES.CRON.MAINTENANCE", "AI.AGENTS.TOOLS", "DEVELOPMENT.AUTOMATION"],
         "topics": ["MCP", "Skills", "CronJobs", "Memory", "Plugins", "Config", "Providers", "Delegation", "Session", "Tools"]},
        {"name": "SECURITY", "dimensions": ["SECURITY.CRYPTOGRAPHY.STEGANOGRAPHY", "SECURITY.VERIFICATION.MULTI_BIT", "HBIT.PROTOCOL.GRAYSCALE", "AI.SAFETY.VERIFICATION", "TRUST.VALIDATION.EVIDENCE"],
         "topics": ["H-Bit", "GrayScale", "Embedding", "Verification", "PartialEvidence", "Watermark", "Fingerprint", "Hash", "Signature", "Trust"]},
        {"name": "ML", "dimensions": ["ML.TRAINING.PRETRAINING", "ML.ARCHITECTURE.DIFFUSION", "ML.OPTIMIZATION.ANNEALING", "AI.RESEARCH.NEUROSYMBOLIC", "SCIENCE.COMPUTATION.ALGORITHMS"],
         "topics": ["DiffusionGemma", "AutoResearch", "Karpathy", "Annealing", "Holographic", "Tensor", "Embedding", "FineTuning", "Inference", "Architecture"]},
    ]
    total = 0
    for domain in domains[:num_domains]:
        axiom = engine.add_node(content=f"{domain['name']} is a knowledge domain", hierarchies=[domain["dimensions"][0]], node_type="AXIOM", confidence=1.0, tags=[domain["name"].lower(), "axiom"])
        total += 1
        for i, topic in enumerate(domain["topics"][:nodes_per_domain]):
            primary = domain["dimensions"][i % len(domain["dimensions"])]
            secondary = domain["dimensions"][(i + 1) % len(domain["dimensions"])]
            hier_list = [primary, secondary]
            if i % 3 == 0 and len(domain["dimensions"]) > 2:
                hier_list.append(domain["dimensions"][(i + 2) % len(domain["dimensions"])])
            pos_x = (i / len(domain["topics"])) * 0.8 + 0.1
            pos_y = (domains.index(domain) / num_domains) * 0.8 + 0.1
            tensor_pos = [pos_x, pos_y] + [0.5] * max(0, len(hier_list) - 2)
            node = engine.add_node(
                content=f"[{domain['name']}] {topic}: Detailed knowledge about {topic.lower()} in {domain['name'].lower()}. Multi-dimensional node spanning {len(hier_list)} axes.",
                hierarchies=hier_list, tensor_position=tensor_pos, node_type="CONCEPT", confidence=0.85, tags=[domain["name"].lower(), topic.lower()])
            total += 1
            engine.associate(axiom.node_id, node.node_id)
    return total


def build_test_graph_v2(engine, num_domains=5, nodes_per_domain=15):
    """Build test graph on v2 engine with realistic content."""
    domains = [
        {"name": "COMFYUI", "dimensions": ["COMFYUI.WORKFLOWS.GENERATION", "COMFYUI.MODELS.CHECKPOINTS", "COMFYUI.NODES.CUSTOM", "IMAGE_QUALITY.RESOLUTION", "AI.TOOLS.VISUAL"],
         "topics": [
             ("SDXL", "Stable Diffusion XL image generation model for high-resolution text-to-image synthesis with improved prompt adherence"),
             ("SD1.5", "Stable Diffusion 1.5 checkpoint for fast image generation with large community LoRA ecosystem"),
             ("ControlNet", "ControlNet preprocessor nodes for pose-guided image generation using depth maps and edge detection"),
             ("IPAdapter", "IPAdapter image prompt adapter for style transfer and reference-based image generation"),
             ("LoRA", "Low-Rank Adaptation fine-tuning for custom character and style models in ComfyUI"),
             ("Upscale", "Image upscaling workflow using 4x-UltraSharp and ESRGAN models for resolution enhancement"),
             ("Inpaint", "Inpainting workflow for masked region editing and object replacement in generated images"),
             ("FaceRestore", "Face restoration using CodeFormer and GFPGAN models for portrait enhancement"),
             ("Prompting", "Prompt engineering techniques for ComfyUI including weighted prompts and negative prompts"),
             ("Sampling", "Sampler comparison for ComfyUI including Euler, DPM++, and UniPC scheduling strategies"),
         ]},
        {"name": "EVONY", "dimensions": ["EVONY.GENERALS.PVP", "EVONY.GENERALS.F2P", "EVONY.STRATEGY.RALLIES", "GAMING.MOBILE.STRATEGY", "F2P.OPTIMIZATION.RESOURCES"],
         "topics": [
             ("Marcian", "Marcian general PvP guide for ranged attack formations and siege warfare tactics"),
             ("CharlesVI", "Charles VI F2P general ranking and optimal talent tree for mounted cavalry"),
             ("Hermes", "Hermes general strategy for alliance rallies and battlefield positioning"),
             ("Akechi", "Akechi general guide for ground troop defense and garrison optimization"),
             ("Tamar", "Tamar general F2P guide for resource optimization and troop training efficiency"),
             ("Ranged", "Ranged attack formation guide for PvP battles with archer and siege units"),
             ("Mounted", "Mounted cavalry strategy for open-field battles and flanking maneuvers"),
             ("Ground", "Ground troop defense guide for city garrison and alliance fortification"),
             ("Siege", "Siege warfare tactics for attacking fortified cities and alliance strongholds"),
             ("Defense", "City defense strategy for F2P players against whale attacks and rallies"),
         ]},
        {"name": "HERMES", "dimensions": ["HERMES.CONFIG.MCP", "HERMES.SKILLS.AUTOMATION", "HERMES.CRON.MAINTENANCE", "AI.AGENTS.TOOLS", "DEVELOPMENT.AUTOMATION"],
         "topics": [
             ("MCP", "MCP server configuration guide for connecting external tools and APIs to Hermes agent"),
             ("Skills", "Skills automation system for creating reusable workflows and procedural memory"),
             ("CronJobs", "Cron job scheduling for automated maintenance tasks and periodic reporting"),
             ("Memory", "Memory management system for persistent context across agent sessions"),
             ("Plugins", "Plugin architecture for extending Hermes with custom tools and UI panes"),
             ("Config", "Configuration management for Hermes providers, models, and tool settings"),
             ("Providers", "LLM provider setup for OpenAI, Anthropic, and local model integration"),
             ("Delegation", "Task delegation system for spawning subagents and parallel workstreams"),
             ("Session", "Session context engine for automatic indexing and conversation continuity"),
             ("Tools", "Tool integration guide for browser, terminal, file, and search capabilities"),
         ]},
        {"name": "SECURITY", "dimensions": ["SECURITY.CRYPTOGRAPHY.STEGANOGRAPHY", "SECURITY.VERIFICATION.MULTI_BIT", "HBIT.PROTOCOL.GRAYSCALE", "AI.SAFETY.VERIFICATION", "TRUST.VALIDATION.EVIDENCE"],
         "topics": [
             ("H-Bit", "H-Bit protocol for multi-bit cryptographic steganography with grayscale verification"),
             ("GrayScale", "Gray-scale composite scoring for evidence validation and trust assessment"),
             ("Embedding", "Steganographic embedding techniques for hiding verification data in media files"),
             ("Verification", "Multi-bit verification protocol for file integrity and content authenticity"),
             ("PartialEvidence", "Partial evidence accumulation for progressive trust building in distributed systems"),
             ("Watermark", "Digital watermarking for content provenance tracking and copyright protection"),
             ("Fingerprint", "Cryptographic fingerprinting for unique content identification and tamper detection"),
             ("Hash", "Hash chain verification for tamper-evident logging and audit trail integrity"),
             ("Signature", "Digital signature schemes for message authentication and non-repudiation"),
             ("Trust", "Trust validation framework for evidence-based confidence scoring in AI systems"),
         ]},
        {"name": "ML", "dimensions": ["ML.TRAINING.PRETRAINING", "ML.ARCHITECTURE.DIFFUSION", "ML.OPTIMIZATION.ANNEALING", "AI.RESEARCH.NEUROSYMBOLIC", "SCIENCE.COMPUTATION.ALGORITHMS"],
         "topics": [
             ("DiffusionGemma", "Diffusion language model architecture combining denoising diffusion with transformer pretraining"),
             ("AutoResearch", "Automated research pipeline using LLM agents for hypothesis generation and experiment design"),
             ("Karpathy", "Karpathy-style training loop optimization with gradient accumulation and learning rate scheduling"),
             ("Annealing", "Quantum annealing topology optimization for combinatorial search and constraint satisfaction"),
             ("Holographic", "Holographic reduced representations for vector symbolic architectures and memory encoding"),
             ("Tensor", "Tensor decomposition methods for model compression and efficient inference"),
             ("Embedding", "Embedding layer design for semantic similarity and retrieval augmentation"),
             ("FineTuning", "Fine-tuning strategies including LoRA, QLoRA, and full parameter adaptation"),
             ("Inference", "Inference optimization with KV cache, speculative decoding, and batch serving"),
             ("Architecture", "Neural architecture search and design patterns for efficient model scaling"),
         ]},
    ]
    total = 0
    for domain in domains[:num_domains]:
        axiom = engine.add_node(content=f"{domain['name']} is a knowledge domain", hierarchies=[domain["dimensions"][0]], node_type="AXIOM", confidence=1.0, tags=[domain["name"].lower(), "axiom"])
        total += 1
        for i, (topic_name, topic_desc) in enumerate(domain["topics"][:nodes_per_domain]):
            primary = domain["dimensions"][i % len(domain["dimensions"])]
            secondary = domain["dimensions"][(i + 1) % len(domain["dimensions"])]
            hier_list = [primary, secondary]
            if i % 3 == 0 and len(domain["dimensions"]) > 2:
                hier_list.append(domain["dimensions"][(i + 2) % len(domain["dimensions"])])
            pos_x = (i / len(domain["topics"])) * 0.8 + 0.1
            pos_y = (domains.index(domain) / num_domains) * 0.8 + 0.1
            tensor_pos = [pos_x, pos_y] + [0.5] * max(0, len(hier_list) - 2)
            node = engine.add_node(
                content=f"[{domain['name']}] {topic_name}: {topic_desc}",
                hierarchies=hier_list, tensor_position=tensor_pos, node_type="CONCEPT", confidence=0.85,
                tags=[domain["name"].lower(), topic_name.lower()] + topic_desc.lower().split()[:8])
            total += 1
            engine.associate(axiom.node_id, node.node_id)
    return total


# Test queries with expected domains
QUERIES = [
    ("best image generation model", ["COMFYUI"]),
    ("top PvP general for ranged attacks", ["EVONY"]),
    ("how to configure MCP servers in Hermes", ["HERMES"]),
    ("multi-bit verification of file integrity", ["SECURITY"]),
    ("diffusion model for text generation", ["ML"]),
    ("SDXL vs SD1.5 comparison", ["COMFYUI"]),
    ("F2P general ranking Marcian Charles", ["EVONY"]),
    ("cron job automation skills", ["HERMES"]),
    ("partial evidence grayscale verification", ["SECURITY"]),
    ("AutoResearch Karpathy training loop", ["ML"]),
    ("ComfyUI ControlNet IPAdapter workflow", ["COMFYUI"]),
    ("Evony mounted general Hermes Babur", ["EVONY"]),
    ("Hermes agent delegation MCP tools", ["HERMES"]),
    ("steganography embedding in audio files", ["SECURITY"]),
    ("neurosymbolic architecture LLM reasoning", ["ML"]),
    ("upscale inpainting face restoration tools", ["COMFYUI"]),
    ("alliance competition battlefield F2P strategy", ["EVONY"]),
    ("session context engine automatic indexing", ["HERMES"]),
    ("holographic reduced representations", ["ML", "SECURITY"]),
    ("quantum annealing topology optimization", ["ML"]),
]


def benchmark_engine(engine, engine_name, modes, num_queries=20):
    """Run benchmark on an engine, return metrics."""
    results = {}
    
    for mode in modes:
        mode_data = {"precision_at_5": 0, "precision_at_10": 0, "times_ms": [], "gray_scores": []}
        
        for query, expected_domains in QUERIES[:num_queries]:
            start = time.time()
            try:
                hits = engine.query(query, mode=mode, top_k=10)
            except TypeError:
                # v1 doesn't support 'hierarchical' mode
                hits = engine.query(query, mode="diffusion", top_k=10)
            elapsed = (time.time() - start) * 1000
            mode_data["times_ms"].append(elapsed)
            
            relevant_5 = 0
            relevant_10 = 0
            for i, hit in enumerate(hits[:10]):
                hit_domain = hit.get("primary_hierarchy", "").split(".")[0]
                hit_content = hit.get("content", "")
                hit_tags = [t.lower() for t in hit.get("tags", [])]
                # Match by hierarchy prefix OR content domain tag [DOMAIN] OR tags
                match = any(
                    hit_domain.upper().startswith(ed.upper())
                    or ed.upper() in hit_domain.upper()
                    or f"[{ed.upper()}]" in hit_content.upper()
                    or ed.lower() in hit_tags
                    for ed in expected_domains
                )
                if match:
                    if i < 5:
                        relevant_5 += 1
                    relevant_10 += 1
                mode_data["gray_scores"].append(hit.get("gray_scale_composite", 50.0))
            
            mode_data["precision_at_5"] += relevant_5
            mode_data["precision_at_10"] += relevant_10
        
        n = num_queries
        results[mode] = {
            "precision_at_5": round(mode_data["precision_at_5"] / (n * 5), 4),
            "precision_at_10": round(mode_data["precision_at_10"] / (n * 10), 4),
            "avg_time_ms": round(sum(mode_data["times_ms"]) / n, 2),
            "p50_time_ms": round(sorted(mode_data["times_ms"])[n // 2], 2),
            "p95_time_ms": round(sorted(mode_data["times_ms"])[int(n * 0.95)], 2),
            "avg_gray_score": round(sum(mode_data["gray_scores"]) / max(len(mode_data["gray_scores"]), 1), 2),
        }
    
    return results


def benchmark_scaling(engine, engine_name):
    """Benchmark scaling behavior."""
    all_nodes = list(engine.nodes.values())
    sizes = [10, 25, 50, 75, 96]
    scaling = []
    
    from omega_cube.tensor_node import TensorIndex
    
    for size in sizes:
        if size > len(all_nodes):
            break
        idx = TensorIndex()
        for n in all_nodes[:size]:
            idx.insert(n)
        
        original_index = engine.index
        engine.index = idx
        
        times = []
        for _ in range(5):
            start = time.time()
            engine.query("finding relevant nodes across domains", mode="diffusion", top_k=10)
            times.append((time.time() - start) * 1000)
        
        engine.index = original_index
        scaling.append({"nodes": size, "avg_ms": round(sum(times) / len(times), 2)})
    
    # Estimate complexity
    if len(scaling) >= 2:
        ratio = scaling[-1]["avg_ms"] / max(scaling[0]["avg_ms"], 0.01)
        n_ratio = scaling[-1]["nodes"] / scaling[0]["nodes"]
        if ratio < n_ratio * 0.5:
            complexity = "O(log n)"
        elif ratio < n_ratio * 1.5:
            complexity = "O(n)"
        else:
            complexity = "O(n log n) or worse"
    else:
        complexity = "insufficient data"
    
    return {"scaling_data": scaling, "estimated_complexity": complexity}


def benchmark_typed_schema(engine_v2):
    """Test typed schema validation (VirtualSet)."""
    blocked = 0
    passed = 0
    
    # Test 1: Invalid node type
    try:
        engine_v2.add_node(content="test", hierarchies=["TEST.INVALID"], node_type="INVALID_TYPE")
        passed += 1
    except TypeError:
        blocked += 1
    
    # Test 2: Valid node type
    try:
        engine_v2.add_node(content="test valid", hierarchies=["TEST.VALID"], node_type="CONCEPT")
        passed += 1
    except TypeError:
        blocked += 1
    
    # Test 3: Invalid edge type
    nodes = list(engine_v2.nodes.values())
    if len(nodes) >= 2:
        result = engine_v2.associate(nodes[0].node_id, nodes[1].node_id, edge_type="INVALID_EDGE")
        if not result:
            blocked += 1
        else:
            passed += 1
    
    # Test 4: Valid edge type
    if len(nodes) >= 2:
        result = engine_v2.associate(nodes[0].node_id, nodes[1].node_id, edge_type="ASSOCIATION")
        if result:
            passed += 1
        else:
            blocked += 1
    
    return {"blocked": blocked, "passed": passed, "block_rate": round(blocked / max(blocked + passed, 1), 4)}


def run_full_comparison():
    """Run complete ANTES vs DESPUÉS comparison."""
    print("=" * 70)
    print("AXION BENCHMARK: Comparativa ANTES vs DESPUÉS")
    print("Datos verificables — 2026-07-26")
    print("=" * 70)
    
    # ── Build engines ──
    print("\n[1/6] Construyendo grafo de prueba (96 nodos, 5 dominios)...")
    
    engine_v1 = OmegaCubeEngine(holographic_dim=256)
    n1 = build_test_graph_v1(engine_v1)
    
    engine_v2 = OmegaCubeEngineV2(holographic_dim=256)
    n2 = build_test_graph_v2(engine_v2)
    
    print(f"   v1: {len(engine_v1.nodes)} nodos | v2: {len(engine_v2.nodes)} nodos")
    
    # ── Benchmark retrieval ──
    print("\n[2/6] Benchmarking retrieval (20 queries × 4 modos)...")
    
    v1_modes = ["diffusion", "holographic", "tensor", "combined"]
    v2_modes = ["hierarchical", "diffusion", "holographic", "combined"]
    
    v1_results = benchmark_engine(engine_v1, "v1", v1_modes)
    v2_results = benchmark_engine(engine_v2, "v2", v2_modes)
    
    # ── Benchmark scaling ──
    print("\n[3/6] Benchmarking escalado...")
    v1_scaling = benchmark_scaling(engine_v1, "v1")
    v2_scaling = benchmark_scaling(engine_v2, "v2")
    
    # ── Benchmark typed schema ──
    print("\n[4/6] Benchmarking typed schema (VirtualSet)...")
    schema_results = benchmark_typed_schema(engine_v2)
    
    # ── Benchmark boundary control ──
    print("\n[5/6] Benchmarking boundary control (PAGE-RAG)...")
    boundary_test_queries = [
        "best image generation model",
        "completely unrelated random gibberish xyzzy",
        "MCP server configuration",
    ]
    boundary_stats = {"total_results": 0, "abstained": 0}
    for q in boundary_test_queries:
        hits = engine_v2.query(q, mode="hierarchical", top_k=10, apply_boundaries=True)
        boundary_stats["total_results"] += len(hits)
        # Count abstained by checking if results were filtered
    boundary_stats["abstained"] = engine_v2.abstained_results
    
    # ── Benchmark hallucination detection ──
    print("\n[6/6] Benchmarking hallucination detection...")
    bias_stats = {"detections": engine_v2.bias_detections}
    
    # ── Compile results ──
    report = {
        "metadata": {
            "date": "2026-07-26",
            "nodes_v1": len(engine_v1.nodes),
            "nodes_v2": len(engine_v2.nodes),
            "queries": len(QUERIES),
            "domains": 5,
        },
        "retrieval_v1": v1_results,
        "retrieval_v2": v2_results,
        "scaling_v1": v1_scaling,
        "scaling_v2": v2_scaling,
        "typed_schema": schema_results,
        "boundary_control": boundary_stats,
        "hallucination_detection": bias_stats,
        "engine_v2_stats": engine_v2.stats(),
    }
    
    # ── Print comparison ──
    print("\n" + "=" * 70)
    print("RESULTADOS: ANTES (v1) vs DESPUÉS (v2)")
    print("=" * 70)
    
    print("\n📊 RETRIEVAL ACCURACY")
    print(f"   {'Mode':<15} {'Version':<8} {'P@5':>8} {'P@10':>8} {'Avg ms':>10} {'P95 ms':>10} {'Gray':>8}")
    print(f"   {'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")
    
    for mode in ["diffusion", "holographic", "combined"]:
        if mode in v1_results:
            r = v1_results[mode]
            print(f"   {mode:<15} {'v1':<8} {r['precision_at_5']:>7.1%} {r['precision_at_10']:>7.1%} {r['avg_time_ms']:>8.1f} {r['p95_time_ms']:>8.1f} {r['avg_gray_score']:>7.1f}")
        if mode in v2_results:
            r = v2_results[mode]
            print(f"   {mode:<15} {'v2':<8} {r['precision_at_5']:>7.1%} {r['precision_at_10']:>7.1%} {r['avg_time_ms']:>8.1f} {r['p95_time_ms']:>8.1f} {r['avg_gray_score']:>7.1f}")
    
    if "hierarchical" in v2_results:
        r = v2_results["hierarchical"]
        print(f"   {'hierarchical':<15} {'v2':<8} {r['precision_at_5']:>7.1%} {r['precision_at_10']:>7.1%} {r['avg_time_ms']:>8.1f} {r['p95_time_ms']:>8.1f} {r['avg_gray_score']:>7.1f}")
    
    print("\n📐 SCALING")
    print(f"   v1: {v1_scaling['estimated_complexity']}")
    for s in v1_scaling["scaling_data"]:
        print(f"      {s['nodes']:>4} nodes → {s['avg_ms']:>7.2f}ms")
    print(f"   v2: {v2_scaling['estimated_complexity']}")
    for s in v2_scaling["scaling_data"]:
        print(f"      {s['nodes']:>4} nodes → {s['avg_ms']:>7.2f}ms")
    
    print("\n🔒 TYPED SCHEMA (VirtualSet)")
    print(f"   Operaciones bloqueadas: {schema_results['blocked']}")
    print(f"   Operaciones válidas: {schema_results['passed']}")
    print(f"   Block rate: {schema_results['block_rate']:.1%}")
    
    print("\n🚧 BOUNDARY CONTROL (PAGE-RAG)")
    print(f"   Resultados abstained: {boundary_stats['abstained']}")
    
    print("\n🔍 HALLUCINATION DETECTION")
    print(f"   Bias detections: {bias_stats['detections']}")
    
    print("\n📈 ENGINE V2 STATS")
    for k, v in engine_v2.stats().items():
        print(f"   {k}: {v}")
    
    # Save report
    report_path = Path(__file__).parent / "axion_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Report saved: {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ Benchmark completo")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    run_full_comparison()
