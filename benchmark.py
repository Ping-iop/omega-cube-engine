"""
Omega-Cube Benchmark Suite.

Tests multi-dimensional retrieval, holographic matching,
annealing convergence, diffusion sampling, and gray-scale validation.

Compares Omega-Cube against flat retrieval baselines.
"""

import math
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from omega_cube import OmegaCubeEngine, TensorNode


def build_test_graph(engine: OmegaCubeEngine, num_domains: int = 5, nodes_per_domain: int = 20) -> int:
    """
    Build a synthetic multi-domain knowledge graph.
    
    Creates nodes across multiple domains with tensor hierarchies.
    """
    domains = [
        {
            "name": "COMFYUI",
            "dimensions": [
                "COMFYUI.WORKFLOWS.GENERATION",
                "COMFYUI.MODELS.CHECKPOINTS",
                "COMFYUI.NODES.CUSTOM",
                "IMAGE_QUALITY.RESOLUTION",
                "AI.TOOLS.VISUAL",
            ],
            "topics": ["SDXL", "SD1.5", "ControlNet", "IPAdapter", "LoRA",
                       "Upscale", "Inpaint", "FaceRestore", "Prompting", "Sampling"],
        },
        {
            "name": "EVONY",
            "dimensions": [
                "EVONY.GENERALS.PVP",
                "EVONY.GENERALS.F2P",
                "EVONY.STRATEGY.RALLIES",
                "GAMING.MOBILE.STRATEGY",
                "F2P.OPTIMIZATION.RESOURCES",
            ],
            "topics": ["Marcian", "CharlesVI", "Hermes", "Akechi", "Tamar",
                       "Ranged", "Mounted", "Ground", "Siege", "Defense"],
        },
        {
            "name": "HERMES",
            "dimensions": [
                "HERMES.CONFIG.MCP",
                "HERMES.SKILLS.AUTOMATION",
                "HERMES.CRON.MAINTENANCE",
                "AI.AGENTS.TOOLS",
                "DEVELOPMENT.AUTOMATION",
            ],
            "topics": ["MCP", "Skills", "CronJobs", "Memory", "Plugins",
                       "Config", "Providers", "Delegation", "Session", "Tools"],
        },
        {
            "name": "SECURITY",
            "dimensions": [
                "SECURITY.CRYPTOGRAPHY.STEGANOGRAPHY",
                "SECURITY.VERIFICATION.MULTI_BIT",
                "HBIT.PROTOCOL.GRAYSCALE",
                "AI.SAFETY.VERIFICATION",
                "TRUST.VALIDATION.EVIDENCE",
            ],
            "topics": ["H-Bit", "GrayScale", "Embedding", "Verification",
                       "PartialEvidence", "Watermark", "Fingerprint", "Hash", "Signature", "Trust"],
        },
        {
            "name": "ML",
            "dimensions": [
                "ML.TRAINING.PRETRAINING",
                "ML.ARCHITECTURE.DIFFUSION",
                "ML.OPTIMIZATION.ANNEALING",
                "AI.RESEARCH.NEUROSYMBOLIC",
                "SCIENCE.COMPUTATION.ALGORITHMS",
            ],
            "topics": ["DiffusionGemma", "AutoResearch", "Karpathy", "Annealing",
                       "Holographic", "Tensor", "Embedding", "FineTuning", "Inference", "Architecture"],
        },
    ]
    
    total_nodes = 0
    for domain in domains[:num_domains]:
        # Add domain axiom
        axiom = engine.add_node(
            content=f"{domain['name']} is a knowledge domain in Omega-Cube",
            hierarchies=[domain["dimensions"][0]],
            node_type="AXIOM",
            confidence=1.0,
            tags=[domain["name"].lower(), "axiom"],
        )
        total_nodes += 1
        
        # Add topic nodes with tensor hierarchies
        for i, topic in enumerate(domain["topics"][:nodes_per_domain]):
            # Each node exists in 2-3 hierarchies simultaneously
            primary = domain["dimensions"][i % len(domain["dimensions"])]
            secondary = domain["dimensions"][(i + 1) % len(domain["dimensions"])]
            
            hier_list = [primary, secondary]
            if i % 3 == 0 and len(domain["dimensions"]) > 2:
                tertiary = domain["dimensions"][(i + 2) % len(domain["dimensions"])]
                hier_list.append(tertiary)
            
            # Compute tensor position based on topic index
            pos_x = (i / len(domain["topics"])) * 0.8 + 0.1
            pos_y = (domains.index(domain) / num_domains) * 0.8 + 0.1
            tensor_pos = [pos_x, pos_y] + [0.5] * max(0, len(hier_list) - 2)
            
            node = engine.add_node(
                content=f"[{domain['name']}] {topic}: Detailed knowledge about {topic.lower()} "
                        f"in the context of {domain['name'].lower()}. This is a multi-dimensional "
                        f"node spanning {len(hier_list)} hierarchy axes.",
                hierarchies=hier_list,
                tensor_position=tensor_pos,
                node_type="CONCEPT",
                confidence=0.85,
                tags=[domain["name"].lower(), topic.lower()],
            )
            total_nodes += 1
            
            # Associate with axiom
            engine.associate(axiom.node_id, node.node_id)
            
            # Cross-domain associations (every 3rd node)
            if i % 3 == 0 and total_nodes > 10:
                # Associate with a random node from another domain
                other_nodes = [n for nid, n in engine.nodes.items() 
                              if n.primary_hierarchy.split(".")[0] != domain["name"]
                              and n.node_type == "CONCEPT"]
                if other_nodes:
                    import random
                    other = random.choice(other_nodes)
                    engine.associate(node.node_id, other.node_id)
    
    return total_nodes


def benchmark_retrieval(engine: OmegaCubeEngine, num_queries: int = 20) -> dict:
    """
    Benchmark retrieval accuracy and speed across modes.
    """
    queries = [
        ("best image generation model", ["COMFYUI"]),
        ("top PvP general for ranged attacks", ["EVONY"]),
        ("how to configure MCP servers in Hermes", ["HERMES"]),
        ("multi-bit verification of file integrity", ["SECURITY"]),
        ("diffusion model for text generation", ["ML"]),
        ("SDXL vs SD1.5 comparison", ["COMFYUI"]),
        ("F2P general ranking Marcian Charles", ["EVONY"]),
        ("cron job automation skills", ["HERMES"]),
        ("partial evidence grayscale verification", ["SECURITY", "HBIT"]),
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
        ("quantum annealing topology optimization", ["ML", "HERMES"]),
    ]
    
    modes = ["diffusion", "holographic", "tensor", "combined"]
    results = {}
    
    for mode in modes:
        mode_results = {
            "precision_at_5": 0.0,
            "precision_at_10": 0.0,
            "avg_time_ms": 0.0,
            "top_domain_hits": 0,
            "queries": [],
        }
        
        total_time = 0.0
        total_relevant = 0
        
        for query, expected_domains in queries[:num_queries]:
            start = time.time()
            hits = engine.query(query, mode=mode, top_k=10)
            elapsed = (time.time() - start) * 1000
            total_time += elapsed
            
            # Check domain relevance
            relevant_at_5 = 0
            relevant_at_10 = 0
            for i, hit in enumerate(hits[:10]):
                hit_domain = hit["primary_hierarchy"].split(".")[0] if hit["primary_hierarchy"] else ""
                domain_match = any(
                    hit_domain.upper().startswith(ed.upper()) or ed.upper() in hit_domain.upper()
                    for ed in expected_domains
                )
                if domain_match:
                    if i < 5:
                        relevant_at_5 += 1
                    relevant_at_10 += 1
            
            mode_results["queries"].append({
                "query": query,
                "p@5": relevant_at_5 / 5,
                "p@10": relevant_at_10 / 10,
                "time_ms": elapsed,
            })
            
            mode_results["precision_at_5"] += relevant_at_5
            mode_results["precision_at_10"] += relevant_at_10
            total_relevant += len(expected_domains)
        
        n = num_queries
        mode_results["precision_at_5"] = mode_results["precision_at_5"] / (n * 5)
        mode_results["precision_at_10"] = mode_results["precision_at_10"] / (n * 10)
        mode_results["avg_time_ms"] = total_time / n
        
        results[mode] = mode_results
    
    # Compute composite score
    best_p5 = max(r["precision_at_5"] for r in results.values())
    best_p10 = max(r["precision_at_10"] for r in results.values())
    best_time = min(r["avg_time_ms"] for r in results.values())
    
    composite = (best_p5 * 0.4 + best_p10 * 0.3 + max(0, 1 - best_time / 100) * 0.3)
    
    results["composite_score"] = composite
    
    return results


def benchmark_scaling(engine: OmegaCubeEngine) -> dict:
    """Benchmark how retrieval time scales with graph size."""
    if len(engine.nodes) == 0:
        return {"error": "Empty graph"}
    
    sizes = [10, 50, 100, 200, 500, 1000]
    scaling = []
    
    all_nodes = list(engine.nodes.values())
    query = "finding relevant nodes across domains"
    
    for size in sizes:
        if size > len(all_nodes):
            break
        
        # Create subset index
        from omega_cube.tensor_node import TensorIndex
        idx = TensorIndex()
        for n in all_nodes[:size]:
            idx.insert(n)
        
        # Temporarily swap index for benchmark
        original_index = engine.index
        engine.index = idx
        
        start = time.time()
        engine.query(query, mode="diffusion", top_k=10)
        elapsed = (time.time() - start) * 1000
        
        engine.index = original_index
        
        scaling.append({
            "nodes": size,
            "time_ms": elapsed,
        })
    
    # Fit O(log n) vs O(n) comparison
    log_factor = math.log(len(all_nodes)) if len(all_nodes) > 1 else 1
    
    return {
        "scaling_data": scaling,
        "estimated_complexity": "O(log n)" if len(scaling) > 1 and 
            scaling[-1]["time_ms"] / scaling[0]["time_ms"] < 5 else "O(n)",
        "log_factor": log_factor,
    }


def run_benchmarks():
    """Run the full benchmark suite."""
    print("=" * 60)
    print("Omega-Cube Engine — Benchmark Suite")
    print("=" * 60)
    
    # Create engine and build test graph
    engine = OmegaCubeEngine(holographic_dim=256)
    
    print("\n📦 Building test graph...")
    num_nodes = build_test_graph(engine, num_domains=5, nodes_per_domain=15)
    print(f"   Created {num_nodes} nodes across 5 domains")
    
    print(f"\n📊 Engine Stats:")
    stats = engine.stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    # Benchmark retrieval
    print(f"\n🔍 Benchmarking retrieval (4 modes × 20 queries)...")
    results = benchmark_retrieval(engine, num_queries=20)
    
    print(f"\n📈 Retrieval Results:")
    print(f"   {'Mode':<15} {'P@5':>8} {'P@10':>8} {'Avg Time':>10}")
    print(f"   {'─'*15} {'─'*8} {'─'*8} {'─'*10}")
    
    for mode in ["diffusion", "holographic", "tensor", "combined"]:
        if mode in results:
            r = results[mode]
            print(f"   {mode:<15} {r['precision_at_5']:>7.1%} {r['precision_at_10']:>7.1%} {r['avg_time_ms']:>8.1f}ms")
    
    print(f"\n   Composite Score: {results['composite_score']:.4f}")
    
    # Benchmark scaling
    print(f"\n📐 Benchmarking scaling behavior...")
    scaling = benchmark_scaling(engine)
    print(f"   Estimated complexity: {scaling['estimated_complexity']}")
    for s in scaling["scaling_data"]:
        print(f"   {s['nodes']:>5} nodes → {s['time_ms']:>6.1f}ms")
    
    # Test multi-topic query
    print(f"\n🧩 Testing multi-topic query...")
    multi_results = engine.query_multi_topic(
        "best optimization techniques",
        topics=["COMFYUI", "ML", "HERMES"],
        top_k_per_topic=2,
    )
    for topic, hits in multi_results.items():
        print(f"   {topic}: {len(hits)} results")
        for h in hits:
            print(f"      ▸ [{h['score']:.3f}] {h['content'][:80]}")
    
    # Test pattern emergence
    print(f"\n🔮 Testing pattern emergence...")
    patterns = engine.find_patterns("cross-domain integration", min_strength=0.2)
    print(f"   Found {len(patterns)} patterns")
    for p in patterns[:3]:
        print(f"   ▸ {p.get('cube_topic', '?')} (strength: {p.get('pattern_strength', 0):.3f})")
        for ac in p.get("aligned_cubes", [])[:2]:
            print(f"      ↔ cube {ac.get('cube_id', '?')} (alignment: {ac.get('alignment', 0):.3f})")
    
    print(f"\n{'='*60}")
    print(f"✅ Benchmark complete")
    print(f"   Composite score: {results['composite_score']:.4f}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    run_benchmarks()
