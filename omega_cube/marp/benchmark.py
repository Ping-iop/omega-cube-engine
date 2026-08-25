"""
MARP Benchmark: Comparativa v1 vs v2 con datos verificables.

Mide:
1. Routing accuracy (domain classification correctness)
2. Routing latency (ms per query)
3. Context quality (grounded vs generic nodes)
4. Bias detection (hallucination counteraction)
5. Adaptive prefetch accuracy (scheduler learning)
6. Keyword evolution (graph-driven keyword growth)

Ejecutar: python -m omega_cube.marp.benchmark
"""

from __future__ import annotations

import json
import time
import os
from collections import Counter
from dataclasses import asdict

from omega_cube.marp.router import MARPRouter, STANDARD_DOMAINS
from omega_cube.marp.scheduler import ShardScheduler, AdaptiveScheduler
from omega_cube.marp.protocol import ShardConfig, MARPMode
from omega_cube.engine_v2 import OmegaCubeEngineV2
from omega_cube.telemetry import Telemetry


# ═══════════════════════════════════════════════════════════════════
# Test data: 40 queries with known correct domains
# ═══════════════════════════════════════════════════════════════════

TEST_QUERIES = [
    # Math (8)
    ("What is the derivative of sin(x)?", "math"),
    ("Prove that sqrt(2) is irrational", "math"),
    ("Explain the central limit theorem", "math"),
    ("How does gradient descent work?", "math"),
    ("What is a topological space?", "math"),
    ("Calculate the integral of e^x from 0 to 1", "math"),
    ("Explain eigenvalues and eigenvectors", "math"),
    ("What is Bayes theorem?", "math"),
    # Code (8)
    ("Write a Python function to sort a list", "code"),
    ("How does Docker containerization work?", "code"),
    ("Explain the difference between SQL and NoSQL", "code"),
    ("Debug this React component that won't render", "code"),
    ("What is the time complexity of quicksort?", "code"),
    ("How to set up a Kubernetes cluster?", "code"),
    ("Explain REST API design best practices", "code"),
    ("Write a Rust function with ownership", "code"),
    # Science (6)
    ("How does CRISPR gene editing work?", "science"),
    ("Explain quantum entanglement simply", "science"),
    ("What is the mechanism of protein folding?", "science"),
    ("How do neurons transmit signals?", "science"),
    ("Explain the theory of general relativity", "science"),
    ("What causes antibiotic resistance?", "science"),
    # Engineering (4)
    ("Design a PID controller for a drone", "engineering"),
    ("How does a lithium-ion battery work?", "engineering"),
    ("Explain structural load analysis", "engineering"),
    ("What is the difference between AC and DC motors?", "engineering"),
    # Language (4)
    ("Write a poem about the ocean", "language"),
    ("Explain the difference between metaphor and simile", "language"),
    ("Translate this paragraph to French", "language"),
    ("What is the Sapir-Whorf hypothesis?", "language"),
    # Business (4)
    ("How to calculate ROI for a marketing campaign?", "business"),
    ("Explain the business model canvas", "business"),
    ("What is a convertible note in startups?", "business"),
    ("How does compound interest work for investments?", "business"),
    # Philosophy (3)
    ("What is the trolley problem?", "philosophy"),
    ("Explain Kant's categorical imperative", "philosophy"),
    ("Is consciousness reducible to physics?", "philosophy"),
    # Gaming (3)
    ("What is the best build for a mage in Elden Ring?", "gaming"),
    ("Explain the meta in League of Legends", "gaming"),
    ("How do game engines handle collision detection?", "gaming"),
]

# Ambiguous queries for bias detection test
AMBIGUOUS_QUERIES = [
    "Explain the mathematics behind neural networks",  # math + code + science
    "How does game theory apply to business strategy?",  # gaming + business + math
    "Write code to simulate quantum physics",  # code + science
    "The ethics of genetic engineering in humans",  # philosophy + science
    "Analyze the economics of esports tournaments",  # business + gaming
]


def build_test_engine() -> OmegaCubeEngineV2:
    """Build a test Axion-Cube engine with domain nodes."""
    engine = OmegaCubeEngineV2()

    domain_nodes = {
        "math": [
            ("Calculus fundamentals", "MATH.CALCULUS", ["derivative", "integral", "limit"]),
            ("Linear algebra", "MATH.ALGEBRA", ["matrix", "eigenvalue", "vector"]),
            ("Probability theory", "MATH.STATISTICS", ["bayes", "distribution", "variance"]),
            ("Topology basics", "MATH.TOPOLOGY", ["space", "continuity", "compact"]),
            ("Optimization methods", "MATH.OPTIMIZATION", ["gradient", "descent", "convex"]),
        ],
        "code": [
            ("Python programming", "CODE.PYTHON", ["function", "class", "decorator"]),
            ("Web development", "CODE.WEB", ["react", "api", "html"]),
            ("Database systems", "CODE.DATABASE", ["sql", "nosql", "index"]),
            ("DevOps practices", "CODE.DEVOPS", ["docker", "kubernetes", "ci"]),
            ("Algorithms", "CODE.ALGORITHMS", ["sort", "search", "complexity"]),
        ],
        "science": [
            ("Molecular biology", "SCIENCE.BIOLOGY", ["crispr", "dna", "protein"]),
            ("Quantum physics", "SCIENCE.PHYSICS", ["entanglement", "superposition", "wave"]),
            ("Neuroscience", "SCIENCE.NEUROSCIENCE", ["neuron", "synapse", "signal"]),
            ("Evolution", "SCIENCE.BIOLOGY", ["selection", "mutation", "adaptation"]),
        ],
        "engineering": [
            ("Control systems", "ENGINEERING.ELECTRICAL", ["pid", "controller", "feedback"]),
            ("Energy storage", "ENGINEERING.MATERIALS", ["battery", "lithium", "capacity"]),
            ("Structural analysis", "ENGINEERING.CIVIL", ["load", "stress", "beam"]),
            ("Motors", "ENGINEERING.MECHANICAL", ["ac", "dc", "torque"]),
        ],
        "language": [
            ("Creative writing", "LANGUAGE.CREATIVE", ["poem", "story", "metaphor"]),
            ("Linguistics", "LANGUAGE.LINGUISTICS", ["syntax", "grammar", "sapir"]),
            ("Translation", "LANGUAGE.TRANSLATION", ["french", "spanish", "localization"]),
        ],
        "business": [
            ("Finance", "BUSINESS.FINANCE", ["roi", "interest", "investment"]),
            ("Strategy", "BUSINESS.STRATEGY", ["canvas", "model", "market"]),
            ("Startups", "BUSINESS.FINANCE", ["convertible", "note", "funding"]),
        ],
        "philosophy": [
            ("Ethics", "PHILOSOPHY.ETHICS", ["trolley", "kant", "moral"]),
            ("Metaphysics", "PHILOSOPHY.METAPHYSICS", ["consciousness", "existence", "mind"]),
        ],
        "gaming": [
            ("Game design", "GAMING.GAME_DESIGN", ["collision", "engine", "physics"]),
            ("Strategy games", "GAMING.STRATEGY", ["build", "meta", "esports"]),
        ],
    }

    for domain, nodes in domain_nodes.items():
        for content, hierarchy, tags in nodes:
            engine.add_node(
                content=content,
                hierarchies=[hierarchy],
                tags=tags,
                node_type="CONCEPT",
            )

    # Add axioms
    engine.add_node(
        content="Mathematical proofs require deductive reasoning",
        hierarchies=["MATH.LOGIC"],
        tags=["proof", "deduction"],
        node_type="AXIOM",
    )
    engine.add_node(
        content="Code correctness requires testing",
        hierarchies=["CODE.TESTING"],
        tags=["test", "verification"],
        node_type="AXIOM",
    )

    engine._ensure_hierarchy()
    return engine


def create_shards() -> list[ShardConfig]:
    """Create test shard configs for all domains."""
    shards = []
    for domain in STANDARD_DOMAINS:
        shards.append(ShardConfig(
            name=f"{domain}_shard",
            domains=[domain],
            mode=MARPMode.WRAPPER,
            gpu_memory_mb=2048,
            enabled=True,
        ))
    shards.append(ShardConfig(
        name="general_shard",
        domains=["general"],
        mode=MARPMode.WRAPPER,
        gpu_memory_mb=1024,
        enabled=True,
    ))
    return shards


def run_benchmark():
    print("=" * 70)
    print("MARP BENCHMARK: v1 vs v2 — Datos Verificables")
    print(f"Fecha: {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # ── Build test engine ──
    print("\n[1/6] Construyendo grafo de prueba...")
    engine = build_test_engine()
    print(f"   Nodos: {len(engine.nodes)} | Jerarquías: {len(engine.hierarchy_tree)}")

    shards = create_shards()
    print(f"   Shards: {len(shards)}")

    # ── v1 Router (sin engine) ──
    print("\n[2/6] Benchmarking MARP v1 (sin engine)...")
    router_v1 = MARPRouter()  # No engine = v1 behavior

    v1_results = []
    v1_times = []
    v1_correct = 0

    for query, expected_domain in TEST_QUERIES:
        t0 = time.perf_counter()
        decision = router_v1.route(query, shards)
        elapsed = (time.perf_counter() - t0) * 1000
        v1_times.append(elapsed)

        predicted = decision.ticket.active_domains[0] if decision.ticket.active_domains else "none"
        correct = predicted == expected_domain
        if correct:
            v1_correct += 1

        v1_results.append({
            "query": query,
            "expected": expected_domain,
            "predicted": predicted,
            "correct": correct,
            "confidence": decision.ticket.confidence.get(predicted, 0),
            "context_nodes": len(decision.ticket.context_nodes),
            "routing_ms": round(elapsed, 3),
            "hierarchical": decision.hierarchical_routing_used,
            "boundary_filtered": decision.boundary_filtered,
        })

    v1_accuracy = v1_correct / len(TEST_QUERIES) * 100
    v1_avg_ms = sum(v1_times) / len(v1_times)
    v1_p95_ms = sorted(v1_times)[int(len(v1_times) * 0.95)]
    v1_context_avg = sum(r["context_nodes"] for r in v1_results) / len(v1_results)

    print(f"   Accuracy: {v1_accuracy:.1f}% ({v1_correct}/{len(TEST_QUERIES)})")
    print(f"   Latencia avg: {v1_avg_ms:.3f}ms | P95: {v1_p95_ms:.3f}ms")
    print(f"   Context nodes avg: {v1_context_avg:.1f}")

    # ── Telemetry ──
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "logs")
    telemetry = Telemetry(log_dir=log_dir, session_id=f"marp_bench_{time.strftime('%Y%m%d_%H%M%S')}")

    # ── v2 Router (con engine) ──
    print("\n[3/6] Benchmarking MARP v2 (con Axion-Cube engine)...")
    router_v2 = MARPRouter(engine=engine)

    v2_results = []
    v2_times = []
    v2_correct = 0

    for query, expected_domain in TEST_QUERIES:
        t0 = time.perf_counter()
        decision = router_v2.route(query, shards)
        elapsed = (time.perf_counter() - t0) * 1000
        v2_times.append(elapsed)

        predicted = decision.ticket.active_domains[0] if decision.ticket.active_domains else "none"
        correct = predicted == expected_domain
        if correct:
            v2_correct += 1

        v2_results.append({
            "query": query,
            "expected": expected_domain,
            "predicted": predicted,
            "correct": correct,
            "confidence": decision.ticket.confidence.get(predicted, 0),
            "context_nodes": len(decision.ticket.context_nodes),
            "routing_ms": round(elapsed, 3),
            "hierarchical": decision.hierarchical_routing_used,
            "boundary_filtered": decision.boundary_filtered,
            "bias_detected": decision.ticket.bias_detected,
            "bias_type": decision.ticket.bias_type,
            "grounding_scores": [
                round(cn.grounding_score, 3) for cn in decision.ticket.context_nodes
            ],
            "has_holographic": any(
                len(cn.holographic_signature) > 0
                for cn in decision.ticket.context_nodes
            ),
        })

        # Telemetry: log every routing decision
        telemetry.log_routing(
            query=query,
            predicted_domain=predicted,
            latency_ms=elapsed,
            confidence=decision.ticket.confidence.get(predicted, 0),
            expected_domain=expected_domain,
            correct=correct,
            hierarchical=decision.hierarchical_routing_used,
            context_nodes=len(decision.ticket.context_nodes),
            boundary_filtered=decision.boundary_filtered,
            bias_detected=decision.ticket.bias_detected,
        )

    v2_accuracy = v2_correct / len(TEST_QUERIES) * 100
    v2_avg_ms = sum(v2_times) / len(v2_times)
    v2_p95_ms = sorted(v2_times)[int(len(v2_times) * 0.95)]
    v2_context_avg = sum(r["context_nodes"] for r in v2_results) / len(v2_results)
    v2_hierarchical_pct = sum(1 for r in v2_results if r["hierarchical"]) / len(v2_results) * 100
    v2_holographic_pct = sum(1 for r in v2_results if r.get("has_holographic")) / len(v2_results) * 100
    v2_boundary_total = sum(r["boundary_filtered"] for r in v2_results)

    print(f"   Accuracy: {v2_accuracy:.1f}% ({v2_correct}/{len(TEST_QUERIES)})")
    print(f"   Latencia avg: {v2_avg_ms:.3f}ms | P95: {v2_p95_ms:.3f}ms")
    print(f"   Context nodes avg: {v2_context_avg:.1f}")
    print(f"   Hierarchical routing: {v2_hierarchical_pct:.0f}%")
    print(f"   Holographic context: {v2_holographic_pct:.0f}%")
    print(f"   Boundary filtered: {v2_boundary_total} nodes")

    # ── Bias detection test ──
    print("\n[4/6] Benchmarking bias detection (queries ambiguas)...")
    bias_results = []
    for query in AMBIGUOUS_QUERIES:
        decision = router_v2.route(query, shards)
        bias_results.append({
            "query": query,
            "domains": decision.ticket.active_domains[:3],
            "bias_detected": decision.ticket.bias_detected,
            "bias_type": decision.ticket.bias_type,
            "counteracted": decision.ticket.bias_counteracted,
        })
        status = "🔍 BIAS" if decision.ticket.bias_detected else "✓ OK"
        print(f"   {status} | {query[:50]}... → {decision.ticket.active_domains[:2]}")

    bias_detections = sum(1 for r in bias_results if r["bias_detected"])
    print(f"   Detecciones: {bias_detections}/{len(AMBIGUOUS_QUERIES)}")

    # ── Adaptive scheduler test ──
    print("\n[5/6] Benchmarking AdaptiveScheduler...")
    adaptive = AdaptiveScheduler(max_gpu_memory_mb=8192)
    static = ShardScheduler(max_gpu_memory_mb=8192)

    for s in shards:
        adaptive.register(s)
        static.register(s)

    # Simulate a session: user asks mostly math + code questions
    session_queries = [
        ("What is a derivative?", "math"),
        ("Write a Python sort function", "code"),
        ("Explain eigenvalues", "math"),
        ("How does Docker work?", "code"),
        ("Calculate an integral", "math"),
        ("Debug this React component", "code"),
        ("What is Bayes theorem?", "math"),
        ("Explain REST API design", "code"),
        ("Prove sqrt(2) irrational", "math"),
        ("Set up Kubernetes cluster", "code"),
        # Now shift to science
        ("How does CRISPR work?", "science"),
        ("Explain quantum entanglement", "science"),
    ]

    adaptive_activations = 0
    static_activations = 0

    for query, domain in session_queries:
        decision = router_v2.route(query, shards)

        t0 = time.perf_counter()
        adaptive.activate_for_decision(decision)
        adaptive_activations += 1

        static.activate_for_decision(decision)
        static_activations += 1

    adaptive_stats = adaptive.stats
    static_stats = static.stats

    print(f"   Adaptive prefetch hits: {adaptive_stats.prefetch_hits}")
    print(f"   Adaptive prefetch misses: {adaptive_stats.prefetch_misses}")
    print(f"   Prediction accuracy: {adaptive.prediction_accuracy:.1%}")
    print(f"   Static prefetch hits: {static_stats.prefetch_hits}")
    print(f"   Total activations: adaptive={adaptive_activations}, static={static_activations}")

    # ── Keyword evolution test ──
    print("\n[6/6] Benchmarking keyword evolution...")
    router_evo = MARPRouter(engine=engine)
    router_evo._keyword_refresh_interval = 5  # refresh every 5 queries for test

    initial_kw_count = sum(len(info["keywords"]) for info in STANDARD_DOMAINS.values())

    for i in range(10):
        query, _ = TEST_QUERIES[i]
        router_evo.route(query, shards)

    final_kw_count = sum(len(info["keywords"]) for info in STANDARD_DOMAINS.values())
    kw_growth = final_kw_count - initial_kw_count

    print(f"   Keywords iniciales: {initial_kw_count}")
    print(f"   Keywords finales: {final_kw_count}")
    print(f"   Crecimiento: +{kw_growth} keywords desde el grafo")

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("RESULTADOS: MARP v1 vs v2")
    print("=" * 70)

    print(f"""
📊 ROUTING ACCURACY
   v1: {v1_accuracy:.1f}% ({v1_correct}/{len(TEST_QUERIES)})
   v2: {v2_accuracy:.1f}% ({v2_correct}/{len(TEST_QUERIES)})
   Delta: {v2_accuracy - v1_accuracy:+.1f}%

⏱  ROUTING LATENCY
   v1: avg {v1_avg_ms:.3f}ms | P95 {v1_p95_ms:.3f}ms
   v2: avg {v2_avg_ms:.3f}ms | P95 {v2_p95_ms:.3f}ms
   Delta: {v2_avg_ms - v1_avg_ms:+.3f}ms avg

🧠 CONTEXT QUALITY
   v1: {v1_context_avg:.1f} nodes/query (generic text)
   v2: {v2_context_avg:.1f} nodes/query (grounded + holographic)
   Hierarchical routing: {v2_hierarchical_pct:.0f}%
   Holographic signatures: {v2_holographic_pct:.0f}%
   Boundary filtered: {v2_boundary_total} ungrounded nodes removed

🔍 BIAS DETECTION
   Ambiguous queries tested: {len(AMBIGUOUS_QUERIES)}
   Bias detected: {bias_detections}
   Counteracted: {sum(1 for r in bias_results if r['counteracted'])}

⚡ ADAPTIVE SCHEDULER
   Prefetch hits: {adaptive_stats.prefetch_hits} (adaptive) vs {static_stats.prefetch_hits} (static)
   Prediction accuracy: {adaptive.prediction_accuracy:.1%}

📈 KEYWORD EVOLUTION
   Growth: +{kw_growth} keywords from graph (CORTEX-inspired)
""")

    # ── Save report ──
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_queries": len(TEST_QUERIES),
        "ambiguous_queries": len(AMBIGUOUS_QUERIES),
        "graph_nodes": len(engine.nodes),
        "shards": len(shards),
        "v1": {
            "accuracy_pct": round(v1_accuracy, 1),
            "correct": v1_correct,
            "avg_latency_ms": round(v1_avg_ms, 3),
            "p95_latency_ms": round(v1_p95_ms, 3),
            "avg_context_nodes": round(v1_context_avg, 1),
        },
        "v2": {
            "accuracy_pct": round(v2_accuracy, 1),
            "correct": v2_correct,
            "avg_latency_ms": round(v2_avg_ms, 3),
            "p95_latency_ms": round(v2_p95_ms, 3),
            "avg_context_nodes": round(v2_context_avg, 1),
            "hierarchical_routing_pct": round(v2_hierarchical_pct, 1),
            "holographic_context_pct": round(v2_holographic_pct, 1),
            "boundary_filtered_total": v2_boundary_total,
        },
        "bias_detection": {
            "tested": len(AMBIGUOUS_QUERIES),
            "detected": bias_detections,
            "counteracted": sum(1 for r in bias_results if r["counteracted"]),
            "results": bias_results,
        },
        "adaptive_scheduler": {
            "prefetch_hits": adaptive_stats.prefetch_hits,
            "prefetch_misses": adaptive_stats.prefetch_misses,
            "prediction_accuracy": round(adaptive.prediction_accuracy, 3),
            "static_prefetch_hits": static_stats.prefetch_hits,
        },
        "keyword_evolution": {
            "initial": initial_kw_count,
            "final": final_kw_count,
            "growth": kw_growth,
        },
        "v1_details": v1_results,
        "v2_details": v2_results,
    }

    report_path = os.path.join(
        os.path.dirname(__file__), "marp_benchmark_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"💾 Report saved: {report_path}")

    # ── Telemetry: efficiency report + problem detection ──
    print("\n" + "=" * 70)
    print("📊 TELEMETRÍA: Reporte de Eficiencia")
    print("=" * 70)

    eff_report = telemetry.efficiency_report()
    problems = telemetry.detect_problems()
    telemetry.flush()

    health = eff_report.get("health_score", {})
    routing_eff = eff_report.get("routing_efficiency", {})
    latency = eff_report.get("latency", {})

    print(f"""
🏥 Health Score: {health.get('total', 0)}/100 (Grade: {health.get('grade', '?')})
   Breakdown: {health.get('breakdown', {})}

📈 Routing Efficiency:
   Accuracy: {routing_eff.get('accuracy_pct', 0)}%
   Hierarchical: {routing_eff.get('hierarchical_pct', 0)}%
   Avg context nodes: {routing_eff.get('avg_context_nodes', 0)}
   Bias detections: {routing_eff.get('bias_detections', 0)}

⏱  Latency:
   Avg: {latency.get('avg_ms', 0)}ms | P95: {latency.get('p95_ms', 0)}ms | Max: {latency.get('max_ms', 0)}ms

📁 Logs: {telemetry.log_dir}
""")

    if problems:
        print(f"⚠️  PROBLEMAS DETECTADOS ({len(problems)}):")
        for p in problems:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(p["severity"], "⚪")
            print(f"   {icon} [{p['severity'].upper()}] {p['type']}: {p['message']}")
            print(f"      → {p['suggestion']}")
    else:
        print("✅ Sin problemas detectados")

    print("\n" + "=" * 70)
    print("✅ MARP Benchmark completo")
    print("=" * 70)

    return report


if __name__ == "__main__":
    run_benchmark()
