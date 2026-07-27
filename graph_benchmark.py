"""
Graph Engineering Benchmark v2 — Prueba con grafos nuevos + medición antes/después.

Basado en: "Graph Engineering Clearly Explained" (Akshay Pachaar, 2026-07-25).
Mide: accuracy de routing, latencia, dominios descubiertos, health score.

Ejecutar: python -m omega_cube.graph_benchmark
"""

from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass, field
from datetime import datetime

from omega_cube.engine_v2 import OmegaCubeEngineV2
from omega_cube.marp.router import MARPRouter, STANDARD_DOMAINS
from omega_cube.marp.scheduler import AdaptiveScheduler
from omega_cube.marp.protocol import ShardConfig, MARPMode
from omega_cube.telemetry import Telemetry


# ═══════════════════════════════════════════════════════════════════
# Graph patterns — jerarquías con dominio como raíz (formato: "dominio.sub")
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GraphNode:
    name: str
    domain: str
    content: str
    tags: list[str] = field(default_factory=list)
    hierarchy: str = ""  # formato: "dominio.subdominio"


@dataclass
class GraphPattern:
    name: str
    description: str
    nodes: list[GraphNode]
    queries: list[tuple[str, str]]  # (query, expected_domain)


def build_graph_patterns() -> list[GraphPattern]:
    """5 patrones de grafo del artículo con dominios nuevos (no STANDARD)."""

    return [
        # ── 1. Sequential pipeline ──
        GraphPattern(
            name="sequential_pipeline",
            description="'A researcher gathers material, a writer drafts, a reviewer judges'",
            nodes=[
                GraphNode("research", "research", "Research agent gathers material from multiple sources, searches academic databases, reads papers and extracts key findings", ["research", "search", "gather", "sources", "papers"], "research.gathering"),
                GraphNode("write", "writing", "Writer agent drafts content based on research material, creates structured text with proper citations", ["write", "draft", "content", "text", "citations"], "writing.drafting"),
                GraphNode("review", "review", "Reviewer agent judges quality, checks accuracy, validates claims against evidence, flags unsupported statements", ["review", "judge", "quality", "validate", "accuracy"], "review.validation"),
                GraphNode("format", "formatting", "Formatter agent structures output, applies templates, generates final document in multiple formats", ["format", "template", "structure", "document", "pdf"], "formatting.output"),
            ],
            queries=[
                ("Search for papers on graph neural networks", "research"),
                ("Draft a summary of the research findings", "writing"),
                ("Check if the claims are supported by evidence", "review"),
                ("Format the final report as PDF", "formatting"),
                ("Gather data from multiple databases", "research"),
                ("Write the introduction section", "writing"),
                ("Validate the methodology description", "review"),
            ],
        ),
        # ── 2. Conditional review loop ──
        GraphPattern(
            name="conditional_review_loop",
            description="'If the review passes, the run ends. If it fails, an edge sends the draft back'",
            nodes=[
                GraphNode("generate", "generation", "Generator agent creates initial draft from prompt, produces first version of content", ["generate", "create", "initial", "draft", "prompt"], "generation.initial"),
                GraphNode("evaluate", "evaluation", "Evaluator agent scores the draft on clarity, accuracy, completeness using rubric", ["evaluate", "score", "clarity", "rubric", "completeness"], "evaluation.scoring"),
                GraphNode("revise", "revision", "Reviser agent rewrites sections that failed evaluation, improves weak areas", ["revise", "rewrite", "improve", "sections", "weak"], "revision.improvement"),
                GraphNode("approve", "approval", "Approver agent gives final sign-off when all criteria pass threshold", ["approve", "signoff", "final", "threshold", "pass"], "approval.final"),
            ],
            queries=[
                ("Generate a first draft of the article", "generation"),
                ("Score the draft on clarity and accuracy", "evaluation"),
                ("Rewrite the weak sections", "revision"),
                ("Give final approval on the document", "approval"),
                ("Create initial content from the prompt", "generation"),
                ("Evaluate completeness against the rubric", "evaluation"),
                ("Improve the introduction paragraph", "revision"),
            ],
        ),
        # ── 3. Parallel fan-out ──
        GraphPattern(
            name="parallel_fanout",
            description="'Research fans out into independent searches' (Anthropic: 90.2% improvement)",
            nodes=[
                GraphNode("web_search", "websearch", "Web search agent queries multiple search engines, retrieves top results with snippets", ["websearch", "search", "engines", "results", "snippets"], "websearch.multi"),
                GraphNode("code_search", "codesearch", "Code search agent scans repositories, finds relevant functions and implementations", ["codesearch", "repositories", "functions", "implementations", "github"], "codesearch.repos"),
                GraphNode("doc_search", "docsearch", "Documentation search agent reads API docs, finds relevant endpoints and examples", ["docsearch", "api", "docs", "endpoints", "examples"], "docsearch.api"),
                GraphNode("merge", "merging", "Merge agent combines results from all searches, deduplicates, ranks by relevance", ["merge", "combine", "deduplicate", "rank", "relevance"], "merging.results"),
            ],
            queries=[
                ("Search the web for recent LLM papers", "websearch"),
                ("Find the implementation in the repository", "codesearch"),
                ("Look up the API documentation for auth", "docsearch"),
                ("Combine and rank all search results", "merging"),
                ("Query multiple search engines simultaneously", "websearch"),
                ("Scan GitHub for similar functions", "codesearch"),
                ("Find the endpoint documentation", "docsearch"),
            ],
        ),
        # ── 4. State corruption detection ──
        GraphPattern(
            name="state_corruption",
            description="'A sloppy write in node two becomes a confident input for node five'",
            nodes=[
                GraphNode("ingest", "ingestion", "Ingestion agent receives raw data, validates schema, normalizes format before storage", ["ingest", "validate", "schema", "normalize", "raw"], "ingestion.validation"),
                GraphNode("transform", "transformation", "Transformation agent converts data between formats, applies business rules, enriches records", ["transform", "convert", "enrich", "rules", "records"], "transformation.rules"),
                GraphNode("detect", "corruption", "Corruption detector agent checks for state drift, validates invariants, flags anomalies", ["corruption", "drift", "invariants", "anomalies", "detect"], "corruption.detection"),
                GraphNode("recover", "recovery", "Recovery agent restores clean state from checkpoint, rolls back corrupted transactions", ["recovery", "restore", "checkpoint", "rollback", "clean"], "recovery.checkpoint"),
            ],
            queries=[
                ("Validate the incoming data schema", "ingestion"),
                ("Transform records using business rules", "transformation"),
                ("Detect state drift between nodes", "corruption"),
                ("Restore clean state from checkpoint", "recovery"),
                ("Normalize the raw input format", "ingestion"),
                ("Check for data anomalies in the pipeline", "corruption"),
                ("Roll back the corrupted transaction", "recovery"),
            ],
        ),
        # ── 5. Multi-model specialists ──
        GraphPattern(
            name="multi_model_specialists",
            description="'Different models per step' + 'reviewer on a different model with fresh context'",
            nodes=[
                GraphNode("plan", "planning", "Planning agent decomposes task into subtasks, estimates cost, selects model per step", ["planning", "decompose", "subtasks", "estimate", "model"], "planning.decomposition"),
                GraphNode("execute", "execution", "Execution agent runs subtasks with assigned model, collects outputs, tracks progress", ["execution", "subtasks", "assigned", "outputs", "progress"], "execution.subtasks"),
                GraphNode("audit", "auditing", "Audit agent on different model reviews execution with fresh context, checks for hallucination", ["audit", "fresh", "hallucination", "review", "different"], "auditing.fresh"),
                GraphNode("report", "reporting", "Reporting agent synthesizes final output, includes provenance and confidence scores", ["report", "synthesize", "provenance", "confidence", "final"], "reporting.synthesis"),
            ],
            queries=[
                ("Decompose the task into subtasks", "planning"),
                ("Execute subtask three with GPT-4", "execution"),
                ("Audit the output for hallucination with fresh context", "auditing"),
                ("Synthesize the final report with provenance", "reporting"),
                ("Estimate the cost of each step", "planning"),
                ("Track progress on all subtasks", "execution"),
                ("Review with a different model for bias", "auditing"),
            ],
        ),
    ]


# ═══════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════

def run_benchmark():
    patterns = build_graph_patterns()
    total_nodes = sum(len(p.nodes) for p in patterns)
    total_queries = sum(len(p.queries) for p in patterns)

    print("=" * 70)
    print("GRAPH ENGINEERING BENCHMARK v2")
    print("Basado en: 'Graph Engineering Clearly Explained' (Akshay Pachaar)")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print(f"\n📐 {len(patterns)} patrones | {total_nodes} nodos | {total_queries} queries")

    # ── Build engine ──
    print("\n[1/5] Construyendo Axion-Cube con nodos de todos los grafos...")
    engine = OmegaCubeEngineV2()

    for pattern in patterns:
        for node in pattern.nodes:
            engine.add_node(
                content=node.content,
                hierarchies=[node.hierarchy],
                tags=node.tags,
            )

    n_nodes = len(engine.nodes)
    n_hierarchies = len(set(
        h for n in engine.nodes.values()
        for h in (n.hierarchies or [])
    ))
    domains_in_graph = sorted(set(
        (n.primary_hierarchy or "").replace("/", ".").split(".")[0].lower()
        for n in engine.nodes.values()
        if n.primary_hierarchy
    ))
    print(f"   Nodos: {n_nodes} | Jerarquías: {n_hierarchies}")
    print(f"   Dominios en grafo: {len(domains_in_graph)} → {', '.join(domains_in_graph)}")

    # ── Setup MARP ──
    print("\n[2/5] Configurando MARP router + scheduler...")
    router = MARPRouter(engine=engine)

    # BEFORE: measure with only STANDARD_DOMAINS (no graph refresh)
    router_before = MARPRouter(engine=engine)
    # Don't call _refresh_keywords_from_graph — simulates old behavior

    # AFTER: with graph-aware keyword evolution
    router._refresh_keywords_from_graph()

    # Count discovered domains
    discovered = set()
    for kw, doms in router._kw_to_domain.items():
        for d in doms:
            if d not in STANDARD_DOMAINS:
                discovered.add(d)
    print(f"   Dominios estándar: {len(STANDARD_DOMAINS)}")
    print(f"   Dominios descubiertos del grafo: {len(discovered)} → {', '.join(sorted(discovered))}")

    # Shards
    all_domains = sorted(set(list(STANDARD_DOMAINS.keys()) + list(discovered)))
    shards = [
        ShardConfig(
            name=f"{d}_shard",
            domains=[d],
            mode=MARPMode.WRAPPER,
            gpu_memory_mb=512,
            enabled=True,
        )
        for d in all_domains
    ]

    scheduler = AdaptiveScheduler(max_gpu_memory_mb=8192)
    for s in shards:
        scheduler.register(s)

    # Telemetry
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    telemetry = Telemetry(
        log_dir=log_dir,
        session_id=f"graph_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # ── Run BEFORE (no graph learning) ──
    print("\n[3/5] Ejecutando queries SIN aprendizaje de grafo (baseline)...")
    before_results = _run_queries(router_before, patterns, shards, scheduler, telemetry, label="BEFORE")

    # ── Run AFTER (with graph learning) ──
    print("\n[4/5] Ejecutando queries CON aprendizaje de grafo (fix v3)...")
    after_results = _run_queries(router, patterns, shards, scheduler, telemetry, label="AFTER")

    # ── Report ──
    print("\n[5/5] Reporte comparativo...")
    _print_comparison(before_results, after_results, patterns, discovered)

    # Telemetry report
    eff_report = telemetry.efficiency_report()
    problems = telemetry.detect_problems()
    telemetry.flush()

    print(f"\n🏥 HEALTH SCORE: {eff_report.get('health_score', 0)}/100 (Grade: {eff_report.get('grade', '?')})")
    print(f"   Breakdown: {eff_report.get('breakdown', {})}")

    if problems:
        print(f"\n⚠️  PROBLEMAS DETECTADOS ({len(problems)}):")
        for p in problems:
            icon = "🔴" if p.get("severity") == "critical" else "🟡"
            print(f"   {icon} [{p.get('severity','?').upper()}] {p.get('type','?')}: {p.get('message','')}")
            if p.get("suggestion"):
                print(f"      → {p['suggestion']}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": "graph_engineering_v2",
        "source_article": "Graph Engineering Clearly Explained (Akshay Pachaar)",
        "patterns": len(patterns),
        "total_nodes": total_nodes,
        "total_queries": total_queries,
        "domains_standard": len(STANDARD_DOMAINS),
        "domains_discovered": sorted(discovered),
        "before": before_results,
        "after": after_results,
        "telemetry": eff_report,
        "problems": problems,
    }

    report_path = os.path.join(os.path.dirname(__file__), "graph_benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Report: {report_path}")
    print(f"📁 Logs: {log_dir}")
    print("\n" + "=" * 70)
    print("✅ Graph Engineering Benchmark v2 completo")
    print("=" * 70)


def _run_queries(router, patterns, shards, scheduler, telemetry, label=""):
    """Run all pattern queries and return per-pattern results."""
    results = {}

    for pattern in patterns:
        correct = 0
        times = []

        for query, expected_domain in pattern.queries:
            t0 = time.perf_counter()
            decision = router.route(query, shards)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

            predicted = decision.ticket.active_domains[0] if decision.ticket and decision.ticket.active_domains else "unknown"
            hit = predicted == expected_domain
            if hit:
                correct += 1

            # Telemetry
            telemetry.log_routing(
                query=query,
                expected_domain=expected_domain,
                predicted_domain=predicted,
                confidence=0.0,
                latency_ms=elapsed,
                context_nodes=1 if decision.context_injected else 0,
                bias_detected=bool(decision.bias_detections),
            )

            scheduler.activate_for_decision(decision)

        accuracy = correct / len(pattern.queries) * 100
        avg_ms = sum(times) / len(times) if times else 0

        results[pattern.name] = {
            "accuracy": round(accuracy, 1),
            "correct": correct,
            "total": len(pattern.queries),
            "avg_ms": round(avg_ms, 3),
        }

        icon = "🟢" if accuracy >= 70 else ("🟡" if accuracy >= 40 else "🔴")
        print(f"   {icon} {pattern.name}: {accuracy:.1f}% ({correct}/{len(pattern.queries)}) | {avg_ms:.1f}ms")

    # Global
    total_correct = sum(r["correct"] for r in results.values())
    total_q = sum(r["total"] for r in results.values())
    global_acc = total_correct / total_q * 100 if total_q else 0
    all_times = []
    for r in results.values():
        all_times.append(r["avg_ms"])
    global_avg = sum(all_times) / len(all_times) if all_times else 0

    results["_global"] = {
        "accuracy": round(global_acc, 1),
        "correct": total_correct,
        "total": total_q,
        "avg_ms": round(global_avg, 3),
    }

    return results


def _print_comparison(before, after, patterns, discovered):
    """Print before/after comparison table."""
    print("\n" + "=" * 70)
    print("RESULTADOS COMPARATIVOS: Sin fix vs Con fix v3")
    print("=" * 70)

    print(f"\n{'Patrón':<30} {'ANTES':>10} {'DESPUÉS':>10} {'Δ':>8}")
    print("-" * 62)

    for pattern in patterns:
        b = before.get(pattern.name, {})
        a = after.get(pattern.name, {})
        b_acc = b.get("accuracy", 0)
        a_acc = a.get("accuracy", 0)
        delta = a_acc - b_acc
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"   {pattern.name:<27} {b_acc:>8.1f}% {a_acc:>8.1f}% {delta_str:>7}%")

    b_g = before.get("_global", {})
    a_g = after.get("_global", {})
    delta_g = a_g.get("accuracy", 0) - b_g.get("accuracy", 0)
    delta_g_str = f"+{delta_g:.1f}" if delta_g >= 0 else f"{delta_g:.1f}"
    print("-" * 62)
    print(f"   {'GLOBAL':<27} {b_g.get('accuracy',0):>8.1f}% {a_g.get('accuracy',0):>8.1f}% {delta_g_str:>7}%")
    print(f"   {'Latencia avg':<27} {b_g.get('avg_ms',0):>7.1f}ms {a_g.get('avg_ms',0):>7.1f}ms")

    print(f"\n📊 Dominios descubiertos del grafo: {len(discovered)}")
    print(f"   {', '.join(sorted(discovered))}")


if __name__ == "__main__":
    run_benchmark()
