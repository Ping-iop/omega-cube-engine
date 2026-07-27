"""Multi-article router test: RAG BQ + Obsidian Vault."""
import time
from omega_cube.engine_v2 import OmegaCubeEngineV2
from omega_cube.marp.router import MARPRouter
from omega_cube.marp.protocol import ShardConfig, MARPMode


def build_engine(nodes):
    e = OmegaCubeEngineV2()
    for c, h, t in nodes:
        e.add_node(content=c, hierarchies=h, tags=t)
    return e


def test_router(engine, queries, label):
    router = MARPRouter(engine=engine)
    router._refresh_keywords_from_graph()
    domains = sorted(set(
        (n.primary_hierarchy or "").split(".")[0].lower()
        for n in engine.nodes.values() if n.primary_hierarchy
    ))
    shards = [
        ShardConfig(name=f"{d}_shard", domains=[d], mode=MARPMode.WRAPPER,
                    gpu_memory_mb=512, enabled=True)
        for d in domains
    ]

    correct = 0
    times = []
    fails = []
    for q, expected in queries:
        t0 = time.perf_counter()
        d = router.route(q, shards)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        pred = (d.ticket.active_domains[0]
                if d.ticket and d.ticket.active_domains else "unknown")
        hit = pred == expected
        if hit:
            correct += 1
        else:
            fails.append((q[:50], pred, expected))
        icon = "✅" if hit else "❌"
        print(f"  {icon} {pred:<14} <- {expected:<14} [{elapsed:.1f}ms] | {q[:55]}")

    acc = correct / len(queries) * 100
    avg = sum(times) / len(times)
    print(f"  -> Accuracy: {acc:.1f}% ({correct}/{len(queries)}) | Latencia: {avg:.1f}ms")
    if fails:
        print("  -> Fallos:")
        for q, p, e in fails:
            print(f'     "{q}" -> {p} (esperado: {e})')
    return acc, avg


# ── Article 1: RAG Binary Quantization (Avi Chawla) ──
rag_nodes = [
    ("Binary quantization converts float32 vectors to binary using np.where and np.packbits for 32x memory reduction",
     ["rag.quantization"], ["quantization", "binary", "float32", "packbits", "memory"]),
    ("Milvus vector database stores binary vectors with BIN_FLAT index and Hamming distance metric",
     ["vectordb.milvus"], ["milvus", "vectordb", "index", "hamming", "binary"]),
    ("BGE-large-en-v1.5 embedding model generates float32 text embeddings for semantic search",
     ["embeddings.bge"], ["embeddings", "bge", "float32", "semantic", "model"]),
    ("LlamaIndex orchestrates the RAG pipeline with SimpleDirectoryReader for document ingestion",
     ["rag.pipeline"], ["llamaindex", "pipeline", "ingestion", "reader", "orchestration"]),
    ("Top-k retrieval using Hamming distance finds most similar binary vectors in under 30ms",
     ["retrieval.search"], ["retrieval", "topk", "hamming", "search", "latency"]),
    ("Kimi-K2 instruct model on Groq generates responses with retrieved context in under 1 second",
     ["inference.groq"], ["inference", "groq", "kimi", "generation", "serving"]),
    ("Re-ranking with float vectors after binary retrieval improves top-1 accuracy by 8-12 percent",
     ["retrieval.reranking"], ["reranking", "float", "accuracy", "binary", "hybrid"]),
    ("Production RAG pulls context from Slack GitHub Jira databases simultaneously with auth and permissions",
     ["rag.production"], ["production", "multisource", "auth", "permissions", "sync"]),
]

rag_queries = [
    ("How does binary quantization reduce memory usage", "rag"),
    ("Store vectors in Milvus with Hamming distance", "vectordb"),
    ("Generate text embeddings with BGE model", "embeddings"),
    ("Orchestrate the RAG pipeline with LlamaIndex", "rag"),
    ("Retrieve top-k similar documents using Hamming search", "retrieval"),
    ("Serve Kimi-K2 model on Groq for fast inference", "inference"),
    ("Re-rank results with float vectors for better accuracy", "retrieval"),
    ("Pull context from multiple sources in production RAG", "rag"),
    ("Convert float32 embeddings to binary with packbits", "rag"),
    ("Index binary vectors with BIN_FLAT in vector database", "vectordb"),
]

# ── Article 2: Obsidian + Claude Vault (seeco) ──
obs_nodes = [
    ("Voice recorder with auto-transcription drops files into inbox folder for Claude to sort",
     ["ingestion.voice"], ["voice", "transcription", "inbox", "recorder", "audio"]),
    ("One inbox folder for raw thoughts, Claude builds out topics projects people from single point",
     ["vault.structure"], ["inbox", "structure", "folders", "topics", "projects"]),
    ("Morning scheduled job between 6:30 and 8:00 Claude files notes adds backlinks writes daily digest",
     ["automation.morning"], ["automation", "morning", "scheduled", "digest", "backlinks"]),
    ("Raw folder is untouchable, never edited, unfiltered voice timestamped as geological layer",
     ["vault.raw"], ["raw", "untouchable", "timestamped", "unfiltered", "archive"]),
    ("Every new note gets at least three backlinks to existing notes and one link to note older than two years",
     ["knowledge.backlinks"], ["backlinks", "connections", "graph", "links", "network"]),
    ("Sunday synthesis reads everything from past seven days compresses into weekly file with themes contradictions",
     ["knowledge.synthesis"], ["synthesis", "weekly", "themes", "contradictions", "compression"]),
    ("New chat starts with automatic pull from vault showing current work open hypotheses last decisions",
     ["context.loading"], ["context", "session", "automatic", "hypotheses", "decisions"]),
    ("Graph growing and link density climbing means vault is alive, files without connections are warehouse",
     ["knowledge.graph"], ["graph", "density", "alive", "connections", "metric"]),
]

obs_queries = [
    ("Record voice notes and transcribe them into the inbox", "ingestion"),
    ("Set up the vault folder structure with one inbox", "vault"),
    ("Schedule morning automation to file notes and add backlinks", "automation"),
    ("Keep raw notes untouchable as timestamped archive", "vault"),
    ("Add backlinks to connect new notes to the graph", "knowledge"),
    ("Generate weekly synthesis with themes and contradictions", "knowledge"),
    ("Load context automatically when starting a new session", "context"),
    ("Monitor graph density to check if the vault is alive", "knowledge"),
    ("Transcribe audio from phone walk into inbox folder", "ingestion"),
    ("Pull current projects and open hypotheses into new chat", "context"),
]

# ── Run ──
print("=" * 70)
print("COMPARACION MULTI-ARTICULO — Router v3 + IDF specificity")
print("=" * 70)

print("\n[ART 1] RAG Binary Quantization (Avi Chawla)")
print("  Dominios: rag, vectordb, embeddings, retrieval, inference")
print("-" * 70)
acc1, ms1 = test_router(build_engine(rag_nodes), rag_queries, "RAG")

print(f"\n[ART 2] Obsidian + Claude Vault (seeco)")
print("  Dominios: ingestion, vault, automation, knowledge, context")
print("-" * 70)
acc2, ms2 = test_router(build_engine(obs_nodes), obs_queries, "OBS")

print("\n" + "=" * 70)
print("RESUMEN COMPARATIVO")
print("=" * 70)
print(f"  Articulo 1 (RAG BQ):     {acc1:.1f}% | {ms1:.1f}ms")
print(f"  Articulo 2 (Obsidian):   {acc2:.1f}% | {ms2:.1f}ms")
print(f"  Promedio:                {(acc1 + acc2) / 2:.1f}% | {(ms1 + ms2) / 2:.1f}ms")
print("=" * 70)
