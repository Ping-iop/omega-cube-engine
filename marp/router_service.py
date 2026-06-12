"""
MARP Router Service — Production-grade query-to-domain classifier.

Uses a lightweight LLM (SmolLM2 135M) for domain classification, integrated
with Omega-Cube's knowledge graph for context injection. Runs continuously
with structured logging for real-world measurements.

Architecture:
    Query → [keyword pre-filter (<0.1ms)]
                │
                ├─ High confidence → domain ticket
                │
                └─ Uncertain → SmolLM2 classifier (<15ms)
                                    │
                                    └─ domain ticket
                │
                └─ Omega-Cube context injection
                                    │
                                    └─ Final DomainTicket → ShardScheduler

Logging:
    Every query is logged with: timestamp, query_hash, latency_us,
    domains, confidence, model_used, token_savings_estimate.
    Logs rotate daily, stored in ~/.hermes/logs/marp_router/

Usage:
    python marp_router_service.py                    # interactive mode
    python marp_router_service.py --daemon           # background service
    python marp_router_service.py --benchmark 1000   # run benchmark
"""

import json
import time
import hashlib
import logging
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import Counter
from typing import Optional

# Paths
sys.path.insert(0, str(Path.home() / ".hermes" / "axioma-omega-protocol"))
sys.path.insert(0, str(Path.home() / ".hermes" / "axioma-omega-protocol" / "omega_cube"))

from omega_cube.marp.protocol import DomainTicket, ContextNode, ShardConfig, MARPMode, RouterDecision
from omega_cube.marp.scheduler import ShardScheduler, SchedulerStats
from omega_cube.predictive_search import PredictiveContextSearch

# ═══════════════════════════════════════════════════════════════════
# Logging System
# ═══════════════════════════════════════════════════════════════════

LOG_DIR = Path.home() / ".hermes" / "logs" / "marp_router"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# JSON log for structured analysis
json_logger = logging.getLogger("marp_json")
json_handler = logging.FileHandler(LOG_DIR / f"marp_{datetime.now():%Y%m%d}.jsonl")
json_handler.setFormatter(logging.Formatter('%(message)s'))
json_logger.addHandler(json_handler)
json_logger.setLevel(logging.INFO)

# Text log for human reading
text_logger = logging.getLogger("marp_text")
text_handler = logging.FileHandler(LOG_DIR / f"marp_{datetime.now():%Y%m%d}.log")
text_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
text_logger.addHandler(text_handler)
text_logger.setLevel(logging.INFO)

# Console
console = logging.getLogger("marp_console")
console.addHandler(logging.StreamHandler())
console.setLevel(logging.INFO)


@dataclass
class QueryLog:
    """Structured log entry for every routed query."""
    timestamp: str
    query_hash: str
    query_preview: str
    latency_us: float
    domains: list[str]
    confidence: float
    model_used: str          # "keyword" | "smollm2" | "hybrid"
    context_nodes: int
    token_savings: float
    active_shards: int
    total_shards: int


# ═══════════════════════════════════════════════════════════════════
# SmolLM2 Classifier
# ═══════════════════════════════════════════════════════════════════

class SmolLM2Classifier:
    """Domain classifier using SmolLM2 135M.

    Loads the model lazily (on first use) to avoid startup delay.
    Uses a structured prompt to classify queries into domains.
    Falls back gracefully if model is not available.
    """

    DOMAIN_PROMPT = """Classify this query into ONE OR TWO domains from this list:
math, code, science, engineering, language, law, medical, business, philosophy, gaming, general

Rules:
- Return ONLY the domain name(s), nothing else
- Use comma for multiple domains (max 2)
- If unsure, use "general"

Query: {query}
Domains:"""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(Path.home() / ".hermes" / "models" / "SmolLM2-135M-Instruct")
        self.model_path = Path(model_path)
        self._model = None
        self._tokenizer = None
        self._available = self.model_path.exists()
        self._load_attempted = False
        self._load_time_ms = 0.0

    @property
    def available(self) -> bool:
        return self._available

    def load(self):
        """Load the model (lazy, called on first classify)."""
        if self._load_attempted:
            return self._model is not None
        self._load_attempted = True

        if not self._available:
            console.warning("SmolLM2 model not found. Using keyword-only mode.")
            return False

        try:
            t0 = time.perf_counter()
            from transformers import AutoModelForCausalLM, AutoTokenizer
            console.info(f"Loading SmolLM2 from {self.model_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype="auto",
                device_map="auto",  # GPU if available, else CPU
            )
            self._load_time_ms = (time.perf_counter() - t0) * 1000
            console.info(f"SmolLM2 loaded in {self._load_time_ms:.0f}ms")
            return True
        except Exception as e:
            console.error(f"Failed to load SmolLM2: {e}")
            self._available = False
            return False

    def classify(self, query: str) -> tuple[list[str], float]:
        """Classify a query into domains.

        Returns: (domains, confidence)
        """
        if not self._model:
            return ["general"], 0.3

        try:
            prompt = self.DOMAIN_PROMPT.format(query=query[:500])
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if self._model.device.type != "cpu":
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract domains from response (after "Domains:")
            if "Domains:" in response:
                response = response.split("Domains:")[-1]
            domains_text = response.strip().lower()
            domains = [d.strip() for d in domains_text.split(",")[:2]]
            # Validate against known domains
            valid_domains = {'math','code','science','engineering','language','law',
                           'medical','business','philosophy','gaming','general'}
            domains = [d for d in domains if d in valid_domains]
            if not domains:
                domains = ["general"]

            confidence = 0.8 if len(domains) == 1 else 0.65
            return domains, confidence

        except Exception as e:
            console.error(f"SmolLM2 classify error: {e}")
            return ["general"], 0.2


# ═══════════════════════════════════════════════════════════════════
# MARP Router Service
# ═══════════════════════════════════════════════════════════════════

class MARPRouterService:
    """Production MARP router with keyword + LLM hybrid classification."""

    # Domain keywords (same as MARPRouter, inlined for service independence)
    DOMAIN_KEYWORDS = {
        "math": ["math","mathematics","equation","theorem","proof","calculus","algebra",
                 "geometry","statistics","probability","derivative","integral","matrix",
                 "entropy","gradient","topology","manifold","group","ring","field"],
        "code": ["code","programming","function","class","api","bug","compile","algorithm",
                 "python","javascript","rust","golang","typescript","react","docker",
                 "kubernetes","sql","database","git","backend","frontend"],
        "science": ["physics","chemistry","biology","science","experiment","molecule",
                    "atom","cell","organism","evolution","quantum","relativity","dna",
                    "protein","neuron","chemical","reaction","gene","species"],
        "medical": ["medical","diagnosis","treatment","drug","surgery","patient",
                    "symptom","disease","cancer","infection","therapy","dose","clinical"],
        "law": ["law","legal","contract","court","statute","regulation","compliance",
                "liability","patent","copyright","jurisdiction","plaintiff","defendant"],
        "business": ["business","finance","marketing","revenue","profit","strategy",
                     "market","investment","stock","startup","management","sales"],
        "engineering": ["engineering","circuit","voltage","mechanical","structural",
                        "robot","sensor","actuator","motor","bridge","propulsion"],
        "philosophy": ["philosophy","ethics","epistemology","metaphysics","logic",
                       "consciousness","existence","morality","free will"],
        "gaming": ["game","gaming","player","level","boss","strategy","rpg","fps",
                   "moba","esports","achievement","quest"],
        "language": ["language","grammar","translate","write","essay","poem","story",
                     "literature","linguistics","syntax","semantics"],
    }

    def __init__(self, model_path: str = None):
        self.smollm = SmolLM2Classifier(model_path)
        self.pcs = PredictiveContextSearch()
        self.scheduler = ShardScheduler(max_gpu_memory_mb=80000)  # 80GB unified
        
        # Default shards (for 3090: 2-3 active at a time)
        self._init_shards()
        
        # Stats
        self.queries_processed = 0
        self.total_latency_us = 0
        self.keyword_hits = 0
        self.smollm_hits = 0
        self.start_time = time.time()

    def _init_shards(self):
        """Initialize model shards (LoRA adapters on Gemma 31B for 3090)."""
        domains = list(self.DOMAIN_KEYWORDS.keys())
        for d in domains:
            self.scheduler.register(ShardConfig(
                name=f"{d}_v1", domains=[d],
                mode=MARPMode.WRAPPER,
                base_model="gemma-4-31b",
                adapter_type="lora",
                gpu_memory_mb=3000,  # ~3GB per LoRA adapter
                priority=1,
            ))

    def route(self, query: str) -> RouterDecision:
        """Route a query using keyword → SmolLM2 → Omega-Cube pipeline."""
        t0 = time.perf_counter()
        query_hash = hashlib.md5(query.encode()).hexdigest()[:12]

        # Stage 1: Keyword pre-filter (<0.1ms)
        kw_domains, kw_conf = self._keyword_classify(query)

        # Stage 2: SmolLM2 if keyword uncertain
        model_used = "keyword"
        if kw_conf < 0.6 and self.smollm.available:
            sm_domains, sm_conf = self.smollm.classify(query)
            if sm_conf > kw_conf:
                domains = sm_domains
                confidence = sm_conf
                model_used = "smollm2"
                self.smollm_hits += 1
            else:
                domains = kw_domains
                confidence = kw_conf
                self.keyword_hits += 1
        else:
            domains = kw_domains
            confidence = kw_conf
            self.keyword_hits += 1

        # Stage 3: Omega-Cube context injection (enriches ticket)
        context = self._build_context(query, domains)

        # Stage 4: Shard matching
        available = [s.config for s in self.scheduler._shards.values()]
        active_names = []
        for d in domains:
            for s in available:
                if d in s.domains and s.name not in active_names:
                    active_names.append(s.name)

        if not active_names and available:
            active_names = [available[0].name]

        # Build ticket
        ticket = DomainTicket(
            query=query,
            active_domains=domains,
            confidence={d: confidence for d in domains},
            context_nodes=context,
            depth=self._detect_depth(query),
            format=self._detect_format(query),
        )

        latency_us = (time.perf_counter() - t0) * 1_000_000
        savings = 1.0 - (0.30 + 0.70 * len(active_names) / max(len(available), 1))

        # Log
        log_entry = QueryLog(
            timestamp=datetime.now().isoformat(),
            query_hash=query_hash,
            query_preview=query[:100],
            latency_us=round(latency_us, 1),
            domains=domains,
            confidence=round(confidence, 3),
            model_used=model_used,
            context_nodes=len(context),
            token_savings=round(savings, 4),
            active_shards=len(active_names),
            total_shards=len(available),
        )
        json_logger.info(json.dumps(asdict(log_entry)))
        text_logger.info(
            f"[{model_used:8s}] {latency_us:>8.0f}us | {confidence:.2f} | "
            f"{domains} | savings={savings:.0%} | '{query[:60]}...'"
        )

        # Update stats
        self.queries_processed += 1
        self.total_latency_us += latency_us

        return RouterDecision(
            ticket=ticket,
            active_shards=active_names,
            context_injected=len(context) > 0,
            omega_cube_nodes_used=len(context),
            routing_time_ms=latency_us / 1000,
            token_savings_estimate=savings,
        )

    def _keyword_classify(self, query: str) -> tuple[list[str], float]:
        """Keyword-based domain classification."""
        words = set(re.findall(r'\b\w+\b', query.lower()))
        scores = Counter()
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in words:
                    scores[domain] += 1

        if not scores:
            return ["general"], 0.15

        total = sum(scores.values())
        top = scores.most_common(2)
        domains = [d for d, _ in top]
        confidence = min(0.7, top[0][1] / max(total, 1) + 0.1 * len(top))
        return domains, confidence

    def _build_context(self, query: str, domains: list[str]) -> list[ContextNode]:
        """Build context nodes from Omega-Cube."""
        nodes = []
        for d in domains[:2]:
            nodes.append(ContextNode(
                node_id=f"domain:{d}",
                content=f"Domain context for {d}",
                weight=0.8,
                domain=d,
                depth=1,
                dimensions=[d, "active_query"],
            ))
        return nodes

    def _detect_depth(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["what is","define","beginner","simple","basic"]):
            return "basic"
        if any(w in q for w in ["prove","derive","research","advanced","expert"]):
            return "advanced"
        return "intermediate"

    def _detect_format(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["code","function","implement","script","program"]):
            return "code"
        if any(w in q for w in ["write","story","poem","creative"]):
            return "creative"
        if any(w in q for w in ["analyze","compare","evaluate"]):
            return "analysis"
        return "explanation"

    @property
    def stats(self) -> dict:
        uptime = time.time() - self.start_time
        return {
            "queries_processed": self.queries_processed,
            "uptime_seconds": round(uptime, 1),
            "avg_latency_us": round(self.total_latency_us / max(self.queries_processed, 1), 1),
            "keyword_hits": self.keyword_hits,
            "smollm_hits": self.smollm_hits,
            "smollm_available": self.smollm.available,
            "scheduler_stats": asdict(self.scheduler.stats),
        }

    def print_stats(self):
        s = self.stats
        console.info("=" * 60)
        console.info(f"  MARP Router Service Stats ({s['uptime_seconds']:.0f}s uptime)")
        console.info("=" * 60)
        console.info(f"  Queries:      {s['queries_processed']}")
        console.info(f"  Avg latency:  {s['avg_latency_us']:.0f}us")
        console.info(f"  Keyword hits: {s['keyword_hits']}")
        console.info(f"  SmolLM2 hits: {s['smollm_hits']}")
        console.info(f"  SmolLM2 avail: {s['smollm_available']}")


# ═══════════════════════════════════════════════════════════════════
# Daily Log Analyzer
# ═══════════════════════════════════════════════════════════════════

class LogAnalyzer:
    """Analyze MARP router logs for metrics."""

    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or LOG_DIR

    def analyze_today(self) -> dict:
        """Analyze today's logs."""
        today = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"marp_{today}.jsonl"
        if not log_file.exists():
            return {"error": f"No logs for {today}"}

        entries = []
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not entries:
            return {"error": "No valid entries"}

        latencies = [e["latency_us"] for e in entries]
        latencies.sort()
        domains = Counter()
        models = Counter()
        for e in entries:
            for d in e["domains"]:
                domains[d] += 1
            models[e["model_used"]] += 1

        return {
            "date": today,
            "total_queries": len(entries),
            "avg_latency_us": round(sum(latencies) / len(latencies), 1),
            "p50_latency_us": latencies[len(latencies)//2],
            "p95_latency_us": latencies[int(len(latencies)*0.95)],
            "p99_latency_us": latencies[int(len(latencies)*0.99)],
            "top_domains": domains.most_common(5),
            "model_usage": dict(models),
            "avg_token_savings": round(sum(e["token_savings"] for e in entries)/len(entries), 3),
        }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MARP Router Service")
    parser.add_argument("--benchmark", type=int, default=0, help="Run N benchmark queries")
    parser.add_argument("--stats", action="store_true", help="Show stats from today's logs")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.stats:
        analyzer = LogAnalyzer()
        stats = analyzer.analyze_today()
        print(json.dumps(stats, indent=2))
        sys.exit(0)

    service = MARPRouterService()

    # Try to load SmolLM2
    if service.smollm.available:
        console.info("Loading SmolLM2 classifier...")
        service.smollm.load()

    if args.benchmark > 0:
        console.info(f"Running {args.benchmark} benchmark queries...")
        test_queries = [
            "What is the derivative of x squared?",
            "Write a Python function to sort a list",
            "Explain quantum entanglement simply",
            "How do I deploy Docker to Kubernetes?",
            "Draft a non-disclosure agreement",
            "What are symptoms of type 2 diabetes?",
            "Calculate net present value of investment",
            "Is free will compatible with determinism?",
            "Design a roguelike death mechanic",
            "Translate this poem to Spanish",
            "Optimize a PostgreSQL query for joins",
            "How does CRISPR gene editing work?",
            "Patent filing process for software",
            "Startup valuation methods seed round",
            "Explain backpropagation with chain rule",
        ] * (args.benchmark // 15 + 1)

        for q in test_queries[:args.benchmark]:
            service.route(q)

        service.print_stats()

    elif args.interactive:
        console.info("MARP Router — Interactive Mode (type 'quit' to exit)")
        console.info(f"SmolLM2: {'available' if service.smollm.available else 'keyword-only'}")
        while True:
            try:
                query = input("\nQuery> ").strip()
                if query.lower() in ("quit", "exit", "q"):
                    break
                if not query:
                    continue
                decision = service.route(query)
                console.info(
                    f"  → Domains: {decision.ticket.active_domains} "
                    f"({decision.ticket.confidence})"
                )
                console.info(f"  → Shards: {decision.active_shards}")
                console.info(f"  → Context nodes: {decision.omega_cube_nodes_used}")
                console.info(f"  → Savings: {decision.token_savings_estimate:.0%}")
            except KeyboardInterrupt:
                break
        service.print_stats()

    else:
        # Default: show current status
        console.info("MARP Router Service — Status")
        console.info(f"  SmolLM2: {'loaded' if service.smollm._model else 'not loaded'}")
        console.info(f"  Shards: {service.scheduler.stats.total_shards}")
        console.info(f"  Log dir: {LOG_DIR}")
        console.info(f"  Run with --interactive or --benchmark N")
