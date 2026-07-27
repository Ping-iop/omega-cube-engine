"""
MARP Router — Axion-Cube native query-to-domain classifier.

Component #10 of Axion-Cube Engine. Uses the existing hierarchical
knowledge graph (HierarchicalSummarizer, TensorNode, HolographicEncoder,
BoundaryController, HallucinationDetector) to classify queries into
domains and build structured DomainTickets.

This is the "clerk" in the restaurant metaphor. It NEVER generates text.
It only classifies, enriches context, and routes to model shards.

v2 improvements (2026-07-26, from arXiv 2026 papers):
1. HierarchicalSummarizer routing: O(log n) coarse-to-fine (H²MT)
2. BoundaryController grounding: filters ungrounded context (PAGE-RAG)
3. HallucinationDetector: detects domain classification bias (2607.00447)
4. Holographic context nodes: 256D embeddings in ContextNode
5. Evolving keywords: extracted from graph, not hardcoded (CORTEX)

Latency target: <5ms (Axion-Cube native) vs 0.25ms (keyword fallback)
Accuracy target: 90%+ (vs 40% keyword-only)
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from omega_cube.marp.protocol import (
    DomainTicket,
    ContextNode,
    RouterDecision,
    ShardConfig,
    MARPMode,
)
from omega_cube.predictive_search import PredictiveContextSearch
from omega_cube.tensor_node import TensorNode

# Stopwords for keyword evolution (content extraction)
_STOPWORDS = frozenset({
    "about", "above", "after", "again", "against", "all", "also", "because",
    "been", "before", "being", "below", "between", "both", "but", "can",
    "cannot", "could", "does", "doing", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "here",
    "how", "into", "its", "itself", "just", "more", "most", "other",
    "out", "over", "own", "same", "should", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "then", "there", "these",
    "they", "this", "those", "through", "too", "under", "until", "very",
    "was", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "your", "yours",
})


# ═══════════════════════════════════════════════════════════════════
# Domain taxonomy (knowledge domains for model sharding)
# ═══════════════════════════════════════════════════════════════════

STANDARD_DOMAINS = {
    "math": {
        "subdomains": ["algebra", "calculus", "statistics", "geometry",
                       "number_theory", "topology", "information_theory", "optimization"],
        "keywords": ["math", "mathematics", "equation", "theorem", "proof",
                     "calculus", "algebra", "geometry", "statistics", "probability",
                     "entropy", "gradient", "derivative", "integral", "matrix"],
    },
    "code": {
        "subdomains": ["python", "javascript", "rust", "cpp", "algorithms",
                       "systems", "web", "database", "devops"],
        "keywords": ["code", "programming", "function", "class", "api", "bug",
                     "compile", "algorithm", "python", "javascript", "rust",
                     "react", "docker", "kubernetes", "sql", "database"],
    },
    "science": {
        "subdomains": ["physics", "chemistry", "biology", "astronomy", "neuroscience"],
        "keywords": ["physics", "chemistry", "biology", "science", "experiment",
                     "molecule", "atom", "cell", "organism", "evolution",
                     "quantum", "relativity", "dna", "protein", "neuron"],
    },
    "engineering": {
        "subdomains": ["electrical", "mechanical", "civil", "robotics", "materials"],
        "keywords": ["engineering", "circuit", "voltage", "mechanical", "structural",
                     "robot", "sensor", "actuator", "motor", "propulsion"],
    },
    "language": {
        "subdomains": ["linguistics", "translation", "grammar", "literature", "creative"],
        "keywords": ["language", "grammar", "translate", "write", "essay",
                     "poem", "story", "literature", "linguistics", "syntax"],
    },
    "law": {
        "subdomains": ["contract", "ip", "criminal", "international", "tax"],
        "keywords": ["law", "legal", "contract", "court", "statute",
                     "compliance", "liability", "patent", "copyright"],
    },
    "medical": {
        "subdomains": ["diagnosis", "treatment", "pharmacology", "surgery", "epidemiology"],
        "keywords": ["medical", "diagnosis", "treatment", "drug", "surgery",
                     "patient", "symptom", "disease", "cancer", "therapy"],
    },
    "business": {
        "subdomains": ["finance", "marketing", "management", "economics", "strategy"],
        "keywords": ["business", "finance", "marketing", "revenue", "profit",
                     "strategy", "market", "investment", "stock", "startup"],
    },
    "philosophy": {
        "subdomains": ["ethics", "epistemology", "metaphysics", "logic", "political"],
        "keywords": ["philosophy", "ethics", "epistemology", "metaphysics",
                     "logic", "consciousness", "existence", "morality"],
    },
    "gaming": {
        "subdomains": ["game_design", "game_dev", "strategy", "esports", "mechanics"],
        "keywords": ["game", "gaming", "player", "level", "strategy",
                     "rpg", "fps", "moba", "esports", "achievement"],
    },
}


@dataclass
class DomainScore:
    domain: str
    omega_score: float      # from Axion-Cube graph search
    keyword_score: float    # from keyword matching
    tensor_score: float     # from TensorNode N-dim scoring
    combined: float
    context_nodes: list[ContextNode] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)


class MARPRouter:
    """Axion-Cube native router: query → domain classification + context.

    v2: Now uses HierarchicalSummarizer for O(log n) routing,
    BoundaryController for grounded context, and HallucinationDetector
    for bias detection in domain classification.

    This replaces the standalone MARP router. All routing flows through
    Axion-Cube's knowledge graph — no external dependencies.
    """

    def __init__(
        self,
        predictive_search: Optional[PredictiveContextSearch] = None,
        engine=None,  # OmegaCubeEngineV2 instance
    ):
        self._predictive = predictive_search or PredictiveContextSearch()
        self._engine = engine
        self._summarizer = None
        self._boundary = None
        self._hallucination_detector = None
        self._init_domain_index()
        self._init_patterns()
        self._keyword_refresh_count = 0
        self._keyword_refresh_interval = 50  # refresh every N queries

    def _ensure_v2_components(self):
        """Lazy-init v2 components from engine."""
        if self._engine is None:
            return
        if self._summarizer is None and hasattr(self._engine, 'summarizer'):
            self._summarizer = self._engine.summarizer
        if self._boundary is None and hasattr(self._engine, 'boundary'):
            self._boundary = self._engine.boundary
        if self._hallucination_detector is None and hasattr(self._engine, 'hallucination_detector'):
            self._hallucination_detector = self._engine.hallucination_detector

    def _init_domain_index(self):
        self._kw_to_domain: dict[str, list[str]] = {}
        for domain, info in STANDARD_DOMAINS.items():
            for kw in info["keywords"]:
                kw = kw.lower()
                self._kw_to_domain.setdefault(kw, []).append(domain)
            for sub in info.get("subdomains", []):
                self._kw_to_domain.setdefault(sub, []).append(domain)

    def _init_patterns(self):
        self._depth_map = {
            "basic": [r"\bwhat is\b", r"\bdefine\b", r"\bbeginner\b", r"\bsimple\b"],
            "intermediate": [r"\bhow (?:does|to|can)\b", r"\bexplain\b", r"\bcompare\b"],
            "advanced": [r"\bprove\b", r"\bderive\b", r"\bimplement\b", r"\bdesign\b"],
            "expert": [r"\bresearch\b", r"\bnovel\b", r"\bstate.of.the.art\b", r"\bpaper\b"],
        }
        self._format_map = {
            "code": [r"\bcode\b", r"\bfunction\b", r"\bimplement\b", r"\bscript\b"],
            "explanation": [r"\bexplain\b", r"\bwhat\b", r"\bwhy\b", r"\bhow\b"],
            "analysis": [r"\banalyze\b", r"\bcompare\b", r"\bevaluate\b"],
            "creative": [r"\bwrite\b", r"\bcreative\b", r"\bstory\b", r"\bpoem\b"],
        }

    # ══════════════════════════════════════════════════════════════
    # Main routing entry point
    # ══════════════════════════════════════════════════════════════

    def route(
        self,
        query: str,
        available_shards: list[ShardConfig],
        conversation_history: list[str] | None = None,
    ) -> RouterDecision:
        t0 = time.perf_counter()
        self._ensure_v2_components()

        # 1. Axion-Cube hierarchical search for domain nodes (v2: O(log n))
        omega_scores = self._omega_search(query)

        # 2. Keyword scoring as fallback/boost
        kw_scores = self._keyword_score(query)

        # 3. Combine Axion-Cube + keyword scores
        domain_scores = self._combine_scores(omega_scores, kw_scores, query)

        # 4. v2: Detect and counteract domain classification bias
        bias_detected = False
        bias_type = "none"
        bias_counteracted = False
        if self._hallucination_detector and domain_scores:
            bias_dicts = [
                {"domain": ds.domain, "score": ds.combined}
                for ds in domain_scores
            ]
            bias = self._hallucination_detector.detect_bias(query, bias_dicts)
            if bias["bias_type"] != "none":
                bias_detected = True
                bias_type = bias["bias_type"]
                counteracted = self._hallucination_detector.counteract(
                    query, bias_dicts, bias
                )
                if counteracted:
                    bias_counteracted = True
                    # Re-sort domain scores after counteraction
                    score_map = {d["domain"]: d["score"] for d in counteracted}
                    for ds in domain_scores:
                        if ds.domain in score_map:
                            ds.combined = score_map[ds.domain]
                    domain_scores.sort(key=lambda x: x.combined, reverse=True)

        # 5. Detect depth, format, audience
        depth = self._detect(query, self._depth_map, "intermediate")
        fmt = self._detect(query, self._format_map, "explanation")
        audience = self._detect_audience(query)

        # 6. Match domains to shards
        active_names, reasons = self._match_shards(domain_scores, available_shards)

        # 7. v2: Build grounded context from Axion-Cube nodes
        context, boundary_filtered = self._build_context(query, domain_scores)

        # 8. Build ticket
        ticket = DomainTicket(
            query=query,
            active_domains=[s.domain for s in domain_scores[:3]],
            confidence={s.domain: s.combined for s in domain_scores[:5]},
            context_nodes=context,
            depth=depth,
            format=fmt,
            audience=audience,
            conversation_history=conversation_history or [],
            bias_detected=bias_detected,
            bias_type=bias_type,
            bias_counteracted=bias_counteracted,
        )

        savings = self._estimate_savings(len(active_names), len(available_shards))
        routing_time = (time.perf_counter() - t0) * 1000

        # v2: Periodic keyword evolution from graph
        self._keyword_refresh_count += 1
        if self._keyword_refresh_count >= self._keyword_refresh_interval:
            self._refresh_keywords_from_graph()
            self._keyword_refresh_count = 0

        return RouterDecision(
            ticket=ticket,
            active_shards=active_names,
            activation_reasons=reasons,
            context_injected=len(context) > 0,
            omega_cube_nodes_used=len(context),
            routing_time_ms=routing_time,
            token_savings_estimate=savings,
            hierarchical_routing_used=self._summarizer is not None,
            boundary_filtered=boundary_filtered,
            bias_detections=1 if bias_detected else 0,
        )

    # ══════════════════════════════════════════════════════════════
    # Axion-Cube graph search (v2: hierarchical)
    # ══════════════════════════════════════════════════════════════

    def _omega_search(self, query: str) -> dict[str, float]:
        """Search Axion-Cube knowledge graph for domain-relevant nodes.

        v2: Uses HierarchicalSummarizer.route_coarse_to_fine() for O(log n)
        routing when engine is available. Falls back to PredictiveContextSearch.
        """
        scores: dict[str, float] = {}

        # v2: Hierarchical routing via Axion-Cube summarizer
        if self._summarizer and self._engine:
            try:
                nodes = self._engine.nodes
                tree = self._engine.hierarchy_tree
                if nodes and tree:
                    results = self._summarizer.route_coarse_to_fine(
                        query, nodes, tree, top_k=10
                    )
                    for node, sim in results:
                        # Extract domain from primary_hierarchy
                        domain = ""
                        if hasattr(node, 'primary_hierarchy') and node.primary_hierarchy:
                            domain = node.primary_hierarchy.split(".")[0].lower()
                        elif hasattr(node, 'hierarchies') and node.hierarchies:
                            domain = node.hierarchies[0].split(".")[0].lower()
                        if domain:
                            scores[domain] = max(scores.get(domain, 0.0), sim)
                    if scores:
                        return scores
            except Exception:
                pass

        # Fallback: PredictiveContextSearch (v1 behavior)
        query_lower = query.lower()
        words = re.findall(r'\b\w{3,}\b', query_lower)

        for word in words:
            try:
                results = self._predictive.predict(word, top_k=3)
                for node_id, node_weight in results:
                    for domain in STANDARD_DOMAINS:
                        if domain in node_id.lower():
                            scores[domain] = max(
                                scores.get(domain, 0.0),
                                node_weight
                            )
            except Exception:
                pass

        # Fallback: semantic overlap with domain keywords (standard + learned)
        if not scores:
            # Check all indexed keywords, not just STANDARD_DOMAINS
            for word in words:
                for domain in self._kw_to_domain.get(word, []):
                    scores[domain] = max(scores.get(domain, 0.0), 0.5)
            # Also check STANDARD_DOMAINS for completeness
            if not scores:
                for domain, info in STANDARD_DOMAINS.items():
                    domain_terms = set(info["keywords"])
                    overlap = len(set(words) & domain_terms)
                    if overlap > 0:
                        scores[domain] = min(0.9, 0.4 + 0.1 * overlap)

        return scores

    def _keyword_score(self, query: str) -> dict[str, tuple[float, list[str]]]:
        """Keyword-based domain scoring with IDF-like specificity weighting.

        Keywords exclusive to one domain score 1.0; keywords shared across N
        domains score 1/N each. This prevents generic terms ('binary', 'model')
        from drowning out domain-specific signals ('milvus', 'packbits').
        """
        scores: dict[str, tuple[float, list[str]]] = {}
        words = set(re.findall(r'\b\w+\b', query.lower()))
        for word in words:
            if word in self._kw_to_domain:
                domains_for_word = self._kw_to_domain[word]
                specificity = 1.0 / len(domains_for_word)  # IDF-like
                for domain in domains_for_word:
                    prev_score, prev_matches = scores.get(domain, (0.0, []))
                    scores[domain] = (prev_score + specificity, prev_matches + [word])
        return scores

    def _combine_scores(
        self,
        omega: dict[str, float],
        kw: dict[str, tuple[float, list[str]]],
        query: str,
    ) -> list[DomainScore]:
        """Combine Axion-Cube semantic scores with keyword scores."""
        results = []
        max_kw = max((s for s, _ in kw.values()), default=1.0)
        all_domains = set(omega.keys()) | set(kw.keys())

        for domain in all_domains:
            oc = omega.get(domain, 0.0)
            kw_score, matched = kw.get(domain, (0.0, []))
            kw_norm = min(kw_score / max(max_kw, 1), 1.0)

            # Axion-Cube weight: 0.6, keyword: 0.4
            combined = 0.6 * oc + 0.4 * kw_norm if oc > 0 else kw_norm

            if combined > 0.08:
                results.append(DomainScore(
                    domain=domain,
                    omega_score=oc,
                    keyword_score=kw_norm,
                    tensor_score=0.0,
                    combined=round(combined, 4),
                    matched_keywords=matched,
                ))

        results.sort(key=lambda x: x.combined, reverse=True)
        return results

    # ══════════════════════════════════════════════════════════════
    # Depth, format, audience detection
    # ══════════════════════════════════════════════════════════════

    def _detect(self, query: str, patterns: dict[str, list[str]], default: str) -> str:
        q = query.lower()
        scores = Counter()
        for label, pats in patterns.items():
            for p in pats:
                if re.search(p, q):
                    scores[label] += 1
        return scores.most_common(1)[0][0] if scores else default

    def _detect_audience(self, query: str) -> str:
        q = query.lower()
        if re.search(r'\b(beginner|simple|basic|newbie|dummies|5.year.old)\b', q):
            return "layperson"
        if re.search(r'\b(advanced|expert|researcher|phd|publication)\b', q):
            return "expert"
        return "technical"

    # ══════════════════════════════════════════════════════════════
    # Shard matching
    # ══════════════════════════════════════════════════════════════

    def _match_shards(
        self, domain_scores: list[DomainScore], shards: list[ShardConfig]
    ) -> tuple[list[str], dict[str, str]]:
        active, reasons = [], {}
        shard_by_domain: dict[str, list[ShardConfig]] = {}
        for s in shards:
            if s.enabled:
                for d in s.domains:
                    shard_by_domain.setdefault(d, []).append(s)

        matched = set()
        for ds in domain_scores:
            if ds.combined < 0.12:
                continue
            for s in shard_by_domain.get(ds.domain, []):
                if s.name not in matched:
                    active.append(s.name)
                    matched.add(s.name)
                    reasons[s.name] = (
                        f"Axion-Cube: {ds.omega_score:.2f}, "
                        f"KW: {ds.keyword_score:.2f} → {ds.combined:.2f}"
                    )

        if not active and shards:
            fallback = next((s for s in shards if "general" in s.domains), shards[0])
            active.append(fallback.name)
            reasons[fallback.name] = "No domain matched; default shard"

        return active, reasons

    # ══════════════════════════════════════════════════════════════
    # Context building from Axion-Cube (v2: grounded + holographic)
    # ══════════════════════════════════════════════════════════════

    def _build_context(
        self, query: str, domain_scores: list[DomainScore]
    ) -> tuple[list[ContextNode], int]:
        """Build context from real Axion-Cube graph nodes.

        v2: Queries the actual graph for relevant nodes, attaches
        holographic signatures and grounding scores, then filters
        through BoundaryController.

        Returns: (context_nodes, boundary_filtered_count)
        """
        raw_nodes: list[ContextNode] = []
        boundary_filtered = 0

        # v2: Query real graph nodes instead of generating generic text
        if self._engine:
            for ds in domain_scores[:3]:
                if ds.combined < 0.2:
                    continue
                try:
                    hits = self._engine.query(
                        ds.domain, mode="hierarchical", top_k=3
                    )
                    for h in hits:
                        node_id = h.get("node_id", f"axion:{ds.domain}")
                        content = h.get("content", "")
                        score = h.get("score", ds.combined)
                        gray = h.get("gray_scale", 0.0)

                        # Get holographic signature from node
                        holo_sig = []
                        if node_id in self._engine.nodes:
                            node = self._engine.nodes[node_id]
                            if hasattr(node, 'holographic_signature') and node.holographic_signature:
                                holo_sig = node.holographic_signature

                        raw_nodes.append(ContextNode(
                            node_id=node_id,
                            content=content,
                            weight=score,
                            domain=ds.domain,
                            depth=1,
                            dimensions=[ds.domain, "knowledge_graph"],
                            holographic_signature=holo_sig,
                            grounding_score=score,
                            gray_scale=gray,
                        ))
                except Exception:
                    pass

        # Fallback: generic domain context (v1 behavior)
        if not raw_nodes:
            for ds in domain_scores[:3]:
                if ds.combined < 0.2:
                    continue
                info = STANDARD_DOMAINS.get(ds.domain, {})
                subs = info.get("subdomains", [])
                if subs:
                    raw_nodes.append(ContextNode(
                        node_id=f"axion:{ds.domain}",
                        content=f"Domain: {ds.domain}. Subdomains: {', '.join(subs[:5])}.",
                        weight=ds.combined,
                        domain=ds.domain,
                        depth=1,
                        dimensions=[ds.domain, "knowledge_graph"],
                    ))

        # v2: Filter through BoundaryController (PAGE-RAG)
        if self._boundary and raw_nodes:
            try:
                grounded = self._boundary.filter_grounded(raw_nodes, query)
                boundary_filtered = len(raw_nodes) - len(grounded)
                return grounded, boundary_filtered
            except Exception:
                pass

        return raw_nodes, boundary_filtered

    # ══════════════════════════════════════════════════════════════
    # v2: Evolving keyword rules from graph (CORTEX-inspired)
    # ══════════════════════════════════════════════════════════════

    def _refresh_keywords_from_graph(self):
        """Extract keywords from graph nodes to evolve domain classification.

        Inspired by CORTEX (arXiv 2607.18821): ontology auto-evolution.
        v3: Discovers NEW domains from node hierarchies (not just STANDARD_DOMAINS).
        Any hierarchy path like 'devops.kubernetes' registers 'devops' as a domain.
        """
        if not self._engine:
            return
        try:
            # ── Phase 1: Discover ALL domains from node hierarchies ──
            domain_nodes: dict[str, list] = {}
            for n in self._engine.nodes.values():
                hierarchies = getattr(n, 'hierarchies', [])
                if not hierarchies:
                    continue
                for h in hierarchies:
                    # Normalize: support both "a.b" and "a/b" separators
                    normalized = h.replace("/", ".")
                    domain = normalized.split(".")[0].lower().strip()
                    if domain and len(domain) >= 2:
                        domain_nodes.setdefault(domain, []).append(n)

            # ── Phase 2: Extract keywords per domain ──
            for domain, nodes in domain_nodes.items():
                # Get or create keyword list for this domain
                if domain in STANDARD_DOMAINS:
                    existing_kws = set(STANDARD_DOMAINS[domain]["keywords"])
                else:
                    existing_kws = set()

                # Collect current keywords already indexed for this domain
                for kw, domains in self._kw_to_domain.items():
                    if domain in domains:
                        existing_kws.add(kw)

                new_keywords = set()
                for n in nodes:
                    # From tags
                    if hasattr(n, 'tags') and n.tags:
                        for tag in n.tags:
                            tag_lower = tag.lower().strip()
                            if len(tag_lower) >= 3 and tag_lower not in existing_kws:
                                new_keywords.add(tag_lower)
                    # From content: significant words (len >= 4, not stopwords)
                    if hasattr(n, 'content') and n.content:
                        words = re.findall(r'\b[a-z]{4,}\b', n.content.lower())
                        for w in words:
                            if w not in _STOPWORDS and w not in existing_kws:
                                new_keywords.add(w)

                # Register domain name itself as a keyword
                if domain not in existing_kws:
                    new_keywords.add(domain)

                # Add to keyword index (max 30 new per domain)
                added = 0
                for kw in sorted(new_keywords):
                    if added >= 30:
                        break
                    self._kw_to_domain.setdefault(kw, []).append(domain)
                    added += 1
        except Exception:
            pass

    @staticmethod
    def _estimate_savings(active: int, total: int) -> float:
        if total == 0:
            return 0.0
        ratio = active / max(total, 1)
        effective = 0.30 + 0.70 * ratio
        return round(max(0.0, 1.0 - effective), 4)
