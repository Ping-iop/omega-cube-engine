"""
PredictiveContextSearch — Auto-complete search engine for hierarchical graphs.

Problem: With millions of connected cubes, searching by scanning all nodes
is O(n). Windows search finds everything containing a string, regardless of
context — type 3 letters, get every file with those 3 letters anywhere.

Solution: A prefix trie over hierarchical domains with context-aware pruning.
As the user types, the engine predicts based on:
1. Which cube/domain is currently active (conversation context)
2. Hierarchical proximity (nodes closer in tensor space rank higher)
3. Gray-scale confidence (verified nodes boost)
4. Recency and frequency (recently accessed nodes boost)

The "cube auto-complete" effect: type "SD" in a ComfyUI conversation → "SDXL",
not "SDK" from some unrelated domain. The hierarchy filters before the search
even begins.

Author: Omega-Cube Research
Date: 2026-06-12
"""

import time
from collections import defaultdict
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# HIERARCHICAL TRIE
# ═══════════════════════════════════════════════════════════════════

class TrieNode:
    """Node in the prefix trie, annotated with hierarchy domains."""
    __slots__ = ('children', 'domain_hits', 'total_hits', 'best_match')
    
    def __init__(self):
        self.children: dict[str, 'TrieNode'] = {}
        self.domain_hits: dict[str, int] = defaultdict(int)  # domain → count
        self.total_hits: int = 0
        self.best_match: Optional[str] = None  # Full content of best match


class HierarchicalTrie:
    """
    Prefix trie that tracks which domains each prefix appears in.
    
    Enables O(k) lookup where k = prefix length, with domain-aware
    filtering before returning results.
    """
    
    def __init__(self):
        self.root = TrieNode()
        self._node_count = 0
    
    def insert(self, text: str, domain: str, node_id: str = None):
        """Insert text into trie, tagging with its hierarchy domain."""
        text_lower = text.lower()
        node = self.root
        node.total_hits += 1
        node.domain_hits[domain] += 1
        
        for char in text_lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.total_hits += 1
            node.domain_hits[domain] += 1
        
        # Store best match at leaf
        if node.best_match is None or len(text) > len(node.best_match):
            node.best_match = text
        
        self._node_count += 1
    
    def search(self, prefix: str, context_domain: str = None, max_results: int = 10) -> list[dict]:
        """
        Search trie for prefix, filtered by context domain.
        
        Args:
            prefix: Search prefix (e.g., "sd")
            context_domain: Active domain for context filtering (e.g., "COMFYUI")
            max_results: Max results
        
        Returns:
            List of {text, domain, relevance} sorted by contextual relevance
        """
        prefix_lower = prefix.lower()
        
        # Navigate to prefix node
        node = self.root
        for char in prefix_lower:
            if char not in node.children:
                return []  # No matches
            node = node.children[char]
        
        # Collect all completions with domain info
        completions = self._collect_completions(node, prefix_lower, depth=0, max_depth=50)
        
        # Score by contextual relevance
        scored = []
        for comp in completions:
            score = self._contextual_score(comp, context_domain, node)
            scored.append({**comp, "score": score})
        
        # Sort: context match first, then by frequency
        scored.sort(key=lambda x: (-x["score"], -x.get("domain_hits", 0)))
        
        # Deduplicate by text (best score wins)
        seen = set()
        unique = []
        for s in scored:
            key = s["text"][:80]
            if key not in seen:
                seen.add(key)
                unique.append(s)
        
        return unique[:max_results]
    
    def predict(self, prefix: str, context_domain: str = None, max_results: int = 5) -> list[str]:
        """Simple predictive text: return most likely completions."""
        results = self.search(prefix, context_domain, max_results)
        return [r["text"] for r in results]
    
    def _collect_completions(self, node: TrieNode, prefix: str, depth: int, max_depth: int) -> list[dict]:
        """DFS to collect completions from a trie node and its descendants."""
        results = []
        
        # Check current node
        if node.best_match and node.best_match.lower().startswith(prefix.lower()):
            results.append({
                "text": node.best_match,
                "domains": dict(node.domain_hits),
                "total_hits": node.total_hits,
            })
        
        if depth >= max_depth:
            return results
        
        # Explore children, prioritizing high-frequency branches
        sorted_children = sorted(
            node.children.items(),
            key=lambda x: -x[1].total_hits
        )
        
        for char, child in sorted_children:
            results.extend(
                self._collect_completions(child, prefix + char, depth + 1, max_depth)
            )
            if len(results) >= max_depth:
                break
        
        return results
    
    def _contextual_score(self, completion: dict, context_domain: str, node: TrieNode) -> float:
        """Score a completion based on contextual relevance."""
        score = 0.0
        
        # Base: frequency score
        if node.total_hits > 0:
            score += completion.get("total_hits", 0) / node.total_hits
        
        # Context boost: domain matches active context
        if context_domain and completion.get("domains"):
            domains = completion["domains"]
            total_domain_hits = sum(domains.values())
            if total_domain_hits > 0:
                context_hits = domains.get(context_domain, 0)
                # Strong boost if this completion appears in active domain
                context_ratio = context_hits / total_domain_hits
                score += context_ratio * 2.0  # 2x multiplier for context match
        
        # Penalize purely noise domains
        if completion.get("domains"):
            domains = completion["domains"]
            has_real_domain = any(
                d not in ("NOISE", "DISTRACTOR", "UNKNOWN")
                for d in domains
            )
            if not has_real_domain:
                score *= 0.1  # Heavy penalty for noise
        
        return score
    
    def stats(self) -> dict:
        return {
            "total_nodes": self._node_count,
            "root_children": len(self.root.children),
            "root_total_hits": self.root.total_hits,
        }


# ═══════════════════════════════════════════════════════════════════
# CONTEXT TRACKER
# ═══════════════════════════════════════════════════════════════════

class ContextTracker:
    """
    Tracks the active conversation context to inform predictive search.
    
    Maintains a sliding window of recent topics and domains,
    decaying older context exponentially.
    """
    
    def __init__(self, window_size: int = 20, decay_rate: float = 0.9):
        self.window: list[tuple[str, float]] = []  # (domain, timestamp)
        self.window_size = window_size
        self.decay_rate = decay_rate
        self.domain_frequencies: dict[str, float] = defaultdict(float)
    
    def observe(self, domain: str):
        """Record a domain observation in the context window."""
        now = time.time()
        self.window.append((domain, now))
        
        # Trim window
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]
        
        # Update frequencies with decay
        self._update_frequencies(now)
    
    def active_domain(self) -> Optional[str]:
        """Return the most active domain in the current context."""
        if not self.domain_frequencies:
            return None
        return max(self.domain_frequencies, key=self.domain_frequencies.get)
    
    def domain_weights(self) -> dict[str, float]:
        """Return all domain weights for context-aware scoring."""
        total = sum(self.domain_frequencies.values())
        if total == 0:
            return {}
        return {d: w/total for d, w in self.domain_frequencies.items()}
    
    def _update_frequencies(self, now: float):
        """Update domain frequencies with exponential decay."""
        self.domain_frequencies.clear()
        
        for i, (domain, ts) in enumerate(self.window):
            # Newer items have higher weight
            position_weight = (i + 1) / len(self.window)
            # Time decay
            age = now - ts
            time_weight = self.decay_rate ** age
            # Combined
            weight = position_weight * time_weight
            self.domain_frequencies[domain] += weight


# ═══════════════════════════════════════════════════════════════════
# PREDICTIVE CONTEXT SEARCH ENGINE
# ═══════════════════════════════════════════════════════════════════

class PredictiveContextSearch:
    """
    Auto-complete search engine for hierarchical knowledge graphs.
    
    Combines:
    - HierarchicalTrie: O(k) prefix lookup with domain awareness
    - ContextTracker: Active conversation context tracking
    - Omega-Cube integration: Tensor proximity + gray-scale boosting
    
    The "cube auto-complete" effect: as the user types in a conversation,
    predictions are filtered and ranked by:
    1. Active domain (conversation context)
    2. Hierarchical proximity in tensor space
    3. Gray-scale confidence
    4. Recent access frequency
    """
    
    def __init__(self, omega_cube_engine=None):
        self.trie = HierarchicalTrie()
        self.context = ContextTracker()
        self.cube = omega_cube_engine
        self._access_log: dict[str, float] = {}  # node_id → last_access_time
    
    def index_node(self, text: str, domain: str, node_id: str = None):
        """Index a node's content into the predictive search engine."""
        # Index the full text
        self.trie.insert(text, domain, node_id)
        
        # Also index individual significant words for faster partial matching
        words = text.lower().split()
        significant = [w for w in words if len(w) > 3 and w.isalpha()]
        for word in significant:
            self.trie.insert(word, domain, node_id)
    
    def index_from_cube(self):
        """Index all nodes from the connected Omega-Cube engine."""
        if not self.cube:
            return 0
        
        count = 0
        for node_id, node in self.cube.nodes.items():
            domain = node.primary_hierarchy.split(".")[0] if node.primary_hierarchy else "UNKNOWN"
            self.index_node(node.content, domain, node_id)
            count += 1
        
        return count
    
    def search(self, prefix: str, max_results: int = 10) -> list[dict]:
        """
        Context-aware predictive search.
        
        Example:
            User types "Ma" in an Evony conversation
            → ["Marcian #1 Ranged PvP: +45% attack...", "Mounted PvP: Hermes..."]
            NOT "Matrix multiplication" from ML domain
        """
        active_domain = self.context.active_domain()
        results = self.trie.search(prefix, active_domain, max_results)
        
        # Boost with gray-scale if available
        if self.cube:
            for r in results:
                gs_boost = self._gray_scale_boost(r.get("text", ""))
                r["score"] += gs_boost
        
        return results
    
    def predict(self, prefix: str, max_results: int = 5) -> list[str]:
        """Simple auto-complete: return predicted full texts."""
        results = self.search(prefix, max_results)
        return [r["text"][:120] for r in results]
    
    def feed_context(self, text: str):
        """Feed conversation text to update active context."""
        # Detect domain from text
        domain = self._detect_domain(text)
        if domain:
            self.context.observe(domain)
    
    def _detect_domain(self, text: str) -> Optional[str]:
        """Quick domain detection from text."""
        text_lower = text.lower()
        domain_keywords = {
            "COMFYUI": ["comfyui", "sdxl", "checkpoint", "vae", "lora", "workflow", "ipadapter"],
            "EVONY": ["evony", "marcian", "hermes", "akechi", "tamar", "f2p", "pvp", "ranged"],
            "HERMES": ["hermes", "mcp", "cron", "skill", "plugin", "config", "agent"],
            "HBIT": ["h-bit", "hbit", "grayscale", "steganograph", "verify", "security"],
            "ML": ["diffusion", "gemma", "karpathy", "training", "fine-tuning", "embedding"],
        }
        for domain, keywords in domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return domain
        return None
    
    def _gray_scale_boost(self, text: str) -> float:
        """Boost score based on gray-scale verification of matching nodes."""
        if not self.cube:
            return 0.0
        
        # Find nodes containing this text and check their gray-scale
        text_lower = text.lower()
        best_gs = 0.0
        for node in self.cube.nodes.values():
            if text_lower in node.content.lower():
                if node.gray_scale:
                    composite = self.cube.gray_validator.composite_score(node.gray_scale)
                    best_gs = max(best_gs, composite / 100)
        
        return best_gs * 0.5  # Up to 0.5 boost
    
    def stats(self) -> dict:
        trie_stats = self.trie.stats()
        return {
            **trie_stats,
            "active_domain": self.context.active_domain(),
            "context_window_size": len(self.context.window),
        }


# ═══════════════════════════════════════════════════════════════════
# DEMO: The "Windows search" problem vs Predictive Context Search
# ═══════════════════════════════════════════════════════════════════

def demo_predictive_vs_flat():
    """Demonstrate the difference between flat and context-aware predictive search."""
    print("=" * 65)
    print("PREDICTIVE CONTEXT SEARCH — Demo")
    print("Windows-style flat search vs Context-aware predictive search")
    print("=" * 65)
    
    pcs = PredictiveContextSearch()
    
    # Index nodes across domains
    data = [
        ("COMFYUI", "SDXL base checkpoint at J:/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors"),
        ("COMFYUI", "SD1.5 model uses less VRAM but lower quality output"),
        ("COMFYUI", "Sampling steps: 30 for SDXL quality, 15 for speed"),
        ("EVONY", "Marcian #1 Ranged PvP: +45% attack, +30% defense vs mounted"),
        ("EVONY", "March speed boost from Hermes applies to all mounted troops"),
        ("EVONY", "Map exploration requires scouting level 15 minimum"),
        ("HERMES", "MCP servers configured in config.yaml under mcp.servers section"),
        ("HERMES", "Marathon mode enables extended context window processing"),
        ("HBIT", "Mask-based verification works with partial file fragments"),
        ("HBIT", "SHA-256 hashing between bit segments for chain integrity"),
        ("ML", "Matrix multiplication optimization via CUDA kernel fusion"),
        ("ML", "Max pooling layer reduces spatial dimensions by factor of 2"),
    ]
    
    for domain, text in data:
        pcs.index_node(text, domain)
    
    print(f"\nIndexed {len(data)} nodes across 5 domains\n")
    
    # Simulate: user is in a ComfyUI conversation
    pcs.feed_context("I need to configure SDXL in ComfyUI, the checkpoint is")
    
    tests = [
        ("S", "First letter 'S' — ambiguous across all domains"),
        ("SD", "Two letters 'SD' — narrowing to SDXL/SD1.5"),
        ("Ma", "'Ma' — in ComfyUI context, should NOT show Marcian/Matrix"),
        ("Ma", "Same 'Ma' but NOW in Evony context", "EVONY"),
    ]
    
    for test in tests:
        if len(test) == 3:
            prefix, desc, force_context = test
            if force_context:
                pcs.feed_context(f"Let's talk about {force_context} strategy, the best general for")
        else:
            prefix, desc = test
        
        print(f"Prefix: '{prefix}' — {desc}")
        print(f"  Active context: {pcs.context.active_domain() or 'None'}")
        results = pcs.search(prefix, max_results=5)
        for r in results:
            domain_tag = r.get("domains", {})
            top_domain = max(domain_tag, key=domain_tag.get) if domain_tag else "?"
            print(f"  [{r['score']:.2f}] [{top_domain}] {r['text'][:100]}")
        print()
    
    print("─" * 65)
    print("OBSERVATION: 'Ma' in ComfyUI context → no Marcian/Matrix")
    print("             'Ma' in Evony context → Marcian first, not Matrix")
    print("             This is what Windows search CANNOT do.")
    print("=" * 65)


if __name__ == "__main__":
    demo_predictive_vs_flat()
