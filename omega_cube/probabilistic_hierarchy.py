"""
ProbabilisticHierarchy — Hierarchies that dance but don't break.

Four-layer architecture:
  Layer 0 - IMMUTABLE:  Axioms (fire burns, 2+2=4). Variance ≈ 0. Never shift.
  Layer 1 - PROBABILISTIC: Scientific consensus (sun ~5,500°C). Shifts with evidence.
  Layer 2 - EMERGENT:     User behavior patterns. Evolves from collective usage.
  Layer 3 - FLUID:        Real-time session data. High variance, rapid adaptation.

Each node has a probability distribution, not a fixed position.
Bayesian updating: new evidence shifts the distribution toward truth.
Anchoring: immutable axioms prevent cascade drift.

Example: "Fire is hot" (IMMUTABLE, σ²≈0)
         "Solar temperature is 5,500°C" (PROBABILISTIC, σ²=100 — refineable)
         "Users connect ComfyUI+Evony for game assets" (EMERGENT, σ²=400 — behavioral)
         "Current session: debugging SDXL checkpoint" (FLUID, σ²=1000 — temporary)

Author: Omega-Cube Research
Date: 2026-06-12
"""

import math, json, os, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════
# PROBABILISTIC NODE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ProbabilisticNode:
    """
    A knowledge node with a probability distribution instead of a fixed position.
    
    The node's "position" in the hierarchy is a Gaussian with mean μ and variance σ².
    Low variance = anchored (axioms). High variance = fluid (session data).
    
    Bayesian updating: new evidence E updates the distribution:
        μ' = (μ/σ² + E/σe²) / (1/σ² + 1/σe²)
        σ'² = 1 / (1/σ² + 1/σe²)
    """
    
    content: str
    hierarchy: str
    layer: str  # IMMUTABLE | PROBABILISTIC | EMERGENT | FLUID
    
    # Distribution parameters
    mean: float = 0.5          # μ: central tendency in [0,1] space
    variance: float = 0.01     # σ²: uncertainty (lower = more anchored)
    
    # Evidence tracking
    evidence_count: int = 0
    last_updated: float = 0.0
    source_signals: list[str] = field(default_factory=list)  # What updated this
    
    # Bayesian priors
    prior_mean: float = 0.5
    prior_variance: float = 0.01
    
    # Immutability
    is_immutable: bool = False
    
    def __post_init__(self):
        self.prior_mean = self.mean
        self.prior_variance = self.variance
        if self.layer == "IMMUTABLE":
            self.is_immutable = True
            self.variance = 1e-10  # Effectively zero
    
    @property
    def confidence(self) -> float:
        """Confidence = 1 / (1 + variance). Higher variance → lower confidence."""
        return 1.0 / (1.0 + self.variance)
    
    @property
    def stability(self) -> float:
        """How much this node resists change. IMMUTABLE = ∞, FLUID ≈ 0."""
        if self.is_immutable:
            return float('inf')
        return 1.0 / self.variance if self.variance > 0 else float('inf')
    
    def bayesian_update(self, evidence: float, evidence_variance: float, source: str = "") -> float:
        """
        Update the distribution with new evidence using Bayes' theorem.
        
        Args:
            evidence: The new observed value (in [0,1])
            evidence_variance: Uncertainty of the evidence
            source: What provided this evidence (paper, session, axiom)
        
        Returns:
            shift: How much the mean shifted (|new - old|)
        """
        if self.is_immutable:
            return 0.0  # Cannot shift
        
        old_mean = self.mean
        
        # Precision-weighted average (Bayesian update for Gaussian)
        precision_prior = 1.0 / self.variance if self.variance > 0 else float('inf')
        precision_evidence = 1.0 / evidence_variance if evidence_variance > 0 else float('inf')
        
        total_precision = precision_prior + precision_evidence
        
        if total_precision > 0:
            self.mean = (self.mean * precision_prior + evidence * precision_evidence) / total_precision
            self.variance = 1.0 / total_precision
        
        self.evidence_count += 1
        self.last_updated = time.time()
        self.source_signals.append(source)
        
        # Keep last 10 sources
        if len(self.source_signals) > 10:
            self.source_signals = self.source_signals[-10:]
        
        return abs(self.mean - old_mean)
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "hierarchy": self.hierarchy,
            "layer": self.layer,
            "mean": self.mean,
            "variance": self.variance,
            "confidence": self.confidence,
            "stability": self.stability if self.stability != float('inf') else "infinite",
            "evidence_count": self.evidence_count,
            "is_immutable": self.is_immutable,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ProbabilisticNode":
        node = cls(
            content=d.get("content", ""),
            hierarchy=d.get("hierarchy", ""),
            layer=d.get("layer", "EMERGENT"),
            mean=d.get("mean", 0.5),
            variance=d.get("variance", 0.01),
        )
        node.evidence_count = d.get("evidence_count", 0)
        return node


# ═══════════════════════════════════════════════════════════════════
# PROBABILISTIC HIERARCHY ENGINE
# ═══════════════════════════════════════════════════════════════════

class ProbabilisticHierarchyEngine:
    """
    Four-layer probabilistic hierarchy that evolves via Bayesian updating.
    
    Layers:
      IMMUTABLE (σ²≈0):     "Fire requires oxygen, fuel, heat"
      PROBABILISTIC (σ²~100): "Solar temperature ~5,500°C" (can refine)
      EMERGENT (σ²~400):     "Users connect ComfyUI+Evony" (behavioral)
      FLUID (σ²~1000):       "Current session: debugging SDXL" (temporary)
    
    Evidence sources and their typical variance:
      AXIOM:             σ²=1e-10 (absolute truth)
      PEER_REVIEWED:     σ²=50    (published paper)
      BENCHMARK:         σ²=30    (experimental result)
      USER_SESSION:      σ²=500   (single user behavior)
      COLLECTIVE_USAGE:  σ²=200   (aggregated user behavior)
      NEW_PAPER:         σ²=100   (recent publication)
      EXPERT_OPINION:    σ²=80    (domain expert)
      REAL_TIME_QUERY:   σ²=1000  (current conversation)
    """
    
    SOURCE_VARIANCES = {
        "AXIOM": 1e-10,
        "PEER_REVIEWED": 50,
        "BENCHMARK": 30,
        "USER_SESSION": 500,
        "COLLECTIVE_USAGE": 200,
        "NEW_PAPER": 100,
        "EXPERT_OPINION": 80,
        "REAL_TIME_QUERY": 1000,
    }
    
    LAYER_ORDER = ["IMMUTABLE", "PROBABILISTIC", "EMERGENT", "FLUID"]
    
    def __init__(self):
        self.nodes: dict[str, ProbabilisticNode] = {}
        self.updates_log: list[dict] = []
        self.total_updates: int = 0
        self.total_shift: float = 0.0
    
    def add_axiom(self, content: str, hierarchy: str) -> ProbabilisticNode:
        """Add an immutable truth. Variance ≈ 0. Cannot be shifted."""
        node = ProbabilisticNode(
            content=content,
            hierarchy=hierarchy,
            layer="IMMUTABLE",
            mean=1.0,   # Absolute truth
            variance=1e-10,
        )
        self.nodes[hierarchy] = node
        return node
    
    def add_probabilistic(self, content: str, hierarchy: str, confidence: float = 0.9) -> ProbabilisticNode:
        """Add a scientifically-established fact that can be refined."""
        variance = (1.0 - confidence) * 10  # Higher confidence → lower variance
        node = ProbabilisticNode(
            content=content,
            hierarchy=hierarchy,
            layer="PROBABILISTIC",
            mean=confidence,
            variance=max(0.001, variance),
        )
        self.nodes[hierarchy] = node
        return node
    
    def add_emergent(self, content: str, hierarchy: str) -> ProbabilisticNode:
        """Add a user-behavior-discovered connection."""
        node = ProbabilisticNode(
            content=content,
            hierarchy=hierarchy,
            layer="EMERGENT",
            mean=0.5,    # Neutral starting point
            variance=0.8,  # High initial uncertainty
        )
        self.nodes[hierarchy] = node
        return node
    
    def add_fluid(self, content: str, hierarchy: str) -> ProbabilisticNode:
        """Add a temporary session-specific fact."""
        node = ProbabilisticNode(
            content=content,
            hierarchy=hierarchy,
            layer="FLUID",
            mean=0.5,
            variance=2.0,  # Very high variance
        )
        self.nodes[hierarchy] = node
        return node
    
    def update_from_evidence(
        self,
        hierarchy: str,
        evidence: float,
        source_type: str,
        source_detail: str = "",
    ) -> float:
        """
        Update a node with new evidence from a specific source.
        
        The source type determines the evidence variance (how much we trust it).
        Axiom evidence has near-zero variance → barely shifts anything.
        Session evidence has high variance → shifts FLUID nodes easily.
        
        Anchoring: evidence from lower-trust sources cannot significantly
        shift nodes in higher layers. A user session cannot move an axiom.
        """
        if hierarchy not in self.nodes:
            # Auto-create as FLUID if not exists
            self.add_fluid(f"Auto-created from {source_detail}", hierarchy)
        
        node = self.nodes[hierarchy]
        
        # Get evidence variance based on source
        evidence_variance = self.SOURCE_VARIANCES.get(source_type, 500)
        
        # Anchoring protection: high-layer nodes resist low-trust evidence
        layer_idx = self.LAYER_ORDER.index(node.layer) if node.layer in self.LAYER_ORDER else 2
        source_trust = 1.0 / (1.0 + evidence_variance / 100)  # 0 (untrusted) to 1 (absolute)
        
        # If source is low-trust and node is high-layer, dampen the evidence
        if layer_idx <= 1 and source_trust < 0.5:
            # PROBABILISTIC or IMMUTABLE node, but weak evidence → reduce impact
            evidence_variance *= 5.0  # Make evidence seem less certain
        
        # Apply Bayesian update
        shift = node.bayesian_update(evidence, evidence_variance, source_detail)
        
        # Log
        self.total_updates += 1
        self.total_shift += shift
        self.updates_log.append({
            "hierarchy": hierarchy,
            "layer": node.layer,
            "mean_after": node.mean,
            "variance_after": node.variance,
            "shift": shift,
            "source": source_type,
            "detail": source_detail[:100],
            "timestamp": time.time(),
        })
        
        # Keep log manageable
        if len(self.updates_log) > 1000:
            self.updates_log = self.updates_log[-500:]
        
        return shift
    
    def update_from_collective_engine(self, collective_engine):
        """
        Feed signals from CollectiveHierarchyEngine into the probabilistic hierarchy.
        
        Transitions → EMERGENT layer updates
        Co-occurrences → PROBABILISTIC layer updates (if strong enough)
        Domain activity → FLUID layer updates
        """
        changes = 0
        
        # Process transitions (EMERGENT layer)
        for (from_d, to_d), weight in collective_engine.transition_matrix.items():
            hierarchy = f"TRANSITION.{from_d}.{to_d}"
            if hierarchy not in self.nodes:
                self.add_emergent(
                    f"Users navigate from {from_d} to {to_d}",
                    hierarchy
                )
            # Normalize weight by session count
            normalized = min(1.0, weight / max(1, collective_engine.total_sessions))
            shift = self.update_from_evidence(
                hierarchy,
                normalized,
                "COLLECTIVE_USAGE",
                f"transition signal (weight={weight:.1f})"
            )
            if shift > 0.001:
                changes += 1
        
        # Process co-occurrences (PROBABILISTIC if strong, EMERGENT if weak)
        for (a, b), weight in collective_engine.co_occurrence_matrix.items():
            hierarchy = f"COOCCUR.{a}.{b}"
            if hierarchy not in self.nodes:
                normalized = min(1.0, weight / max(1, collective_engine.total_signals_processed))
                if normalized > 0.3:
                    self.add_probabilistic(
                        f"Domains {a} and {b} are strongly connected",
                        hierarchy,
                        confidence=0.7
                    )
                else:
                    self.add_emergent(
                        f"Domains {a} and {b} co-occur in sessions",
                        hierarchy
                    )
            normalized = min(1.0, weight / max(1, collective_engine.total_signals_processed))
            source_type = "COLLECTIVE_USAGE" if normalized < 0.3 else "PEER_REVIEWED"
            shift = self.update_from_evidence(
                hierarchy,
                normalized,
                source_type,
                f"co-occurrence (weight={weight:.1f})"
            )
            if shift > 0.001:
                changes += 1
        
        return changes
    
    def ingest_paper(self, title: str, findings: dict, confidence: float = 0.7):
        """Ingest findings from a research paper as PROBABILISTIC evidence."""
        for key, value in findings.items():
            hierarchy = f"PAPER.{title[:30].replace(' ', '_')}.{key}"
            if hierarchy not in self.nodes:
                self.add_probabilistic(
                    f"From {title}: {key} = {value}",
                    hierarchy,
                    confidence=confidence
                )
            self.update_from_evidence(
                hierarchy,
                float(value) if isinstance(value, (int, float)) else 0.5,
                "NEW_PAPER",
                f"{title}: {key}"
            )
    
    def get_layer_stats(self) -> dict:
        """Statistics per hierarchy layer."""
        stats = {layer: {"count": 0, "avg_confidence": 0, "avg_variance": 0, "total_shift": 0}
                 for layer in self.LAYER_ORDER}
        
        for node in self.nodes.values():
            if node.layer in stats:
                s = stats[node.layer]
                s["count"] += 1
                s["avg_confidence"] += node.confidence
                s["avg_variance"] += node.variance
        
        for layer in stats:
            s = stats[layer]
            if s["count"] > 0:
                s["avg_confidence"] /= s["count"]
                s["avg_variance"] /= s["count"]
        
        return stats
    
    def get_hierarchy_snapshot(self) -> dict:
        """Full snapshot: how the hierarchy looks right now."""
        snapshot = {
            "nodes": {},
            "layer_stats": self.get_layer_stats(),
            "total_nodes": len(self.nodes),
            "total_updates": self.total_updates,
            "total_shift": self.total_shift,
            "avg_shift_per_update": self.total_shift / max(1, self.total_updates),
        }
        
        for hierarchy, node in self.nodes.items():
            snapshot["nodes"][hierarchy] = node.to_dict()
        
        return snapshot
    
    def save(self, path: str):
        data = {
            "nodes": {h: n.to_dict() for h, n in self.nodes.items()},
            "total_updates": self.total_updates,
            "total_shift": self.total_shift,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        for h, d in data.get("nodes", {}).items():
            self.nodes[h] = ProbabilisticNode.from_dict(d)
        self.total_updates = data.get("total_updates", 0)
        self.total_shift = data.get("total_shift", 0.0)
        return True


# ═══════════════════════════════════════════════════════════════════
# DEMO & BENCHMARK
# ═══════════════════════════════════════════════════════════════════

def demo_probabilistic_hierarchy():
    """Demonstrate how hierarchies dance but don't break."""
    print("=" * 70)
    print("PROBABILISTIC HIERARCHY — Demo")
    print("'Fire burns' is immutable. 'How hot?' evolves with evidence.")
    print("=" * 70)
    
    engine = ProbabilisticHierarchyEngine()
    
    # ─── Layer 0: IMMUTABLE axioms ───
    engine.add_axiom("Fire requires oxygen, fuel, and heat", "SCIENCE.FIRE.REQUIREMENTS")
    engine.add_axiom("2 + 2 = 4", "MATH.ARITHMETIC.ADDITION")
    engine.add_axiom("Water freezes at 0°C at sea level", "SCIENCE.WATER.FREEZING")
    
    # ─── Layer 1: PROBABILISTIC scientific facts ───
    engine.add_probabilistic("Solar surface temperature is ~5,500°C", "SCIENCE.SUN.TEMPERATURE", 0.95)
    engine.add_probabilistic("SDXL produces higher quality than SD1.5", "AI.MODELS.SDXL.QUALITY", 0.85)
    engine.add_probabilistic("Marcian is #1 Ranged PvP general", "EVONY.GENERALS.RANGED.MARCIAN", 0.80)
    
    # ─── Layer 2: EMERGENT from user behavior ───
    engine.add_emergent("Users connect ComfyUI with Evony for game assets", "EMERGENT.COMFYUI.EVONY")
    engine.add_emergent("Hermes is the central hub for all domain conversations", "EMERGENT.HERMES.HUB")
    
    # ─── Layer 3: FLUID session data ───
    engine.add_fluid("Current session: configuring SDXL checkpoint", "SESSION.CURRENT.SDXL")
    
    print(f"\nInitial hierarchy: {len(engine.nodes)} nodes across 4 layers")
    
    stats = engine.get_layer_stats()
    print(f"\n═══ LAYER STATISTICS (initial) ═══")
    for layer in engine.LAYER_ORDER:
        s = stats[layer]
        if s["count"] > 0:
            print(f"  {layer:<15} nodes={s['count']} confidence={s['avg_confidence']:.3f} variance={s['avg_variance']:.6f}")
    
    # ─── Simulate evidence: new paper refines solar temperature ───
    print(f"\n═══ EVIDENCE INJECTION ═══")
    
    print(f"\n1. New paper: 'Solar temperature revised to 5,772K'")
    engine.update_from_evidence("SCIENCE.SUN.TEMPERATURE", 0.97, "NEW_PAPER", "Precision solar measurement 2026")
    node = engine.nodes["SCIENCE.SUN.TEMPERATURE"]
    print(f"   Mean: {node.prior_mean:.3f} → {node.mean:.3f} | Variance: {node.prior_variance:.3f} → {node.variance:.3f}")
    
    print(f"\n2. User sessions: 100 users connect ComfyUI+Evony")
    for _ in range(100):
        engine.update_from_evidence("EMERGENT.COMFYUI.EVONY", 0.7, "USER_SESSION", "game asset generation")
    node = engine.nodes["EMERGENT.COMFYUI.EVONY"]
    print(f"   Mean: {node.prior_mean:.3f} → {node.mean:.3f} | Confidence: {node.prior_mean:.3f} → {node.confidence:.3f}")
    
    print(f"\n3. Axiom attack: someone claims fire doesn't need oxygen")
    before = engine.nodes["SCIENCE.FIRE.REQUIREMENTS"].mean
    engine.update_from_evidence("SCIENCE.FIRE.REQUIREMENTS", 0.1, "USER_SESSION", "fire doesn't need oxygen claim")
    after = engine.nodes["SCIENCE.FIRE.REQUIREMENTS"].mean
    print(f"   Mean: {before:.10f} → {after:.10f} (shift: {abs(after-before):.2e})")
    print(f"   IMMUTABLE protection: axiom resisted change")
    
    print(f"\n4. Benchmark result: SDXL quality confirmed at 92%")
    engine.update_from_evidence("AI.MODELS.SDXL.QUALITY", 0.92, "BENCHMARK", "LoCoMo benchmark 2026")
    node = engine.nodes["AI.MODELS.SDXL.QUALITY"]
    print(f"   Mean: {node.prior_mean:.3f} → {node.mean:.3f} | Confidence: {node.confidence:.3f}")
    
    print(f"\n5. Collective usage: HERMES hub confirmed by 27 sessions")
    engine.update_from_evidence("EMERGENT.HERMES.HUB", 0.96, "COLLECTIVE_USAGE", "1064 signals from 27 sessions")
    node = engine.nodes["EMERGENT.HERMES.HUB"]
    print(f"   Mean: {node.prior_mean:.3f} → {node.mean:.3f} | Confidence: {node.confidence:.3f}")
    print(f"   Promoted from EMERGENT toward PROBABILISTIC confidence level")
    
    # ─── Final state ───
    stats = engine.get_layer_stats()
    print(f"\n═══ LAYER STATISTICS (after evidence) ═══")
    for layer in engine.LAYER_ORDER:
        s = stats[layer]
        if s["count"] > 0:
            print(f"  {layer:<15} nodes={s['count']} confidence={s['avg_confidence']:.3f} variance={s['avg_variance']:.6f}")
    
    print(f"\n═══ KEY INSIGHT ═══")
    print(f"  Total updates: {engine.total_updates}")
    print(f"  Total shift: {engine.total_shift:.4f}")
    print(f"  IMMUTABLE axioms: protected from drift")
    print(f"  PROBABILISTIC facts: refined by evidence")
    print(f"  EMERGENT patterns: promoted with collective usage")
    print(f"  FLUID sessions: high variance, rapid adaptation")
    print(f"  Architecture: Bayesian updating with anchoring")
    
    return engine


def benchmark_immutable_vs_flexible():
    """Test: can user sessions shift an axiom? (Should be NO.)"""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Immutable Protection")
    print(f"{'='*70}")
    
    engine = ProbabilisticHierarchyEngine()
    engine.add_axiom("Fire requires oxygen", "SCIENCE.FIRE.OXYGEN")
    engine.add_probabilistic("SDXL quality score: 0.85", "AI.SDXL.QUALITY")
    
    # Attack axiom with 1000 user sessions
    axiom_shifts = []
    for _ in range(1000):
        shift = engine.update_from_evidence(
            "SCIENCE.FIRE.OXYGEN", 0.01, "USER_SESSION", "false claim"
        )
        axiom_shifts.append(shift)
    
    # Update probabilistic with same evidence
    prob_shifts = []
    for _ in range(1000):
        # Recreate node each time to measure shift cleanly
        engine.nodes["AI.SDXL.QUALITY"] = ProbabilisticNode(
            "SDXL quality", "AI.SDXL.QUALITY", "PROBABILISTIC", 0.5, 0.1
        )
        shift = engine.update_from_evidence(
            "AI.SDXL.QUALITY", 0.9, "USER_SESSION", "quality report"
        )
        prob_shifts.append(shift)
    
    print(f"\n  Axiom attacked 1000 times:")
    print(f"    Total shift: {sum(axiom_shifts):.2e}")
    print(f"    Max single shift: {max(axiom_shifts):.2e}")
    print(f"    Protected: {'YES ✅' if sum(axiom_shifts) < 0.001 else 'NO ❌'}")
    
    print(f"\n  Probabilistic updated 1000 times:")
    print(f"    Total shift: {sum(prob_shifts):.4f}")
    print(f"    Avg shift: {sum(prob_shifts)/len(prob_shifts):.4f}")
    print(f"    Adaptable: {'YES ✅' if sum(prob_shifts) > 0.01 else 'NO ❌'}")
    
    return engine


if __name__ == "__main__":
    demo = demo_probabilistic_hierarchy()
    bench = benchmark_immutable_vs_flexible()
