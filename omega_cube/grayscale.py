"""
GrayScaleValidator — Multi-bit truth assessment for graph nodes.

H-Bit inspired: instead of binary 0/1 verification, each node carries
a multi-bit "gray scale" that encodes degrees of truth across dimensions.

In H-Bit, security is embedded even if only part of a file is checked.
Similarly, GrayScaleValidator assesses partial evidence and produces
confidence scores with uncertainty quantification.

A node scoring 75% on "factuality" and 90% on "relevance" is treated
differently than one scoring 90% on both — the gray scale preserves
the nuance that binary classification loses.

Author: Omega-Cube Research
Date: 2026-06-11
"""

import hashlib
from typing import Optional


class GrayScaleValidator:
    """
    Multi-bit truth assessment system for graph nodes.
    
    Each node is evaluated along multiple truth dimensions:
    - factuality: grounded in verified axioms
    - relevance: connected to query context
    - recency: temporal freshness
    - coherence: consistency with related nodes
    - provenance: traceability to source
    
    The result is a "gray scale" score per dimension, not a binary label.
    """
    
    DIMENSIONS = [
        "factuality",    # Anchored to axioms
        "relevance",     # Query alignment
        "recency",       # Temporal freshness
        "coherence",     # Internal consistency
        "provenance",    # Source traceability
        "specificity",   # Detail granularity
    ]
    
    def __init__(self, default_scale: int = 50):
        """
        Args:
            default_scale: Default gray value (0-100) for unevaluated dimensions.
                          50 = neutral (neither true nor false).
        """
        self.default_scale = default_scale
    
    def evaluate_node(
        self,
        node,
        query: str = "",
        axioms: list = None,
        related_nodes: list = None,
    ) -> dict[str, float]:
        """
        Evaluate a node across all dimensions, producing a gray-scale profile.
        
        Returns:
            Dict mapping dimension → gray_score (0-100)
        """
        axioms = axioms or []
        related_nodes = related_nodes or []
        
        return {
            "factuality": self._assess_factuality(node, axioms),
            "relevance": self._assess_relevance(node, query),
            "recency": self._assess_recency(node),
            "coherence": self._assess_coherence(node, related_nodes),
            "provenance": self._assess_provenance(node),
            "specificity": self._assess_specificity(node),
        }
    
    def composite_score(self, gray_profile: dict[str, float], weights: dict[str, float] = None) -> float:
        """
        Combine gray-scale dimensions into a single confidence score (0-100).
        
        Args:
            gray_profile: Output from evaluate_node
            weights: Per-dimension importance weights
        
        Returns:
            Weighted composite score 0-100
        """
        if weights is None:
            weights = {
                "factuality": 0.35,
                "relevance": 0.25,
                "recency": 0.10,
                "coherence": 0.15,
                "provenance": 0.10,
                "specificity": 0.05,
            }
        
        total = 0.0
        total_weight = 0.0
        for dim, score in gray_profile.items():
            w = weights.get(dim, 0.0)
            total += score * w
            total_weight += w
        
        return total / total_weight if total_weight > 0 else 50.0
    
    def partial_evidence_score(self, gray_profile: dict[str, float], available_dimensions: list[str]) -> float:
        """
        H-Bit style: compute confidence even when only partial evidence is available.
        
        The key insight: you don't need all dimensions to make a useful assessment.
        Even checking 2 of 6 dimensions provides actionable confidence.
        """
        if not available_dimensions:
            return 50.0
        
        scores = [gray_profile.get(d, self.default_scale) for d in available_dimensions]
        return sum(scores) / len(scores)
    
    def verify_against_axioms(
        self,
        node,
        axioms: list,
        threshold: float = 60.0,
    ) -> tuple[bool, float, str]:
        """
        Verify a node's factual grounding against known axioms.
        
        Returns:
            (is_verified, confidence, explanation)
        """
        if not axioms:
            return False, 0.0, "No axioms available for verification"
        
        matches = []
        node_text = (node.content or "").lower()
        
        for axiom in axioms:
            axiom_text = (axiom.content if hasattr(axiom, 'content') else str(axiom)).lower()
            # Check keyword overlap with axiom
            overlap = self._keyword_overlap(node_text, axiom_text)
            if overlap > 0.3:
                matches.append(overlap)
        
        if not matches:
            return False, 0.0, "No axiom matches found"
        
        confidence = (sum(matches) / len(matches)) * 100
        verified = confidence >= threshold
        
        return verified, confidence, f"Matched {len(matches)} axioms (avg: {confidence:.1f}%)"
    
    def compute_gray_scale_hash(self, gray_profile: dict[str, float]) -> str:
        """
        Create a hash fingerprint of the gray-scale profile.
        
        Enables quick comparison without storing the full profile.
        """
        # Quantize to 5-bit precision (32 levels per dimension)
        quantized = [int(v / 100 * 31) & 0x1F for v in gray_profile.values()]
        # Pack into hash
        packed = sum(q << (5 * i) for i, q in enumerate(quantized))
        return hashlib.sha256(str(packed).encode()).hexdigest()[:12]
    
    # ── Individual dimension assessors ───────────────────────────
    
    def _assess_factuality(self, node, axioms: list) -> float:
        if not axioms:
            return self.default_scale
        verified, confidence, _ = self.verify_against_axioms(node, axioms)
        return confidence if verified else max(10, confidence * 0.5)
    
    def _assess_relevance(self, node, query: str) -> float:
        if not query:
            return self.default_scale
        overlap = self._keyword_overlap(
            (node.content or "").lower(),
            query.lower(),
        )
        return overlap * 100
    
    def _assess_recency(self, node) -> float:
        """Score based on creation/access recency."""
        import time
        now = time.time()
        created = getattr(node, 'created_at', 0)
        if created == 0:
            return self.default_scale
        # Decay: half-life of 30 days
        age_days = (now - created) / 86400
        return 100 * (0.5 ** (age_days / 30))
    
    def _assess_coherence(self, node, related_nodes: list) -> float:
        if not related_nodes:
            return self.default_scale
        # Average content similarity with related nodes
        sims = []
        for rn in related_nodes:
            rn_content = (rn.content if hasattr(rn, 'content') else str(rn)).lower()
            sim = self._keyword_overlap(
                (node.content or "").lower(),
                rn_content,
            )
            sims.append(sim)
        return (sum(sims) / len(sims)) * 100 if sims else self.default_scale
    
    def _assess_provenance(self, node) -> float:
        """Score based on traceability to source."""
        tags = getattr(node, 'tags', []) or []
        if "axiom" in tags:
            return 95.0
        if "verified" in tags:
            return 85.0
        if "source" in tags or "citation" in tags:
            return 70.0
        return 40.0  # Unknown provenance
    
    def _assess_specificity(self, node) -> float:
        """Score based on content detail level."""
        content = node.content or ""
        # More specific content tends to be longer and more detailed
        length_score = min(100, len(content) / 5)
        # Check for numbers, dates, paths (specificity indicators)
        import re
        specificity_indicators = len(re.findall(
            r'\d+|[A-Z]:\\|/\w+|config\.\w+', content
        ))
        detail_score = min(100, specificity_indicators * 20)
        return (length_score + detail_score) / 2
    
    @staticmethod
    def _keyword_overlap(text1: str, text2: str) -> float:
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)
