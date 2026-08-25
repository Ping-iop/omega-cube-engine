"""
ConflictDetector v2 - Enhanced semantic contradiction detection.

Improvements over v1:
- Smart tokenization (handles underscores, compound words)
- TF-IDF based cosine similarity for better semantic capture  
- Expanded contradictory pairs with contextual synonyms + vendor exclusions
- Contradiction-first detection: explicit patterns bypass similarity filter
- Hierarchical co-occurrence as confidence signal

Author: Bit/Hermes  
Date: 2026-06-29
"""

from typing import List, Dict, Any, Tuple
import math
from collections import Counter


class ConflictDetectorV2:
    """Enhanced conflict detector with semantic awareness and contradiction-first logic."""

    # Contradictory pairs: (positive_signal, negative_signal)
    CONTRADICTORY_PATTERNS = [
        # Direct antonyms
        ("selected", "rejected"),
        ("approved", "denied"),
        ("yes", "no"),
        ("true", "false"),
        ("positive", "negative"),
        ("increase", "decrease"),
        ("expand", "contract"),
        ("adopt", "abandon"),
        
        # Decision outcome synonyms
        ("chosen", "discarded"),
        ("preferred", "rejected"),
        ("favored", "disfavored"),
        ("recommended", "advised_against"),
        ("implemented", "deferred"),
        ("pursued", "abandoned"),
        ("committed_to", "backed_off_from"),
        ("moved_forward", "halted"),
        
        # Action polarity
        ("go_ahead", "no_go"),
        ("proceed", "pause"),
        ("accept", "decline"),
        ("include", "exclude"),
        ("enable", "disable"),
        ("proactive", "reactive"),
        ("internal", "outsourced"),
        ("build", "buy"),
    ]

    # Vendor/technology mutual exclusion pairs (choosing one implies rejecting the other)
    VENDOR_EXCLUSIONS = [
        ("aws", "azure"),
        ("aws", "gcp"),
        ("azure", "gcp"),
        ("kubernetes", "docker_compose"),
        ("monolith", "microservices"),
    ]

    def __init__(self, similarity_threshold: float = 0.3):
        """
        Args:
            similarity_threshold: Minimum composite score to flag as MEDIUM severity.
                Lowered from 0.85 (v1) to 0.3 because contradiction-first detection
                doesn't rely on this for initial filtering.
        """
        self.similarity_threshold = similarity_threshold
        
    def detect_conflicts(
        self,
        new_node,
        existing_nodes: List,
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts using contradiction-first approach.
        
        Algorithm:
        1. For each existing node, check if explicit contradictory patterns exist
        2. If found, calculate composite confidence based on hierarchy overlap
           and semantic similarity
        3. Only report conflicts where composite score exceeds threshold
        """
        conflicts = []
        
        for existing in existing_nodes:
            # Step 1: Check for explicit contradictions (bypasses similarity filter)
            is_contradictory, contradiction_info = self._detect_explicit_contradiction(
                new_node.content, existing.content
            )
            
            if not is_contradictory:
                continue
            
            # Step 2: Calculate composite confidence score
            text_sim = self._calculate_semantic_similarity(
                new_node.content, existing.content
            )
            hierarchy_sim = self._calculate_hierarchy_overlap(
                getattr(new_node, 'hierarchies', []),
                getattr(existing, 'hierarchies', [])
            )
            
            # Composite: weighted combination where hierarchy overlap boosts confidence
            composite_score = 0.4 * text_sim + 0.6 * hierarchy_sim
            
            # Step 3: Determine severity and report
            if composite_score >= self.similarity_threshold:
                severity = "HIGH" if composite_score > 0.7 else "MEDIUM"
                
                conflicts.append({
                    "node_a": getattr(existing, 'node_id', str(id(existing))),
                    "node_b": getattr(new_node, 'node_id', str(id(new_node))),
                    "type": contradiction_info["type"],
                    "severity": severity,
                    "similarity_score": round(composite_score, 3),
                    "text_similarity": round(text_sim, 3),
                    "hierarchy_overlap": round(hierarchy_sim, 3),
                    "contradiction_pairs": contradiction_info["pairs"],
                    "description": (
                        f"Nodes {getattr(existing, 'node_id', '?')} and "
                        f"{getattr(new_node, 'node_id', '?')} appear contradictory: "
                        f"'{new_node.content}' vs '{existing.content}'"
                    ),
                })
        
        return conflicts
    
    def _tokenize_smart(self, text: str) -> List[str]:
        """Smart tokenizer handling underscores and compound words."""
        normalized = text.replace("_", " ")
        tokens = normalized.lower().split()
        # Keep tokens >= 2 chars (filters out noise like single letters)
        return [t for t in tokens if len(t) >= 2]
    
    def _calculate_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """TF-IDF based cosine similarity with token overlap boost."""
        tokens_a = self._tokenize_smart(text_a)
        tokens_b = self._tokenize_smart(text_b)
        
        if not tokens_a or not tokens_b:
            return 0.0
        
        vocab = set(tokens_a) | set(tokens_b)
        tf_a = Counter(tokens_a)
        tf_b = Counter(tokens_b)
        
        # Normalize term frequencies
        max_tf_a = max(tf_a.values()) if tf_a else 1
        max_tf_b = max(tf_b.values()) if tf_b else 1
        
        norm_tf_a = {t: c / max_tf_a for t, c in tf_a.items()}
        norm_tf_b = {t: c / max_tf_b for t, c in tf_b.items()}
        
        # Simplified IDF (assuming equal document weight)
        idf = {}
        for term in vocab:
            doc_count = sum(1 for txt in [text_a, text_b] 
                          if term in self._tokenize_smart(txt))
            idf[term] = 1.0 + math.log(2 / (1 + doc_count))
        
        # Cosine similarity on TF-IDF vectors
        numerator = sum(
            norm_tf_a.get(t, 0) * idf[t] * norm_tf_b.get(t, 0) * idf[t]
            for t in vocab
        )
        denom_a = math.sqrt(sum((norm_tf_a.get(t, 0) * idf[t]) ** 2 for t in vocab))
        denom_b = math.sqrt(sum((norm_tf_b.get(t, 0) * idf[t]) ** 2 for t in vocab))
        
        if denom_a == 0 or denom_b == 0:
            cosine_sim = 0.0
        else:
            cosine_sim = numerator / (denom_a * denom_b)
        
        # Token overlap ratio (Jaccard on token sets)
        shared = set(tokens_a) & set(tokens_b)
        token_overlap = len(shared) / max(len(set(tokens_a)), len(set(tokens_b)))
        
        # Combine: 60% TF-IDF cosine + 40% token overlap
        return 0.6 * cosine_sim + 0.4 * token_overlap
    
    def _calculate_hierarchy_overlap(self, hier_a: List[str], hier_b: List[str]) -> float:
        """Jaccard similarity on hierarchy paths."""
        if not hier_a or not hier_b:
            return 0.0
        
        set_a = set(hier_a)
        set_b = set(hier_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_explicit_contradiction(self, text_a: str, text_b: str) -> Tuple[bool, Dict]:
        """
        Detect explicit contradictions using pattern matching.
        Returns (is_contradictory, contradiction_info).
        
        Contradiction-first approach: this check runs BEFORE similarity filtering,
        because explicit contradictory signals should not be suppressed by low
        text similarity (e.g., "approved_budget" vs "denied_budget_request").
        """
        tokens_a = self._tokenize_smart(text_a)
        tokens_b = self._tokenize_smart(text_b)
        
        # Full-text lowercase for prefix matching
        lower_a = text_a.lower()
        lower_b = text_b.lower()
        
        matched_pairs = []
        
        # Check direct contradictory patterns
        for pat_a, pat_b in self.CONTRADICTORY_PATTERNS:
            a_match = any(pat_a in t or pat_a == t for t in tokens_a) or pat_a in lower_a
            b_match = any(pat_b in t or pat_b == t for t in tokens_b) or pat_b in lower_b
            
            if a_match and b_match:
                matched_pairs.append((pat_a, pat_b))
        
        # Check vendor exclusions (choosing one implies rejecting the other)
        for v_a, v_b in self.VENDOR_EXCLUSIONS:
            a_has = any(v_a in t.lower() for t in tokens_a) or v_a in lower_a
            b_has = any(v_b in t.lower() for t in tokens_b) or v_b in lower_b
            
            if a_has and b_has:
                matched_pairs.append((v_a, v_b))
        
        if not matched_pairs:
            return False, {}
        
        # Classify contradiction type
        is_vendor = any(
            pa.lower() in ["aws", "azure", "gcp"] or 
            pb.lower() in ["aws", "azure", "gcp"]
            for pa, pb in matched_pairs
        )
        is_decision = any(
            pa.lower() in ["selected", "approved", "chosen", "preferred", "adopted"] and
            pb.lower() in ["rejected", "denied", "discarded", "abandoned"]
            for pa, pb in matched_pairs
        )
        
        if is_vendor:
            ctype = "VENDOR_EXCLUSION"
        elif is_decision:
            ctype = "DECISION_OUTCOME_CONFLICT"
        else:
            ctype = "EXPLICIT_CONTRADICTION"
        
        return True, {
            "type": ctype,
            "pairs": matched_pairs,
        }


# Backward compatibility alias
ConflictDetector = ConflictDetectorV2
