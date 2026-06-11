"""
HolographicEncoder — Compressed distributed representations for graph nodes.

Each node encodes information about its entire neighborhood in a fixed-size
vector (the "holographic signature"), enabling:
- O(1) approximate retrieval of structural context without graph traversal
- Pattern completion: partial query → full neighborhood reconstruction
- Interference-resistant encoding via circular convolution

Inspired by:
- Smolensky's tensor product representations (1990)
- Plate's Holographic Reduced Representations (1995)
- H-Bit multi-scale truth assessment (gray-scale encoding)

Author: Omega-Cube Research
Date: 2026-06-11
"""

import math
import random
from typing import Optional


class HolographicEncoder:
    """
    Encodes graph nodes into fixed-dimensional holographic vectors.
    
    Uses circular convolution to bind nodes with their structural context,
    creating a distributed representation where every element contains
    partial information about the whole neighborhood.
    
    Key property: the signature of a node can be decoded to recover
    information about its neighbors without traversing the graph.
    """
    
    def __init__(self, dimension: int = 256, seed: int = 42):
        self.dim = dimension
        self.seed = seed
        
        # Pre-compute basis vectors for encoding
        # Uses random phase vectors in frequency domain for clean circular convolution
        random.seed(seed)
        self._basis_cache: dict[str, list[float]] = {}
    
    def encode_node(self, content: str, hierarchy: str) -> list[float]:
        """Create a base vector for a node from its content and hierarchy."""
        key = f"{content[:80]}:{hierarchy}"
        if key in self._basis_cache:
            return self._basis_cache[key]
        
        # Deterministic pseudo-random vector from content seed
        seed_val = hash(key) % (2**31)
        rng = random.Random(seed_val)
        
        # Phase vector (frequency domain encoding)
        vec = [rng.uniform(-1, 1) for _ in range(self.dim)]
        
        # Normalize
        norm = math.sqrt(sum(v**2 for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        
        self._basis_cache[key] = vec
        return vec
    
    def bind(self, v1: list[float], v2: list[float]) -> list[float]:
        """
        Circular convolution: the fundamental binding operation.
        
        Binds two vectors to create a new vector that represents 
        their association. Approximate inverse of unbind.
        """
        result = [0.0] * self.dim
        for i in range(self.dim):
            for j in range(self.dim):
                result[(i + j) % self.dim] += v1[i] * v2[j]
        # Normalize
        norm = math.sqrt(sum(v**2 for v in result))
        if norm > 0:
            result = [v / norm for v in result]
        return result
    
    def unbind(self, bound: list[float], v1: list[float]) -> list[float]:
        """
        Circular correlation: recovers v2 from bind(v1, v2) given v1.
        Uses the approximate inverse property of circular convolution.
        """
        result = [0.0] * self.dim
        v1_inv = [v1[-i] if i > 0 else v1[0] for i in range(self.dim)]
        for i in range(self.dim):
            for j in range(self.dim):
                result[(i + j) % self.dim] += bound[i] * v1_inv[j]
        norm = math.sqrt(sum(v**2 for v in result))
        if norm > 0:
            result = [v / norm for v in result]
        return result
    
    def bundle(self, vectors: list[list[float]]) -> list[float]:
        """Superposition: combine multiple vectors into one."""
        if not vectors:
            return [0.0] * self.dim
        result = [sum(vec[i] for vec in vectors) / len(vectors) for i in range(self.dim)]
        norm = math.sqrt(sum(v**2 for v in result))
        if norm > 0:
            result = [v / norm for v in result]
        return result
    
    def encode_holographic_signature(
        self,
        node_content: str,
        node_hierarchy: str,
        parent_content: Optional[str] = None,
        parent_hierarchy: Optional[str] = None,
        children: list[tuple[str, str]] = None,
        neighbors: list[tuple[str, str]] = None,
    ) -> list[float]:
        """
        Create a holographic signature encoding the full structural context.
        
        The signature bundles:
        - Self vector (node identity)
        - Parent binding (hierarchical context)
        - Children bundle (downward context)
        - Neighbors bundle (lateral context)
        """
        children = children or []
        neighbors = neighbors or []
        
        vectors = []
        
        # Self
        self_vec = self.encode_node(node_content, node_hierarchy)
        vectors.append(self_vec)
        
        # Parent binding (encode hierarchy relationship)
        if parent_content:
            parent_vec = self.encode_node(parent_content, parent_hierarchy or "")
            bound = self.bind(self_vec, parent_vec)
            vectors.append(bound)
        
        # Children bundle
        if children:
            child_vecs = [
                self.bind(self_vec, self.encode_node(c[0], c[1]))
                for c in children
            ]
            vectors.append(self.bundle(child_vecs))
        
        # Neighbors bundle
        if neighbors:
            neighbor_vecs = [
                self.encode_node(n[0], n[1])
                for n in neighbors
            ]
            vectors.append(self.bundle(neighbor_vecs))
        
        return self.bundle(vectors)
    
    def similarity(self, v1: list[float], v2: list[float]) -> float:
        """Cosine similarity between two holographic vectors."""
        if len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(v**2 for v in v1))
        norm2 = math.sqrt(sum(v**2 for v in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return max(0.0, min(1.0, (dot / (norm1 * norm2) + 1) / 2))
    
    def partial_match(self, query_vec: list[float], signature: list[float]) -> float:
        """
        Check if query_vec is contained within the holographic signature.
        Uses cosine similarity in the vector space.
        """
        return self.similarity(query_vec, signature)
