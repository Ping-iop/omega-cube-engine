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
import hashlib
import numpy as np
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
    
    # Common English stopwords that cause cross-domain collisions
    _STOPWORDS = frozenset({
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'just', 'because', 'but', 'and', 'or', 'if', 'while', 'about',
        'knowledge', 'domain', 'axiom', 'concept', 'instance',
    })
    
    def encode_node(self, content: str, hierarchy: str) -> list[float]:
        """
        Create a semantic vector via feature hashing (hashing trick).
        
        Each token maps to a deterministic position in the vector space.
        Shared tokens between query and document produce similar vectors,
        enabling genuine semantic retrieval via HNSW cosine similarity.
        
        Uses double hashing: h1 → position, h2 → sign (+1/-1).
        This is the standard HashingVectorizer approach (Weinberger 2009).
        """
        key = f"{content[:120]}:{hierarchy}"
        if key in self._basis_cache:
            return self._basis_cache[key]
        
        vec = [0.0] * self.dim
        
        # Tokenize: content (filtered by stopwords)
        tokens = []
        for w in content.lower().split():
            w = w.strip('.,;:!?()[]{}"\'-')
            if len(w) > 1 and w not in self._STOPWORDS:
                tokens.append(w)
        
        # Bigrams: capture word-pair semantics ("transformer attention" ≠ "attention transformer")
        for i in range(len(tokens) - 1):
            tokens.append(tokens[i] + "_" + tokens[i + 1])
        
        # Hierarchy terms get 1.5x weight (moderate domain signal)
        if hierarchy:
            for part in hierarchy.replace('.', ' ').replace('/', ' ').lower().split():
                part = part.strip()
                if len(part) > 1:
                    tokens.append(part)
                    # 0.5x extra via half-contribution trick: add once more only if
                    # we want >1x. For 1.5x we add the token once (base) and rely on
                    # the fact that hierarchy parts are short and distinctive.
                    # Actually just append once — 1x is enough when content is filtered.
        
        # Feature hashing: each token → (position, sign)
        for token in tokens:
            # h1: position in vector
            h1 = int(hashlib.md5(token.encode()).hexdigest(), 16)
            pos = h1 % self.dim
            # h2: sign (+1 or -1) to reduce collision bias
            h2 = int(hashlib.sha1(token.encode()).hexdigest(), 16)
            sign = 1.0 if (h2 % 2 == 0) else -1.0
            vec[pos] += sign
        
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
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
        # Use FFT for circular convolution O(n log n)
        # bind(a,b) = ifft(fft(a) * fft(b))
        v1_arr = np.array(v1)
        v2_arr = np.array(v2)
        result = np.real(np.fft.ifft(np.fft.fft(v1_arr) * np.fft.fft(v2_arr)))
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.tolist()
    
    def unbind(self, bound: list[float], v1: list[float]) -> list[float]:
        """
        Circular correlation: recovers v2 from bind(v1, v2) given v1.
        Uses FFT for O(n log n) performance (matches bind complexity).
        """
        bound_arr = np.array(bound)
        v1_arr = np.array(v1)
        # Circular correlation = ifft(fft(bound) * conj(fft(v1)))
        result = np.real(np.fft.ifft(np.fft.fft(bound_arr) * np.conj(np.fft.fft(v1_arr))))
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.tolist()
    
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
