"""
Axion-Cube Vector Index — HNSW-based ANN search via USearch.

Layer 2 of the scalable search architecture:
  Layer 0: MARP Router (domain routing, O(1))
  Layer 1: Shard Partitioning (domain-scoped node sets)
  Layer 2: HNSW Index (this module, O(log n) ANN)
  Layer 3: Multi-signal Re-ranking (keyword + content + depth)
  Layer 4: Gray-scale Validation (existing)

USearch benchmarks (64-core AWS Graviton 3):
  f32 x256: 131,654 QPS, 99.3% recall@1
  i8  x256: 274,653 QPS, 98.9% recall@1

Memory per vector (256 dims):
  f32: 1,024 B | i8: 256 B | f16: 512 B | b1: 32 B

Author: Axion Research
Date: 2026-07-26
"""

import numpy as np
import time
from typing import Optional

try:
    from usearch.index import Index, MetricKind, ScalarKind
    USEARCH_AVAILABLE = True
except ImportError:
    USEARCH_AVAILABLE = False


class HNSWVectorIndex:
    """
    HNSW approximate nearest neighbor index for holographic signatures.
    
    Replaces O(n) linear scan with O(log n) ANN search.
    Supports incremental add (no full rebuild needed).
    
    Quantization options:
      - 'f32': full precision, 1024 B/vector
      - 'i8':  8-bit integer, 256 B/vector, ~0.4% recall loss
      - 'f16': half precision, 512 B/vector, ~0.9% recall loss
    """
    
    def __init__(
        self,
        dimension: int = 256,
        metric: str = "cos",
        quantization: str = "i8",
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ):
        self.dimension = dimension
        self.metric = metric
        self.quantization = quantization
        self.connectivity = connectivity
        self.expansion_add = expansion_add
        self.expansion_search = expansion_search
        
        # Map node_id -> internal integer key
        self._id_map: dict[str, int] = {}
        self._reverse_map: dict[int, str] = {}
        self._next_key = 0
        
        # Stats
        self.total_searches = 0
        self.total_search_time = 0.0
        self.total_adds = 0
        
        if USEARCH_AVAILABLE:
            dtype_map = {
                "f32": ScalarKind.F32,
                "i8": ScalarKind.I8,
                "f16": ScalarKind.F16,
            }
            metric_map = {
                "cos": MetricKind.Cos,
                "l2sq": MetricKind.L2sq,
                "ip": MetricKind.IP,
                "hamming": MetricKind.Hamming,
            }
            self._index = Index(
                ndim=dimension,
                metric=metric_map.get(metric, MetricKind.Cos),
                dtype=dtype_map.get(quantization, ScalarKind.I8),
                connectivity=connectivity,
                expansion_add=expansion_add,
                expansion_search=expansion_search,
            )
        else:
            self._index = None
    
    @property
    def is_available(self) -> bool:
        return USEARCH_AVAILABLE and self._index is not None
    
    @property
    def size(self) -> int:
        return len(self._id_map)
    
    def add(self, node_id: str, vector: list[float]) -> bool:
        """Add a vector to the index. O(log n) amortized."""
        if not self.is_available:
            return False
        
        if node_id in self._id_map:
            # Update: remove old, add new
            old_key = self._id_map[node_id]
            self._index.remove(old_key)
            del self._reverse_map[old_key]
        
        key = self._next_key
        self._next_key += 1
        
        vec = np.array(vector, dtype=np.float32)
        self._index.add(key, vec)
        
        self._id_map[node_id] = key
        self._reverse_map[key] = node_id
        self.total_adds += 1
        return True
    
    def add_batch(self, items: list[tuple[str, list[float]]]) -> int:
        """Batch add for initial index build. Returns count added."""
        if not self.is_available:
            return 0
        
        count = 0
        for node_id, vector in items:
            if self.add(node_id, vector):
                count += 1
        return count
    
    def search(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """
        ANN search: returns [(node_id, similarity_score), ...]
        
        O(log n) instead of O(n) linear scan.
        """
        if not self.is_available or self.size == 0:
            return []
        
        start = time.time()
        
        vec = np.array(query_vector, dtype=np.float32)
        k = min(top_k, self.size)
        
        results = self._index.search(vec, k)
        
        # USearch returns distances; convert to similarity
        output = []
        keys = results.keys.tolist() if hasattr(results.keys, 'tolist') else list(results.keys)
        distances = results.distances.tolist() if hasattr(results.distances, 'tolist') else list(results.distances)
        
        for key, dist in zip(keys, distances):
            node_id = self._reverse_map.get(key)
            if node_id:
                # Cosine distance -> similarity: sim = 1 - dist
                # L2 distance -> similarity: sim = 1 / (1 + dist)
                if self.metric == "cos":
                    sim = max(0.0, 1.0 - dist)
                else:
                    sim = 1.0 / (1.0 + dist)
                output.append((node_id, sim))
        
        elapsed = time.time() - start
        self.total_searches += 1
        self.total_search_time += elapsed
        
        return output
    
    def remove(self, node_id: str) -> bool:
        """Remove a vector from the index."""
        if not self.is_available or node_id not in self._id_map:
            return False
        key = self._id_map[node_id]
        self._index.remove(key)
        del self._id_map[node_id]
        del self._reverse_map[key]
        return True
    
    def rebuild(self, items: list[tuple[str, list[float]]]):
        """Full rebuild from scratch. Use after bulk changes."""
        if not self.is_available:
            return
        self._index.clear()
        self._id_map.clear()
        self._reverse_map.clear()
        self._next_key = 0
        self.add_batch(items)
    
    def stats(self) -> dict:
        avg_ms = (
            (self.total_search_time / self.total_searches * 1000)
            if self.total_searches else 0
        )
        return {
            "engine": "usearch" if self.is_available else "unavailable",
            "vectors": self.size,
            "dimension": self.dimension,
            "quantization": self.quantization,
            "metric": self.metric,
            "connectivity": self.connectivity,
            "total_searches": self.total_searches,
            "total_adds": self.total_adds,
            "avg_search_ms": round(avg_ms, 3),
        }
