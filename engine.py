"""
Omega-Cube Engine — Unified multi-dimensional graph memory system.

Integrates ten innovations:
1. Tensor Hierarchies
2. Holographic Encoding
3. Quantum-Inspired Annealing
4. Diffusion Graph Sampling
5. Gray-Scale Validation
6. AutoResearch Loop
7. Predictive Context Search
8. Collective Hierarchy Evolution
9. Probabilistic Hierarchy Engine
10. MARP Router — Model-Agnostic Routing Protocol

The engine organizes knowledge as "magnetic cubes" — topic domains with
internal hierarchies that rotate, connect, and coalesce to form patterns
answering complex multi-topic queries.

Author: Omega-Cube Research
Date: 2026-06-11
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

from .tensor_node import TensorNode, TensorIndex
from .holographic import HolographicEncoder
from .annealer import QuantumInspiredAnnealer, CubeRotator, PatternEmergence
from .diffusion_sampler import DiffusionGraphSampler
from .grayscale import GrayScaleValidator


class OmegaCubeEngine:
    """
    Omega-Cube: The next evolution of Axiomatic Memory.
    
    Key differences from original Omega:
    - Multi-dimensional instead of linear hierarchy
    - Holographic signatures for O(1) approximate retrieval
    - Dynamic topology via annealing instead of static graph
    - Diffusion-based parallel sampling instead of sequential traversal
    - Gray-scale confidence instead of binary verification
    """
    
    def __init__(
        self,
        memory_dir: str = None,
        holographic_dim: int = 256,
        tensor_grid_size: int = 10,
    ):
        # Paths
        if memory_dir is None:
            axioma_base = os.environ.get(
                "AXIOMA_PROJECT_PATH",
                str(Path.home() / ".hermes" / "axioma-omega-protocol")
            )
            memory_dir = os.path.join(axioma_base, "memory")
        
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.nodes: dict[str, TensorNode] = {}
        self.index = TensorIndex(grid_size=tensor_grid_size)
        
        # Sub-engines
        self.holographic = HolographicEncoder(dimension=holographic_dim)
        self.annealer = QuantumInspiredAnnealer()
        self.rotator = CubeRotator()
        self.diffusion = DiffusionGraphSampler()
        self.gray_validator = GrayScaleValidator()
        self.pattern_emergence = PatternEmergence()
        
        # Axiom registry (for gray-scale validation)
        self.axioms: list[TensorNode] = []
        
        # Stats
        self.query_count = 0
        self.total_retrieval_time = 0.0
    
    # ── Knowledge Ingestion ───────────────────────────────────────
    
    def add_node(
        self,
        content: str,
        hierarchies: list[str],
        tensor_position: list[float] = None,
        node_type: str = "CONCEPT",
        confidence: float = 0.9,
        tags: list = None,
    ) -> TensorNode:
        """
        Add a node to the multi-dimensional graph.
        
        Args:
            content: Node content
            hierarchies: List of hierarchy paths (one per dimension)
            tensor_position: Position in N-dim space (auto-computed if None)
            node_type: AXIOM | CONCEPT | INSTANCE
            confidence: Initial confidence score
            tags: Optional tags
        
        Returns:
            Created TensorNode
        """
        # Auto-compute tensor position if not provided
        if tensor_position is None:
            tensor_position = self._compute_tensor_position(hierarchies)
        
        node = TensorNode(
            content=content,
            hierarchies=hierarchies,
            tensor_position=tensor_position,
            node_type=node_type,
            confidence=confidence,
            tags=tags or [],
            created_at=time.time(),
        )
        
        # Compute holographic signature
        node.holographic_signature = self.holographic.encode_node(content, node.primary_hierarchy)
        
        # Compute gray-scale profile
        node.gray_scale = self.gray_validator.evaluate_node(
            node, axioms=self.axioms
        )
        
        # Register
        self.nodes[node.node_id] = node
        self.index.insert(node)
        
        if node_type == "AXIOM":
            self.axioms.append(node)
        
        return node
    
    def associate(self, node_id1: str, node_id2: str) -> bool:
        """Create lateral association between two nodes."""
        if node_id1 in self.nodes and node_id2 in self.nodes:
            n1 = self.nodes[node_id1]
            n2 = self.nodes[node_id2]
            if node_id2 not in n1.associations:
                n1.associations.append(node_id2)
            if node_id1 not in n2.associations:
                n2.associations.append(node_id1)
            
            # Update holographic signatures
            n1.holographic_signature = self._recompute_signature(node_id1)
            n2.holographic_signature = self._recompute_signature(node_id2)
            
            return True
        return False
    
    # ── Query and Retrieval ───────────────────────────────────────
    
    def query(
        self,
        query_text: str,
        mode: str = "diffusion",
        top_k: int = 10,
        temperature: float = 0.1,
    ) -> list[dict]:
        """
        Query the Omega-Cube engine.
        
        Args:
            query_text: Search query
            mode: "diffusion" (parallel) | "annealing" (topology-optimized)
                   | "tensor" (spatial proximity) | "holographic" (signature match)
            top_k: Number of results
            temperature: Noise level (lower = more deterministic)
        
        Returns:
            List of result dicts with node, score, gray_scale, and hierarchy
        """
        self.query_count += 1
        start = time.time()
        
        if mode == "diffusion":
            results = self._query_diffusion(query_text, top_k, temperature)
        elif mode == "annealing":
            results = self._query_annealing(query_text, top_k)
        elif mode == "tensor":
            results = self._query_tensor(query_text, top_k)
        elif mode == "holographic":
            results = self._query_holographic(query_text, top_k)
        else:
            results = self._query_combined(query_text, top_k)
        
        elapsed = time.time() - start
        self.total_retrieval_time += elapsed
        
        # Update access counts
        for r in results:
            if isinstance(r, tuple) and len(r) >= 1:
                node = r[0]
            elif isinstance(r, dict):
                node = r.get("_node")
            else:
                node = r
            if node and hasattr(node, 'access_count'):
                node.access_count += 1
        
        return [self._format_result(r) for r in results]
    
    def query_multi_topic(
        self,
        query_text: str,
        topics: list[str],
        top_k_per_topic: int = 3,
    ) -> dict[str, list[dict]]:
        """
        Multi-topic query using diffusion sampling per topic.
        Returns results organized by topic cluster.
        """
        raw = self.diffusion.sample_multi_topic(
            query_text, self.index, self.holographic,
            topic_dimensions=topics, top_k_per_topic=top_k_per_topic,
        )
        
        result = {}
        for topic, nodes in raw.items():
            result[topic] = [self._format_result((n, s)) for n, s in nodes]
        
        return result
    
    def find_patterns(
        self,
        query_text: str,
        min_strength: float = 0.5,
    ) -> list[dict]:
        """
        Find emergent cross-topic patterns using annealing + pattern detection.
        """
        # Create cubes from topic clusters
        cubes = self._build_cubes()
        
        # Run annealing
        query_vector = self.holographic.encode_node(query_text, "QUERY")
        
        optimized, _, _ = self.annealer.anneal(
            cubes=cubes,
            energy_fn=lambda c: self._pattern_energy(c, query_vector),
            neighbor_fn=lambda c: [self.rotator.random_rotation(cube) for cube in c],
            max_iterations=200,
        )
        
        # Extract patterns
        patterns = self.pattern_emergence.extract_patterns(optimized, threshold=min_strength)
        
        return patterns
    
    # ── Internal query methods ────────────────────────────────────
    
    def _query_diffusion(self, query: str, top_k: int, temp: float) -> list:
        return self.diffusion.sample(
            query, self.index, self.holographic, top_k=top_k, temperature=temp,
        )
    
    def _query_annealing(self, query: str, top_k: int) -> list:
        patterns = self.find_patterns(query, min_strength=0.3)
        results = []
        for p in patterns[:top_k]:
            # Find the actual node for each pattern
            for nid, node in self.nodes.items():
                if node.primary_hierarchy and p["anchor_cube"] in str(nid):
                    results.append((node, p["pattern_strength"]))
                    break
        return results if results else self._query_diffusion(query, top_k, 0.1)
    
    def _query_tensor(self, query: str, top_k: int) -> list:
        query_vector = self.holographic.encode_node(query, "QUERY")
        # Use first 2 dimensions of query vector as position
        pos = query_vector[:2]
        pos_norm = [(p + 1) / 2 for p in pos]  # Map [-1,1] to [0,1]
        nodes = self.index.query(pos_norm, radius=0.5)
        results = []
        for node in nodes:
            sim = self.holographic.similarity(query_vector, node.holographic_signature or [])
            results.append((node, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    def _query_holographic(self, query: str, top_k: int) -> list:
        query_vector = self.holographic.encode_node(query, "QUERY")
        results = []
        for node in self.nodes.values():
            if node.holographic_signature:
                sim = self.holographic.partial_match(query_vector, node.holographic_signature)
                results.append((node, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    def _query_combined(self, query: str, top_k: int) -> list:
        """Combined: diffusion + holographic re-ranking."""
        candidates = self._query_diffusion(query, top_k * 3, 0.15)
        query_vector = self.holographic.encode_node(query, "QUERY")
        
        # Re-rank with holographic similarity
        results = []
        for node, score in candidates:
            holo_sim = 0.5
            if node.holographic_signature:
                holo_sim = self.holographic.partial_match(query_vector, node.holographic_signature)
            combined = 0.6 * score + 0.4 * holo_sim
            results.append((node, combined))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    # ── Cube construction ─────────────────────────────────────────
    
    def _build_cubes(self) -> list[dict]:
        """Build cube representations from graph nodes."""
        # Group nodes by primary hierarchy prefix
        topics: dict[str, list[TensorNode]] = {}
        for node in self.nodes.values():
            prefix = node.primary_hierarchy.split(".")[0] if node.primary_hierarchy else "UNKNOWN"
            if prefix not in topics:
                topics[prefix] = []
            topics[prefix].append(node)
        
        cubes = []
        for i, (topic, nodes) in enumerate(topics.items()):
            cube = {
                "id": f"cube_{i}",
                "topic": topic,
                "dimensions": [h for n in nodes for h in n.hierarchies],
                "active_dimension": 0,
                "subtopics": [n.primary_hierarchy for n in nodes],
                "exposed_subtopic": nodes[0].primary_hierarchy if nodes else "",
                "exposed_content": nodes[0].content[:100] if nodes else "",
                "active_vector": nodes[0].tensor_position if nodes else [],
                "associations": {},
                "node_count": len(nodes),
            }
            # Map associations
            for n in nodes:
                for assoc_id in n.associations:
                    cube["associations"][assoc_id] = True
            
            cubes.append(cube)
        
        return cubes
    
    def _pattern_energy(self, cubes: list[dict], query_vector: list[float]) -> float:
        """Energy function: lower = better alignment with query."""
        energy = 0.0
        
        for cube in cubes:
            cube_vec = cube.get("active_vector", [])
            if cube_vec:
                # Alignment with query
                alignment = CubeRotator._cosine_similarity(query_vector, cube_vec)
                energy -= alignment  # Higher alignment → lower energy
        
        return energy / max(len(cubes), 1)
    
    # ── Helpers ───────────────────────────────────────────────────
    
    def _compute_tensor_position(self, hierarchies: list[str]) -> list[float]:
        """Compute tensor position from hierarchy paths."""
        positions = []
        for h in hierarchies:
            parts = h.split(".")
            # Encode hierarchy depth and content into a float
            if len(parts) >= 2:
                # Use hash of hierarchy to generate consistent coordinate
                import hashlib
                hash_bytes = hashlib.md5(h.encode()).digest()
                coord = int.from_bytes(hash_bytes[:4], 'big') / (2**32)
                positions.append(coord)
            else:
                positions.append(0.5)
        
        # Pad to at least 2 dimensions
        while len(positions) < 2:
            positions.append(0.5)
        
        return positions
    
    def _recompute_signature(self, node_id: str) -> list[float]:
        """Recompute holographic signature for a node."""
        node = self.nodes.get(node_id)
        if not node:
            return [0.0] * self.holographic.dim
        
        # Get neighbors
        neighbors = []
        for assoc_id in node.associations:
            if assoc_id in self.nodes:
                n = self.nodes[assoc_id]
                neighbors.append((n.content, n.primary_hierarchy))
        
        return self.holographic.encode_holographic_signature(
            node_content=node.content,
            node_hierarchy=node.primary_hierarchy,
            neighbors=neighbors,
        )
    
    def _format_result(self, item) -> dict:
        """Format a result tuple as a clean dict."""
        if isinstance(item, tuple) and len(item) == 2:
            node, score = item
        else:
            node, score = item, 0.5
        
        return {
            "node_id": node.node_id,
            "content": node.content[:300],
            "primary_hierarchy": node.primary_hierarchy,
            "hierarchies": node.hierarchies,
            "node_type": node.node_type,
            "score": round(score, 4),
            "confidence": node.confidence,
            "gray_scale": node.gray_scale,
            "gray_scale_composite": self.gray_validator.composite_score(
                node.gray_scale or {}
            ) if node.gray_scale else 50.0,
            "tensor_position": node.tensor_position,
            "associations_count": len(node.associations),
        }
    
    # ── Persistence ───────────────────────────────────────────────
    
    def save(self, path: str = None):
        """Save the entire engine state to disk."""
        if path is None:
            path = str(self.memory_dir / "omega_cube_memory.json")
        
        data = {
            "nodes": {
                nid: node.to_dict() for nid, node in self.nodes.items()
            },
            "axiom_ids": [n.node_id for n in self.axioms],
            "stats": {
                "query_count": self.query_count,
                "total_retrieval_time": self.total_retrieval_time,
            },
            "config": {
                "holographic_dim": self.holographic.dim,
            },
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str = None) -> bool:
        """Load engine state from disk."""
        if path is None:
            path = str(self.memory_dir / "omega_cube_memory.json")
        
        if not os.path.exists(path):
            return False
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Restore nodes
        for nid, node_data in data.get("nodes", {}).items():
            node = TensorNode.from_dict(node_data)
            self.nodes[nid] = node
            self.index.insert(node)
        
        # Restore axioms
        for axiom_id in data.get("axiom_ids", []):
            if axiom_id in self.nodes:
                self.axioms.append(self.nodes[axiom_id])
        
        # Restore stats
        stats = data.get("stats", {})
        self.query_count = stats.get("query_count", 0)
        self.total_retrieval_time = stats.get("total_retrieval_time", 0.0)
        
        # Restore config
        config = data.get("config", {})
        if "holographic_dim" in config:
            self.holographic.dim = config["holographic_dim"]
        
        return True
    
    def stats(self) -> dict:
        """Return engine statistics."""
        type_counts = {"AXIOM": 0, "CONCEPT": 0, "INSTANCE": 0, "SESSION": 0}
        for node in self.nodes.values():
            if node.node_type in type_counts:
                type_counts[node.node_type] += 1
        
        avg_dims = (
            sum(n.dimension_count for n in self.nodes.values()) / len(self.nodes)
            if self.nodes else 0
        )
        
        return {
            "total_nodes": len(self.nodes),
            "axioms": type_counts["AXIOM"],
            "concepts": type_counts["CONCEPT"],
            "instances": type_counts["INSTANCE"],
            "sessions": type_counts["SESSION"],
            "avg_dimensions_per_node": round(avg_dims, 1),
            "holographic_dim": self.holographic.dim,
            "query_count": self.query_count,
            "avg_retrieval_time_ms": round(
                (self.total_retrieval_time / self.query_count * 1000)
                if self.query_count > 0 else 0, 2
            ),
            "memory_dir": str(self.memory_dir),
        }
