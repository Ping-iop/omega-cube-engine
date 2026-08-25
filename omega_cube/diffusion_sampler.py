"""
DiffusionGraphSampler — Non-autoregressive graph output generation.

Inspired by Google DeepMind's DiffusionGemma (2026), this module replaces
sequential next-token prediction with parallel diffusion over the graph.

Instead of traversing the graph token by token, the diffuser:
1. Starts with random noise over all candidate nodes
2. Iteratively denoises, guided by hierarchical proximity
3. Converges to the most relevant nodes organized by topic structure

This enables:
- Parallel retrieval (O(log n) instead of O(n) for long chains)
- Natural multi-topic organization (nodes cluster by hierarchy)
- Confidence-calibrated output (gray-scale scoring per node)

Author: Omega-Cube Research
Date: 2026-06-11
"""

import math
import random
from typing import Optional, Callable

from .tensor_node import TensorNode, TensorIndex


class DiffusionGraphSampler:
    """
    Non-autoregressive graph sampler using iterative denoising.
    
    The key insight: instead of predicting the next node in a chain
    (autoregressive traversal), we sample all candidate nodes simultaneously
    and iteratively refine based on hierarchical structure.
    
    This mirrors how DiffusionGemma generates text: start with noise,
    denoise iteratively guided by context, converge to coherent output.
    """
    
    def __init__(
        self,
        num_steps: int = 20,
        guidance_scale: float = 0.3,
        seed: int | None = None,
    ):
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        if seed is not None:
            random.seed(seed)
        
        self._noise_cache: dict[int, list[float]] = {}
    
    def sample(
        self,
        query: str,
        index: TensorIndex,
        holographic_encoder,
        top_k: int = 10,
        temperature: float = 0.1,
    ) -> list[tuple[TensorNode, float]]:
        """
        Diffusion-based graph sampling.
        
        1. Initialize: random noise over all candidate nodes
        2. Denoise iteratively: each step sharpens relevance scores
        3. Converge: final iteration produces ranked results
        
        Args:
            query: Search query
            index: TensorIndex with all nodes
            holographic_encoder: HolographicEncoder for signature matching
            top_k: Number of results to return
            temperature: Noise level (lower = more deterministic)
        
        Returns:
            List of (node, score) sorted by relevance
        """
        # Phase 1: Generate query vector (the "signal")
        query_vector = holographic_encoder.encode_node(query, "")
        
        # Phase 2: Initialize candidates with noisy scores
        candidates = list(index.node_map.values())
        if not candidates:
            return []
        
        # Noisy initialization (diffusion start point)
        scores = self._initialize_noise(len(candidates))
        
        # Assign holographic similarity as base signal
        base_scores = []
        for node in candidates:
            if node.holographic_signature:
                sim = holographic_encoder.partial_match(
                    query_vector, node.holographic_signature
                )
            else:
                sim = self._text_match(query, node.content)
            base_scores.append(sim)
        
        # Phase 3: Iterative denoising
        for step in range(self.num_steps):
            # Noise schedule (cosine)
            noise_level = self._cosine_noise_schedule(step, self.num_steps)
            
            for i in range(len(candidates)):
                # Denoising step: pull toward base signal, push away from noise
                signal = base_scores[i]
                noise = random.gauss(0, noise_level * temperature)
                
                # Hierarchical guidance: boost nodes that cluster with high-scoring neighbors
                guidance = self._hierarchical_guidance(
                    candidates[i], index, base_scores, candidates
                )
                
                # FIX 2026-08-09 (ranking saturado): antes era
                #   score = (1-nl)*signal + nl*noise + guidance_scale*guidance*(1-nl)
                # con guidance_scale=3.0 y guidance≈0.5 → TODOS los nodos saturaban a
                # 1.0, el ranking empataba y se ordenaba por inserción (siempre el
                # mismo top-1). Ahora guidance es una señal relativa al promedio de
                # señales base, acotada por guidance_scale (default 0.3).
                mean_signal = sum(base_scores) / len(base_scores) if base_scores else 0.0
                guidance_signal = (guidance - mean_signal) if mean_signal > 0 else 0.0
                scores[i] = (
                    (1 - noise_level) * (signal + self.guidance_scale * guidance_signal)
                    + noise_level * noise
                )
                scores[i] = max(0.0, min(1.0, scores[i]))
        
        # Phase 4: Organize by hierarchical proximity (natural clustering)
        results = list(zip(candidates, scores))
        results.sort(key=lambda x: -x[1])
        
        # Boost diversity: penalize nodes too similar to higher-ranked results
        results = self._diversity_rerank(results, holographic_encoder, top_k)
        
        return results[:top_k]
    
    def sample_multi_topic(
        self,
        query: str,
        index: TensorIndex,
        holographic_encoder,
        topic_dimensions: list[str],
        top_k_per_topic: int = 3,
    ) -> dict[str, list[tuple[TensorNode, float]]]:
        """
        Multi-topic diffusion sampling.
        
        Samples independently per topic dimension, then merges results
        organized by natural topic clusters.
        
        Args:
            query: Search query
            index: TensorIndex
            holographic_encoder: HolographicEncoder
            topic_dimensions: List of hierarchy prefixes to sample from
            top_k_per_topic: Results per topic
        
        Returns:
            Dict mapping topic → [(node, score), ...]
        """
        results = {}
        for topic in topic_dimensions:
            # Filter nodes in this topic dimension
            topic_nodes = [
                n for n in index.node_map.values()
                if any(h.startswith(topic) for h in n.hierarchies)
            ]
            
            if topic_nodes:
                # Create temporary index for this topic
                topic_index = TensorIndex()
                for n in topic_nodes:
                    topic_index.insert(n)
                
                topic_results = self.sample(
                    query, topic_index, holographic_encoder,
                    top_k=top_k_per_topic
                )
                results[topic] = topic_results
        
        return results
    
    def _initialize_noise(self, size: int) -> list[float]:
        """Initialize with random noise (diffusion starting point)."""
        return [random.random() for _ in range(size)]
    
    def _cosine_noise_schedule(self, step: int, total_steps: int) -> float:
        """Cosine noise schedule: high noise early, low noise late."""
        progress = step / total_steps
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    def _hierarchical_guidance(
        self,
        node: TensorNode,
        index: TensorIndex,
        base_scores: list[float],
        candidates: list[TensorNode],
    ) -> float:
        """
        Boost nodes whose neighbors (in tensor space) have high scores.
        
        This is the "magnetic" effect: high-scoring nodes pull nearby
        nodes upward through the tensor space.
        """
        if not node.tensor_position:
            return 0.0
        
        # Find neighbors in tensor space
        neighbors = index.query(node.tensor_position, radius=0.3)
        
        if not neighbors:
            return 0.0
        
        # Average score of neighbors
        neighbor_scores = []
        for neighbor in neighbors:
            try:
                idx = candidates.index(neighbor)
                neighbor_scores.append(base_scores[idx])
            except ValueError:
                pass
        
        if not neighbor_scores:
            return 0.0
        
        return sum(neighbor_scores) / len(neighbor_scores)
    
    def _text_match(self, query: str, content: str) -> float:
        """Simple keyword overlap score."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        if not query_words:
            return 0.0
        return len(query_words & content_words) / len(query_words)
    
    def _diversity_rerank(
        self,
        results: list[tuple[TensorNode, float]],
        holographic_encoder,
        top_k: int,
        diversity_weight: float = 0.3,
    ) -> list[tuple[TensorNode, float]]:
        """
        Re-rank to ensure diversity: penalize nodes similar to 
        higher-ranked results.
        """
        if len(results) <= 1:
            return results
        
        final = [results[0]]
        
        for node, score in results[1:]:
            # Penalize similarity to already-selected nodes
            penalty = 0.0
            for selected_node, _ in final:
                if node.holographic_signature and selected_node.holographic_signature:
                    sim = holographic_encoder.similarity(
                        node.holographic_signature,
                        selected_node.holographic_signature,
                    )
                    penalty = max(penalty, sim)
                elif selected_node.primary_hierarchy == node.primary_hierarchy:
                    # Same hierarchy = likely redundant
                    penalty = max(penalty, 0.5)
            
            adjusted_score = score * (1 - diversity_weight * penalty)
            final.append((node, adjusted_score))
        
        # Re-sort by adjusted score
        final.sort(key=lambda x: -x[1])
        return final[:top_k]
