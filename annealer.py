"""
QuantumInspiredAnnealer — Topology optimization via simulated annealing.

Implements the "rotating magnetic cubes" concept: each cube (topic domain)
searches for its optimal configuration simultaneously. The system converges
to a minimum-energy state where all cubes are aligned to answer the query.

Key properties:
- Non-deterministic: same query can find novel patterns
- Parallel: all cubes anneal simultaneously
- Emergent: patterns arise from local interactions, not central planning

Author: Omega-Cube Research
Date: 2026-06-11
"""

import math
import random
from typing import Callable, Optional


class QuantumInspiredAnnealer:
    """
    Topology optimizer that finds optimal graph configurations via
    simulated annealing with quantum-inspired tunneling.
    
    Each "cube" (topic domain subgraph) can rotate through its 
    dimensions, and the system settles into configurations that
    maximize query relevance.
    """
    
    def __init__(
        self,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.95,
        min_temp: float = 0.01,
        steps_per_temp: int = 5,
        tunneling_prob: float = 0.1,  # Quantum-inspired: jump to random config
        seed: int = 42,
    ):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.steps_per_temp = steps_per_temp
        self.tunneling_prob = tunneling_prob
        random.seed(seed)
    
    def anneal(
        self,
        cubes: list[dict],
        energy_fn: Callable[[list[dict]], float],
        neighbor_fn: Callable[[list[dict]], list[dict]],
        max_iterations: int = 500,
    ) -> tuple[list[dict], float, list[float]]:
        """
        Run annealing to find optimal cube configuration.
        
        Args:
            cubes: List of cube states (each cube = dict with config)
            energy_fn: Function that scores a configuration (lower = better)
            neighbor_fn: Function that generates a neighboring configuration
            max_iterations: Maximum annealing steps
        
        Returns:
            (best_config, best_energy, energy_history)
        """
        current = [dict(c) for c in cubes]
        current_energy = energy_fn(current)
        
        best = [dict(c) for c in current]
        best_energy = current_energy
        
        temp = self.initial_temp
        energy_history = [current_energy]
        
        iteration = 0
        while temp > self.min_temp and iteration < max_iterations:
            for _ in range(self.steps_per_temp):
                # Generate neighbor (rotate one random cube)
                if random.random() < self.tunneling_prob:
                    # Quantum tunneling: jump to a distant configuration
                    candidate = self._tunnel(cubes, neighbor_fn)
                else:
                    candidate = neighbor_fn(current)
                
                candidate_energy = energy_fn(candidate)
                delta = candidate_energy - current_energy
                
                # Metropolis criterion
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current = candidate
                    current_energy = candidate_energy
                    
                    if current_energy < best_energy:
                        best = [dict(c) for c in current]
                        best_energy = current_energy
                
                iteration += 1
                energy_history.append(current_energy)
            
            temp *= self.cooling_rate
        
        return best, best_energy, energy_history
    
    def _tunnel(self, cubes: list[dict], neighbor_fn: Callable) -> list[dict]:
        """Quantum tunneling: make multiple random moves to escape local minima."""
        result = [dict(c) for c in cubes]
        for _ in range(random.randint(2, 5)):
            result = neighbor_fn(result)
        return result
    
    def multi_objective_anneal(
        self,
        cubes: list[dict],
        energy_fns: list[tuple[Callable[[list[dict]], float], float]],  # (fn, weight)
        neighbor_fn: Callable[[list[dict]], list[dict]],
        max_iterations: int = 500,
    ) -> tuple[list[dict], list[float]]:
        """
        Multi-objective annealing with weighted energy functions.
        
        Each energy function captures a different quality criterion
        (e.g., relevance, coherence, novelty, efficiency).
        """
        def combined_energy(config):
            return sum(w * fn(config) for fn, w in energy_fns)
        
        best, _, history = self.anneal(cubes, combined_energy, neighbor_fn, max_iterations)
        return best, history


class CubeRotator:
    """
    Manages rotation of individual cubes (topic domains) through 
    their hierarchy dimensions.
    
    A "rotation" = changing which hierarchy dimension is primary,
    which subtopic is exposed, and which associations are active.
    """
    
    @staticmethod
    def random_rotation(cube: dict) -> dict:
        """Randomly rotate a cube through its accessible dimensions."""
        new_cube = dict(cube)
        
        if "dimensions" in new_cube and new_cube["dimensions"]:
            # Rotate primary dimension
            n_dims = len(new_cube["dimensions"])
            new_cube["active_dimension"] = random.randint(0, n_dims - 1)
        
        if "subtopics" in new_cube and new_cube["subtopics"]:
            # Expose a different subtopic
            new_cube["exposed_subtopic"] = random.choice(new_cube["subtopics"])
        
        if "associations" in new_cube and new_cube["associations"]:
            # Activate/deactivate random associations
            for assoc_key in list(new_cube["associations"].keys()):
                new_cube["associations"][assoc_key] = random.random() > 0.5
        
        return new_cube
    
    @staticmethod
    def query_aligned_rotation(cube: dict, query_vector: list[float]) -> dict:
        """
        Rotate cube to align with query vector (deterministic alternative).
        Each dimension is scored against query, best dimension is selected.
        """
        new_cube = dict(cube)
        
        if "dimension_vectors" in new_cube and new_cube["dimension_vectors"]:
            dims = new_cube["dimension_vectors"]
            scores = [
                CubeRotator._cosine_similarity(query_vector, dim_vec)
                for dim_vec in dims
            ]
            if scores:
                new_cube["active_dimension"] = scores.index(max(scores))
        
        return new_cube
    
    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        if len(v1) != len(v2) or len(v1) == 0:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(v**2 for v in v1))
        n2 = math.sqrt(sum(v**2 for v in v2))
        return dot / (n1 * n2) if n1 and n2 else 0.0


# ─── Pattern Emergence ──────────────────────────────────────────────

class PatternEmergence:
    """
    Detects emergent patterns across cubes after annealing.
    
    A "pattern" is a configuration where multiple cubes independently
    settled into compatible states, forming a multi-topic answer.
    """
    
    @staticmethod
    def extract_patterns(cubes: list[dict], threshold: float = 0.7) -> list[dict]:
        """
        Extract cross-cube patterns from an annealed configuration.
        
        Returns patterns where cubes show high inter-alignment scores.
        """
        patterns = []
        
        for i, cube in enumerate(cubes):
            aligned_with = []
            cube_vector = cube.get("active_vector", [])
            
            for j, other in enumerate(cubes):
                if i == j:
                    continue
                other_vector = other.get("active_vector", [])
                
                alignment = PatternEmergence._alignment_score(cube_vector, other_vector)
                if alignment > threshold:
                    aligned_with.append({
                        "cube_id": other.get("id", j),
                        "alignment": alignment,
                    })
            
            if aligned_with:
                patterns.append({
                    "anchor_cube": cube.get("id", i),
                    "cube_topic": cube.get("topic", ""),
                    "aligned_cubes": aligned_with,
                    "pattern_strength": sum(a["alignment"] for a in aligned_with) / len(aligned_with),
                    "exposed_content": cube.get("exposed_content", ""),
                })
        
        # Sort by pattern strength
        patterns.sort(key=lambda p: -p["pattern_strength"])
        return patterns
    
    @staticmethod
    def _alignment_score(v1: list[float], v2: list[float]) -> float:
        if not v1 or not v2:
            return 0.0
        min_len = min(len(v1), len(v2))
        dot = sum(v1[i] * v2[i] for i in range(min_len))
        n1 = math.sqrt(sum(v**2 for v in v1[:min_len]))
        n2 = math.sqrt(sum(v**2 for v in v2[:min_len]))
        if n1 == 0 or n2 == 0:
            return 0.0
        # Map from [-1,1] to [0,1]
        return max(0.0, (dot / (n1 * n2) + 1) / 2)
