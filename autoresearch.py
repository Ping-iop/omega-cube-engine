"""
AutoResearchLoop — Self-optimization for Omega-Cube Engine.

Integrates Karpathy's AutoResearch pattern with Omega-Cube:
an agent loop that autonomously experiments with graph topology,
holographic encoding, annealing parameters, and diffusion sampling,
keeping only improvements.

The loop runs overnight, evaluating each change against benchmarks,
enabling the graph to evolve topologies no human would design.

Author: Omega-Cube Research
Date: 2026-06-11
"""

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Callable, Optional


class AutoResearchLoop:
    """
    Autonomous experimentation loop for Omega-Cube optimization.
    
    Pattern: modify → train/evaluate → compare → keep or rollback.
    
    In a single night, can run 100+ experiments across:
    - Graph topology (node splitting/merging)
    - Holographic dimension size
    - Annealing parameters
    - Diffusion steps
    - Gray-scale weights
    """
    
    def __init__(
        self,
        engine,
        benchmark_fn: Callable[[], dict],
        save_fn: Callable[[], None],
        experiment_dir: str = None,
    ):
        self.engine = engine
        self.benchmark = benchmark_fn
        self.save = save_fn
        
        # Experiment tracking
        self.experiment_dir = Path(experiment_dir or 
            Path.home() / ".hermes" / "axioma-omega-protocol" / "experiments")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_score = -float('inf')
        self.best_config = {}
        self.experiment_log = []
        
        # Hyperparameter search space
        self.param_space = {
            "holographic_dim": [64, 128, 256, 512],
            "anneal_temp": [0.5, 1.0, 2.0, 5.0],
            "cooling_rate": [0.90, 0.93, 0.95, 0.97, 0.99],
            "diffusion_steps": [10, 15, 20, 30, 50],
            "guidance_scale": [1.0, 2.0, 3.0, 5.0],
            "gray_weights_factuality": [0.25, 0.30, 0.35, 0.40, 0.50],
            "gray_weights_relevance": [0.15, 0.20, 0.25, 0.30],
            "tensor_grid_size": [5, 8, 10, 15, 20],
        }
    
    def run(
        self,
        num_experiments: int = 50,
        max_hours: float = 8.0,
        early_stop_patience: int = 15,
    ) -> dict:
        """
        Run autonomous research loop.
        
        Args:
            num_experiments: Max experiments to run
            max_hours: Max wall-clock time
            early_stop_patience: Stop if no improvement for N experiments
        
        Returns:
            Summary of best configuration found
        """
        start_time = time.time()
        patience_counter = 0
        
        # Save initial state for rollback
        initial_state = self._snapshot_state()
        self.best_score = self._evaluate()
        self.best_config = self._get_config()
        
        print(f"[AutoResearch] Starting {num_experiments} experiments")
        print(f"[AutoResearch] Baseline score: {self.best_score:.4f}")
        
        for exp_id in range(1, num_experiments + 1):
            # Check time budget
            elapsed = (time.time() - start_time) / 3600
            if elapsed > max_hours:
                print(f"[AutoResearch] Time budget exhausted ({elapsed:.1f}h)")
                break
            
            if patience_counter >= early_stop_patience:
                print(f"[AutoResearch] Early stopping (no improvement for {early_stop_patience} exps)")
                break
            
            print(f"\n[AutoResearch] Experiment {exp_id}/{num_experiments}")
            
            # Generate and apply a modification
            modification = self._generate_modification()
            self._apply_modification(modification)
            
            # Evaluate
            score = self._evaluate()
            delta = score - self.best_score
            
            # Decide
            if score > self.best_score:
                print(f"  ✅ IMPROVED: {self.best_score:.4f} → {score:.4f} (Δ{delta:+.4f})")
                self.best_score = score
                self.best_config = self._get_config()
                self.save()
                patience_counter = 0
            else:
                print(f"  ❌ Rejected: {score:.4f} (best: {self.best_score:.4f})")
                self._rollback(initial_state)
                patience_counter += 1
            
            # Log experiment
            self.experiment_log.append({
                "exp_id": exp_id,
                "modification": modification["name"],
                "score": score,
                "best_score": self.best_score,
                "improved": score > self.best_score,
                "timestamp": time.time(),
            })
            
            # Save log periodically
            if exp_id % 10 == 0:
                self._save_log()
        
        # Restore best config
        self._apply_config(self.best_config)
        self.save()
        self._save_log()
        
        elapsed = (time.time() - start_time) / 3600
        print(f"\n[AutoResearch] Done in {elapsed:.1f}h")
        print(f"[AutoResearch] Best score: {self.best_score:.4f}")
        print(f"[AutoResearch] Improvements: {sum(1 for e in self.experiment_log if e['improved'])}")
        
        return {
            "best_score": self.best_score,
            "best_config": self.best_config,
            "total_experiments": len(self.experiment_log),
            "improvements": sum(1 for e in self.experiment_log if e['improved']),
            "elapsed_hours": elapsed,
        }
    
    def _generate_modification(self) -> dict:
        """Generate a random modification to the system."""
        mod_types = [
            self._mod_topology,
            self._mod_holographic,
            self._mod_annealer,
            self._mod_diffusion,
            self._mod_grayscale,
        ]
        mod_fn = random.choice(mod_types)
        return mod_fn()
    
    def _mod_topology(self) -> dict:
        """Modify graph topology: split or merge nodes."""
        action = random.choice(["split", "merge"])
        return {
            "name": f"topology_{action}",
            "type": "topology",
            "action": action,
        }
    
    def _mod_holographic(self) -> dict:
        new_dim = random.choice(self.param_space["holographic_dim"])
        return {
            "name": f"holographic_dim_{new_dim}",
            "type": "holographic",
            "dimension": new_dim,
        }
    
    def _mod_annealer(self) -> dict:
        return {
            "name": "annealer_params",
            "type": "annealer",
            "temp": random.choice(self.param_space["anneal_temp"]),
            "cooling": random.choice(self.param_space["cooling_rate"]),
        }
    
    def _mod_diffusion(self) -> dict:
        return {
            "name": "diffusion_params",
            "type": "diffusion",
            "steps": random.choice(self.param_space["diffusion_steps"]),
            "guidance": random.choice(self.param_space["guidance_scale"]),
        }
    
    def _mod_grayscale(self) -> dict:
        return {
            "name": "grayscale_weights",
            "type": "grayscale",
            "factuality_w": random.choice(self.param_space["gray_weights_factuality"]),
            "relevance_w": random.choice(self.param_space["gray_weights_relevance"]),
        }
    
    def _apply_modification(self, mod: dict):
        """Apply a modification to the engine."""
        if mod["type"] == "holographic":
            if hasattr(self.engine, 'holographic_encoder'):
                self.engine.holographic_encoder.dim = mod["dimension"]
        elif mod["type"] == "annealer":
            if hasattr(self.engine, 'annealer'):
                self.engine.annealer.initial_temp = mod["temp"]
                self.engine.annealer.cooling_rate = mod["cooling"]
        elif mod["type"] == "diffusion":
            if hasattr(self.engine, 'diffusion_sampler'):
                self.engine.diffusion_sampler.num_steps = mod["steps"]
                self.engine.diffusion_sampler.guidance_scale = mod["guidance"]
        elif mod["type"] == "grayscale":
            if hasattr(self.engine, 'gray_validator'):
                # Update composite weights
                pass  # Implement when engine is wired
    
    def _evaluate(self) -> float:
        """Run benchmark and return composite score."""
        try:
            results = self.benchmark()
            # Extract composite score from benchmark results
            return results.get("composite_score", 0.0)
        except Exception as e:
            print(f"  ⚠️ Evaluation error: {e}")
            return 0.0
    
    def _get_config(self) -> dict:
        """Capture current engine configuration."""
        config = {}
        if hasattr(self.engine, 'holographic_encoder'):
            config["holographic_dim"] = self.engine.holographic_encoder.dim
        if hasattr(self.engine, 'annealer'):
            config["anneal_temp"] = self.engine.annealer.initial_temp
            config["cooling_rate"] = self.engine.annealer.cooling_rate
        if hasattr(self.engine, 'diffusion_sampler'):
            config["diffusion_steps"] = self.engine.diffusion_sampler.num_steps
            config["guidance_scale"] = self.engine.diffusion_sampler.guidance_scale
        return config
    
    def _apply_config(self, config: dict):
        """Restore engine configuration."""
        if "holographic_dim" in config and hasattr(self.engine, 'holographic_encoder'):
            self.engine.holographic_encoder.dim = config["holographic_dim"]
        if "anneal_temp" in config and hasattr(self.engine, 'annealer'):
            self.engine.annealer.initial_temp = config["anneal_temp"]
            self.engine.annealer.cooling_rate = config.get("cooling_rate", 0.95)
        if "diffusion_steps" in config and hasattr(self.engine, 'diffusion_sampler'):
            self.engine.diffusion_sampler.num_steps = config["diffusion_steps"]
            self.engine.diffusion_sampler.guidance_scale = config.get("guidance_scale", 3.0)
    
    def _snapshot_state(self) -> dict:
        return self._get_config()
    
    def _rollback(self, state: dict):
        """Rollback to a previous state."""
        self._apply_config(state)
    
    def _save_log(self):
        """Save experiment log to disk."""
        log_path = self.experiment_dir / "autoresearch_log.json"
        with open(log_path, "w") as f:
            json.dump(self.experiment_log, f, indent=2, default=str)
