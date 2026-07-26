"""
MARP Scheduler — GPU-native shard activation for Axion-Cube Engine.

Manages model shard lifecycle in GPU unified memory. All shards reside
in VRAM; activation means allowing compute kernels to run, not moving data.

This is the "kitchen manager" in the restaurant metaphor.

v2 (2026-07-26): Added AdaptiveScheduler with session-based domain
frequency learning for smarter prefetch predictions.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from omega_cube.marp.protocol import ShardConfig, ShardActivation, RouterDecision


@dataclass
class SchedulerStats:
    total_shards: int = 0
    active_shards: int = 0
    idle_shards: int = 0
    total_gpu_memory_mb: int = 0
    used_gpu_memory_mb: int = 0
    activations_total: int = 0
    avg_activation_time_ms: float = 0.0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    uptime_seconds: float = 0.0
    # v2 additions
    adaptive_predictions: int = 0
    adaptive_correct: int = 0


class ShardScheduler:
    """GPU-native shard lifecycle manager."""

    def __init__(self, max_gpu_memory_mb: int = 0):
        self._shards: dict[str, ShardActivation] = {}
        self._max_gpu = max_gpu_memory_mb
        self._stats = SchedulerStats()
        self._start = time.time()

    def register(self, config: ShardConfig) -> ShardActivation:
        act = ShardActivation(config=config, gpu_memory_used_mb=config.gpu_memory_mb)
        self._shards[config.name] = act
        self._stats.total_shards = len(self._shards)
        self._stats.total_gpu_memory_mb += config.gpu_memory_mb
        return act

    def activate_for_decision(self, decision: RouterDecision) -> list[ShardActivation]:
        t0 = time.perf_counter()
        target = set(decision.active_shards)
        current = {n for n, s in self._shards.items() if s.is_active}

        # Deactivate
        for name in current - target:
            if name in self._shards:
                self._shards[name].is_active = False
                self._stats.active_shards -= 1
                self._stats.idle_shards += 1

        # Activate
        for name in target - current:
            if name in self._shards:
                if not self._can_activate(self._shards[name]):
                    self._evict_lru()
                self._shards[name].is_active = True
                self._shards[name].last_used = time.time()
                self._stats.active_shards += 1
                self._stats.idle_shards -= 1
                self._stats.activations_total += 1

        t = (time.perf_counter() - t0) * 1000
        n = self._stats.activations_total
        self._stats.avg_activation_time_ms = (
            (self._stats.avg_activation_time_ms * (n - 1) + t) / n if n > 0 else 0
        )
        self._stats.uptime_seconds = time.time() - self._start
        self._stats.used_gpu_memory_mb = sum(
            s.gpu_memory_used_mb for s in self._shards.values() if s.is_active
        )
        return [s for s in self._shards.values() if s.is_active]

    def prefetch(self, domains: list[str], max_n: int = 2) -> list[str]:
        prefetched = []
        for domain in domains[:max_n]:
            for name, shard in self._shards.items():
                if domain in shard.config.domains and not shard.is_active:
                    if self._can_activate(shard):
                        shard.is_active = True
                        self._stats.active_shards += 1
                        self._stats.idle_shards -= 1
                        self._stats.prefetch_hits += 1
                        prefetched.append(name)
                        break
                    else:
                        self._stats.prefetch_misses += 1
        return prefetched

    def get_active(self) -> list[ShardActivation]:
        return [s for s in self._shards.values() if s.is_active]

    @property
    def stats(self) -> SchedulerStats:
        self._stats.uptime_seconds = time.time() - self._start
        self._stats.used_gpu_memory_mb = sum(
            s.gpu_memory_used_mb for s in self._shards.values() if s.is_active
        )
        return self._stats

    def _can_activate(self, shard: ShardActivation) -> bool:
        if self._max_gpu == 0:
            return True
        current = sum(s.gpu_memory_used_mb for s in self._shards.values() if s.is_active)
        return (current + shard.gpu_memory_used_mb) <= self._max_gpu

    def _evict_lru(self):
        active = [s for s in self._shards.values() if s.is_active]
        if active:
            lru = min(active, key=lambda s: s.last_used)
            lru.is_active = False
            self._stats.active_shards -= 1
            self._stats.idle_shards += 1


class AdaptiveScheduler(ShardScheduler):
    """v2: Scheduler that learns domain usage patterns per session.

    Tracks domain frequency over a sliding window and uses it to
    predict which shards to prefetch next. Replaces static prefetch
    with data-driven predictions.

    Improvement #5 from MARP review (2026-07-26).
    """

    def __init__(self, max_gpu_memory_mb: int = 0, window_size: int = 20):
        super().__init__(max_gpu_memory_mb)
        self._domain_history: list[str] = []
        self._window_size = window_size
        self._last_predicted: list[str] = []

    def activate_for_decision(self, decision: RouterDecision) -> list[ShardActivation]:
        # Record domains for learning
        self._domain_history.extend(decision.ticket.active_domains)
        # Keep sliding window
        if len(self._domain_history) > self._window_size * 3:
            self._domain_history = self._domain_history[-self._window_size * 2:]

        # Check if previous prediction was correct
        if self._last_predicted:
            actual = set(decision.ticket.active_domains)
            predicted = set(self._last_predicted)
            if actual & predicted:
                self._stats.adaptive_correct += 1
            self._stats.adaptive_predictions += 1

        # Adaptive prefetch based on learned patterns
        predicted = self._predict_next_domains()
        self._last_predicted = predicted
        if predicted:
            self.prefetch(predicted, max_n=2)

        return super().activate_for_decision(decision)

    def _predict_next_domains(self) -> list[str]:
        """Predict next domains from frequency in recent window."""
        if len(self._domain_history) < 3:
            return []
        recent = Counter(self._domain_history[-self._window_size:])
        # Return top-2 most frequent domains (excluding the most recent)
        last_domain = self._domain_history[-1] if self._domain_history else None
        predictions = []
        for domain, count in recent.most_common(4):
            if domain != last_domain and count >= 2:
                predictions.append(domain)
            if len(predictions) >= 2:
                break
        return predictions

    @property
    def prediction_accuracy(self) -> float:
        """Accuracy of adaptive prefetch predictions."""
        if self._stats.adaptive_predictions == 0:
            return 0.0
        return self._stats.adaptive_correct / self._stats.adaptive_predictions
