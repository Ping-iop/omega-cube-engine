"""
MARP Scheduler — GPU-native shard activation for Omega-Cube Engine.

Manages model shard lifecycle in GPU unified memory. All shards reside
in VRAM; activation means allowing compute kernels to run, not moving data.

This is the "kitchen manager" in the restaurant metaphor.
"""

from __future__ import annotations

import time
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
