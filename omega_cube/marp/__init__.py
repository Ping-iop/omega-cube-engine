"""
MARP — Model-Agnostic Routing Protocol (Axion-Cube Component #10).

Dynamic hierarchical model loading using Axion-Cube's knowledge graph.
Routes queries to domain-specific model shards instead of loading entire models.

v2 (2026-07-26): Added AdaptiveScheduler with session-based domain learning,
holographic context nodes, boundary control, and hallucination detection.

Usage:
    from omega_cube.marp import MARPRouter, AdaptiveScheduler
    from omega_cube.marp.protocol import ShardConfig

    router = MARPRouter(engine=axion_cube_engine)  # v2: pass engine for hierarchical routing
    scheduler = AdaptiveScheduler()
    scheduler.register(ShardConfig(name="math_v1", domains=["math"], ...))
    
    decision = router.route("Explain entropy", [...])
    active = scheduler.activate_for_decision(decision)
"""

from omega_cube.marp.protocol import (
    DomainTicket, ContextNode, ShardConfig, ShardActivation,
    RouterDecision, MARPMode,
)
from omega_cube.marp.router import MARPRouter, DomainScore
from omega_cube.marp.scheduler import ShardScheduler, AdaptiveScheduler, SchedulerStats

__all__ = [
    "MARPRouter", "ShardScheduler", "AdaptiveScheduler",
    "DomainTicket", "ContextNode", "ShardConfig", "ShardActivation",
    "RouterDecision", "MARPMode",
    "DomainScore", "SchedulerStats",
]
