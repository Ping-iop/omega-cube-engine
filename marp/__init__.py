"""
MARP — Model-Agnostic Routing Protocol (Omega-Cube Component #10).

Dynamic hierarchical model loading using Omega-Cube's knowledge graph.
Routes queries to domain-specific model shards instead of loading entire models.

Usage:
    from omega_cube.marp import MARPRouter, ShardScheduler
    from omega_cube.marp.protocol import ShardConfig

    router = MARPRouter()
    scheduler = ShardScheduler()
    scheduler.register(ShardConfig(name="math_v1", domains=["math"], ...))
    
    decision = router.route("Explain entropy", [...])
    active = scheduler.activate_for_decision(decision)
"""

from omega_cube.marp.protocol import (
    DomainTicket, ContextNode, ShardConfig, ShardActivation,
    RouterDecision, MARPMode,
)
from omega_cube.marp.router import MARPRouter, DomainScore
from omega_cube.marp.scheduler import ShardScheduler, SchedulerStats

__all__ = [
    "MARPRouter", "ShardScheduler",
    "DomainTicket", "ContextNode", "ShardConfig", "ShardActivation",
    "RouterDecision", "MARPMode",
    "DomainScore", "SchedulerStats",
]
