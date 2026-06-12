"""
MARP Protocol — Omega-Cube native data structures.

These are the data types that flow between the Omega-Cube knowledge graph
and model shards. They're defined here (not in a separate repo) because
MARP IS Omega-Cube's routing layer — it cannot exist without the graph.

Component #10 of Omega-Cube Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class MARPMode(Enum):
    WRAPPER = auto()   # Existing model + LoRA adapters per domain
    NATIVE = auto()    # Model trained with Omega-Cube sharding
    HYBRID = auto()    # Route between API providers per domain


@dataclass
class ContextNode:
    """A knowledge graph node injected as pre-resolved context."""
    node_id: str
    content: str
    weight: float
    domain: str
    depth: int = 1
    dimensions: list[str] = field(default_factory=list)


@dataclass
class DomainTicket:
    """Structured request from Omega-Cube Router to model shards."""
    query: str
    active_domains: list[str]
    confidence: dict[str, float] = field(default_factory=dict)
    context_nodes: list[ContextNode] = field(default_factory=list)
    depth: str = "intermediate"
    format: str = "explanation"
    audience: str = "technical"
    max_tokens: int = 2048
    temperature: float = 0.7
    conversation_history: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ShardConfig:
    """Configuration for a model shard. Model-agnostic."""
    name: str
    domains: list[str]
    mode: MARPMode = MARPMode.WRAPPER
    base_model: str = ""
    adapter_type: str = "lora"
    adapter_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_model: Optional[str] = None
    gpu_memory_mb: int = 0
    priority: int = 0
    capabilities: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class ShardActivation:
    """Runtime state of a shard."""
    config: ShardConfig
    is_active: bool = False
    last_used: float = 0.0
    tokens_generated: int = 0
    gpu_memory_used_mb: int = 0


@dataclass
class RouterDecision:
    """Result of Omega-Cube routing."""
    ticket: DomainTicket
    active_shards: list[str]
    activation_reasons: dict[str, str] = field(default_factory=dict)
    context_injected: bool = False
    omega_cube_nodes_used: int = 0
    routing_time_ms: float = 0.0
    token_savings_estimate: float = 0.0
