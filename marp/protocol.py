"""
MARP Protocol — Axion-Cube native data structures.

These are the data types that flow between the Axion-Cube knowledge graph
and model shards. They're defined here (not in a separate repo) because
MARP IS Axion-Cube's routing layer — it cannot exist without the graph.

Component #10 of Axion-Cube Engine.
Updated 2026-07-26: Holographic context nodes + grounding scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class MARPMode(Enum):
    WRAPPER = auto()   # Existing model + LoRA adapters per domain
    NATIVE = auto()    # Model trained with Axion-Cube sharding
    HYBRID = auto()    # Route between API providers per domain


@dataclass
class ContextNode:
    """A knowledge graph node injected as pre-resolved context.

    v2: Now carries holographic signature (256D) and grounding score
    so the worker can do similarity search over injected context
    and verify evidence quality before using it.
    """
    node_id: str
    content: str
    weight: float
    domain: str
    depth: int = 1
    dimensions: list[str] = field(default_factory=list)
    # ── v2 additions (arXiv 2026 improvements) ──
    holographic_signature: list[float] = field(default_factory=list)  # 256D embedding
    grounding_score: float = 0.0  # PAGE-RAG boundary control score
    gray_scale: float = 0.0       # Axion gray-scale truth value


@dataclass
class DomainTicket:
    """Structured request from Axion-Cube Router to model shards."""
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
    # ── v2 additions ──
    bias_detected: bool = False
    bias_type: str = "none"
    bias_counteracted: bool = False


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
    """Result of Axion-Cube routing."""
    ticket: DomainTicket
    active_shards: list[str]
    activation_reasons: dict[str, str] = field(default_factory=dict)
    context_injected: bool = False
    omega_cube_nodes_used: int = 0
    routing_time_ms: float = 0.0
    token_savings_estimate: float = 0.0
    # ── v2 additions ──
    hierarchical_routing_used: bool = False
    boundary_filtered: int = 0
    bias_detections: int = 0
