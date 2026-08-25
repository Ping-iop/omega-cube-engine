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


# ═══════════════════════════════════════════════════════════════════
# Domain taxonomy — SINGLE SOURCE OF TRUTH (P1.6/P1.7, 2026-08-25)
#
# Previously defined in 3 places (router.py, gpu_router.py GBNF +
# intelligent_pipeline.py). Everything derives from here now: keyword
# routing, GPU grammar, validation sets and prompts.
# A new domain = add ONE entry here (or discover it into this dict);
# grammar/validation regenerate automatically.
# ═══════════════════════════════════════════════════════════════════

STANDARD_DOMAINS = {
    "math": {
        "subdomains": ["algebra", "calculus", "statistics", "geometry",
                       "number_theory", "topology", "information_theory", "optimization"],
        "keywords": ["math", "mathematics", "equation", "theorem", "proof",
                     "calculus", "algebra", "geometry", "statistics", "probability",
                     "entropy", "gradient", "derivative", "integral", "matrix"],
    },
    "code": {
        "subdomains": ["python", "javascript", "rust", "cpp", "algorithms",
                       "systems", "web", "database", "devops"],
        "keywords": ["code", "programming", "function", "class", "api", "bug",
                     "compile", "algorithm", "python", "javascript", "rust",
                     "react", "docker", "kubernetes", "sql", "database"],
    },
    "science": {
        "subdomains": ["physics", "chemistry", "biology", "astronomy", "neuroscience"],
        "keywords": ["physics", "chemistry", "biology", "science", "experiment",
                     "molecule", "atom", "cell", "organism", "evolution",
                     "quantum", "relativity", "dna", "protein", "neuron"],
    },
    "engineering": {
        "subdomains": ["electrical", "mechanical", "civil", "robotics", "materials"],
        "keywords": ["engineering", "circuit", "voltage", "mechanical", "structural",
                     "robot", "sensor", "actuator", "motor", "propulsion"],
    },
    "language": {
        "subdomains": ["linguistics", "translation", "grammar", "literature", "creative"],
        "keywords": ["language", "grammar", "translate", "write", "essay",
                     "poem", "story", "literature", "linguistics", "syntax"],
    },
    "law": {
        "subdomains": ["contract", "ip", "criminal", "international", "tax"],
        "keywords": ["law", "legal", "contract", "court", "statute",
                     "compliance", "liability", "patent", "copyright"],
    },
    "medical": {
        "subdomains": ["diagnosis", "treatment", "pharmacology", "surgery", "epidemiology"],
        "keywords": ["medical", "diagnosis", "treatment", "drug", "surgery",
                     "patient", "symptom", "disease", "cancer", "therapy"],
    },
    "business": {
        "subdomains": ["finance", "marketing", "management", "economics", "strategy"],
        "keywords": ["business", "finance", "marketing", "revenue", "profit",
                     "strategy", "market", "investment", "stock", "startup"],
    },
    "philosophy": {
        "subdomains": ["ethics", "epistemology", "metaphysics", "logic", "political"],
        "keywords": ["philosophy", "ethics", "epistemology", "metaphysics",
                     "logic", "consciousness", "existence", "morality"],
    },
    "gaming": {
        "subdomains": ["game_design", "game_dev", "strategy", "esports", "mechanics"],
        "keywords": ["game", "gaming", "player", "level", "strategy",
                     "rpg", "fps", "moba", "esports", "achievement"],
    },
}

# Catch-all / system domains without rich taxonomy but valid classifier output
EXTRA_DOMAINS: tuple[str, ...] = ("general", "memory")

ALL_DOMAINS: tuple[str, ...] = tuple(STANDARD_DOMAINS.keys()) + EXTRA_DOMAINS

VALID_DOMAINS = frozenset(ALL_DOMAINS)


def gbnf_domain_grammar() -> str:
    """GBNF grammar forcing the model to output EXACTLY one valid domain."""
    alts = " | ".join(f'"{d}"' for d in ALL_DOMAINS)
    return f'root ::= ({alts})'


def domains_prompt_line() -> str:
    """Comma-separated domain list for classification prompts."""
    return ", ".join(ALL_DOMAINS)
