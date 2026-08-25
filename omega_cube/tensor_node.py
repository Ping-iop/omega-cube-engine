"""
TensorNode — Multi-dimensional hierarchical nodes for Omega-Cube Engine.

Each node exists simultaneously in multiple hierarchies (tensor product),
enabling multi-perspective retrieval without graph duplication.

Concept: Like a Rubik's cube where each sticker has 3 colors — the same
knowledge element is accessible from different dimensional axes.

Author: Omega-Cube Research
Date: 2026-06-11
"""

from dataclasses import dataclass, field
from typing import Optional
import hashlib


@dataclass
class TensorNode:
    """
    A knowledge node existing in N-dimensional hierarchy space.
    
    Instead of one hierarchy path, the node has multiple "coordinates"
    in different dimensional axes (topic, resolution, price, etc.).
    
    The tensor_position represents the node's location in a normalized
    N-dimensional space [0,1]^N, enabling geometric queries.
    """
    
    content: str
    hierarchies: list[str]          # Multiple hierarchy paths (one per dimension)
    tensor_position: list[float]    # Position in N-dim space [0,1] per dimension
    node_type: str = "CONCEPT"      # AXIOM, CONCEPT, INSTANCE, SESSION
    confidence: float = 0.9
    tags: list = field(default_factory=list)
    associations: list[str] = field(default_factory=list)
    
    # Holographic signature: compressed representation of self + neighbors
    holographic_signature: Optional[list[float]] = None
    
    # Gray-scale verification (H-Bit inspired): multi-bit truth assessment
    gray_scale: Optional[dict] = None  # {dimension: scale_0_to_100, ...}
    
    # ── Cadena de color (Fase 2, PLAN 2026-08-09) ─────────────────
    # hue: tono [0,360) heredado del axioma origen (identidad de la cadena)
    # saturation: λ^depth (profundidad derivacional / certeza)
    # lineage: ruta completa al centro [axiom_id, ..., node_id]
    # hue_origin: node_id del axioma origen (None = huérfano)
    # hue_vector: [cos, sin] resultante cuando el nodo es mezcla de cadenas
    hue: Optional[float] = None
    saturation: Optional[float] = None
    lineage: list[str] = field(default_factory=list)
    hue_origin: Optional[str] = None
    hue_vector: Optional[list[float]] = None
    
    # Metadata
    node_id: Optional[str] = None
    created_at: float = 0.0
    access_count: int = 0
    
    def __post_init__(self):
        if self.node_id is None:
            self.node_id = hashlib.sha256(
                (self.content[:100] + str(self.hierarchies)).encode()
            ).hexdigest()[:16]
    
    @property
    def primary_hierarchy(self) -> str:
        """The first (primary) hierarchy path."""
        return self.hierarchies[0] if self.hierarchies else ""
    
    @property
    def dimension_count(self) -> int:
        """Number of simultaneous hierarchy dimensions."""
        return len(self.hierarchies)
    
    def distance_to(self, other: 'TensorNode') -> float:
        """Euclidean distance in tensor space."""
        if not self.tensor_position or not other.tensor_position:
            return 1.0
        min_len = min(len(self.tensor_position), len(other.tensor_position))
        return sum(
            (self.tensor_position[i] - other.tensor_position[i]) ** 2
            for i in range(min_len)
        ) ** 0.5 / (min_len ** 0.5)  # Normalized
    
    def matches_dimension(self, dimension_idx: int, query_pos: float, tolerance: float = 0.2) -> bool:
        """Check if node matches a query position in a specific dimension."""
        if dimension_idx >= len(self.tensor_position):
            return False
        return abs(self.tensor_position[dimension_idx] - query_pos) <= tolerance
    
    def gray_scale_score(self, dimension: str = None) -> float:
        """
        H-Bit inspired multi-bit truth assessment.
        Returns a score 0-100 indicating how close this node is to 
        ground truth in the given dimension.
        """
        if self.gray_scale is None:
            return self.confidence * 100
        if dimension and dimension in self.gray_scale:
            return self.gray_scale[dimension]
        return sum(self.gray_scale.values()) / len(self.gray_scale) if self.gray_scale else 50.0
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "content": self.content,
            "hierarchies": self.hierarchies,
            "tensor_position": self.tensor_position,
            "node_type": self.node_type,
            "confidence": self.confidence,
            "tags": self.tags,
            "associations": self.associations,
            "holographic_signature": self.holographic_signature,
            "gray_scale": self.gray_scale,
            "hue": self.hue,
            "saturation": self.saturation,
            "lineage": self.lineage,
            "hue_origin": self.hue_origin,
            "hue_vector": self.hue_vector,
            "created_at": self.created_at,
            "access_count": self.access_count,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TensorNode':
        # Campos de color con defaults → load() tolera nodos sin colorear
        return cls(
            content=d.get("content", ""),
            hierarchies=d.get("hierarchies", []),
            tensor_position=d.get("tensor_position", []),
            node_type=d.get("node_type", "CONCEPT"),
            confidence=d.get("confidence", 0.9),
            tags=d.get("tags", []),
            associations=d.get("associations", []),
            holographic_signature=d.get("holographic_signature"),
            gray_scale=d.get("gray_scale"),
            hue=d.get("hue"),
            saturation=d.get("saturation"),
            lineage=d.get("lineage", []),
            hue_origin=d.get("hue_origin"),
            hue_vector=d.get("hue_vector"),
            node_id=d.get("node_id"),
            created_at=d.get("created_at", 0.0),
            access_count=d.get("access_count", 0),
        )


# ─── Tensor Index: spatial lookup structure ─────────────────────────

class TensorIndex:
    """
    Spatial index for fast N-dimensional proximity queries.
    
    Implements a simple grid-based index over the tensor space,
    enabling O(1) lookups for approximate nearest neighbors.
    """
    
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.grid: dict[tuple, list[str]] = {}  # cell_coords → [node_ids]
        self.node_map: dict[str, TensorNode] = {}
    
    def insert(self, node: TensorNode):
        cell = self._get_cell(node.tensor_position)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(node.node_id)
        self.node_map[node.node_id] = node
    
    def query(self, position: list[float], radius: float = 0.3) -> list[TensorNode]:
        """Find nodes within radius in tensor space."""
        cell = self._get_cell(position)
        candidates = []
        seen = set()
        
        # Search current cell + neighbors
        grid_radius = max(1, int(radius * self.grid_size))
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in self.grid:
                    for nid in self.grid[neighbor_cell]:
                        if nid not in seen:
                            seen.add(nid)
                            node = self.node_map.get(nid)
                            if node:
                                # Check exact distance
                                dist = self._distance(position, node.tensor_position)
                                if dist <= radius:
                                    candidates.append((node, dist))
        
        candidates.sort(key=lambda x: x[1])
        return [n for n, _ in candidates]
    
    def _get_cell(self, position: list[float]) -> tuple:
        if len(position) >= 2:
            return (
                min(self.grid_size - 1, int(position[0] * self.grid_size)),
                min(self.grid_size - 1, int(position[1] * self.grid_size)),
            )
        return (0, 0)
    
    @staticmethod
    def _distance(p1: list[float], p2: list[float]) -> float:
        min_len = min(len(p1), len(p2))
        return sum((p1[i] - p2[i])**2 for i in range(min_len)) ** 0.5 / (min_len ** 0.5)
