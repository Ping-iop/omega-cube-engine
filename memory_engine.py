"""AxiomaticMemoryEngine - Motor de memoria axiomática jerárquica.

Búsqueda por navegación desde verdades absolutas (axiomas) hacia abajo 
en la jerarquía con asociaciones laterales y razonamiento por cadena.

Niveles:
  - AXIOM (1.0): Verdades absolutas, punto de partida de búsqueda
  - CONCEPT (0.9): Conocimiento estructurado, categorías
  - INSTANCE (0.8): Datos específicos, casos particulares
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class MemoryNode:
    """Nodo en el grafo de memoria axiomática."""
    content: str
    hierarchy: str
    node_type: str  # "AXIOM", "CONCEPT", "INSTANCE"
    tags: list = field(default_factory=list)
    associations: list = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self):
        return {
            "content": self.content,
            "hierarchy": self.hierarchy,
            "node_type": self.node_type,
            "tags": self.tags,
            "associations": self.associations,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            content=d["content"],
            hierarchy=d["hierarchy"],
            node_type=d["node_type"],
            tags=d.get("tags", []),
            associations=d.get("associations", []),
            confidence=d.get("confidence", 1.0),
        )


class AxiomaticMemoryEngine:
    """Motor de memoria con búsqueda axiomática jerárquica."""

    CONFIDENCE_LEVELS = {
        "AXIOM": 1.0,
        "CONCEPT": 0.9,
        "INSTANCE": 0.8,
    }

    def __init__(self, memory_dir: str = None):
        """Initialize the memory engine."""
        if memory_dir is None:
            axioma_base = os.environ.get("AXIOMA_PROJECT_PATH", 
                str(Path.home() / ".hermes" / "axioma-omega-protocol"))
            memory_dir = os.path.join(axioma_base, "memory")

        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.nodes: dict[str, MemoryNode] = {}
        # Telemetría recalls/usages (métrica de salud 2026-05-07, implementada 2026-08-09):
        # recalls = cuántas veces se consulta la memoria
        # usages  = cuántas veces el consumidor confirma que INTEGRÓ lo recuperado
        self.telemetry_path = self.memory_dir / "telemetry.json"
        self.telemetry_data = self._load_telemetry()

    # --- Telemetría (recalls vs usages) ---

    def _load_telemetry(self) -> dict:
        if self.telemetry_path.exists():
            try:
                return json.loads(self.telemetry_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                pass
        return {"recalls": 0, "usages": 0, "recall_log": [], "usage_log": []}

    def _save_telemetry(self):
        self.telemetry_path.write_text(
            json.dumps(self.telemetry_data, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    def mark_used(self, hierarchies: list, context: str = ""):
        """El consumidor confirma que INTEGRÓ estos nodos en su razonamiento.
        Sin esta llamada, recalls>0 con usages=0 → memoria decorativa."""
        self.telemetry_data["usages"] += 1
        self.telemetry_data["usage_log"].append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "hierarchies": hierarchies[:10],
            "context": context[:150],
        })
        self.telemetry_data["usage_log"] = self.telemetry_data["usage_log"][-200:]
        self._save_telemetry()

    def telemetry(self) -> dict:
        t = self.telemetry_data
        health = "OK"
        if t["recalls"] > 0 and t["usages"] == 0:
            health = ("ALERTA: recalls>0 con usages=0 — la memoria se consulta "
                      "pero NO se integra. Riesgo de memoria decorativa.")
        return {
            "recalls": t["recalls"],
            "usages": t["usages"],
            "health": health,
            "last_recalls": t["recall_log"][-5:],
            "last_usages": t["usage_log"][-5:],
        }

    def add_axiom(self, content: str, hierarchy: str, tags: list = None) -> MemoryNode:
        """Add an axiom (absolute truth)."""
        node = MemoryNode(
            content=content,
            hierarchy=hierarchy,
            node_type="AXIOM",
            tags=tags or [],
            confidence=self.CONFIDENCE_LEVELS["AXIOM"],
        )
        self.nodes[hierarchy] = node
        return node

    def add_concept(self, content: str, hierarchy: str, tags: list = None) -> MemoryNode:
        """Add a concept (structured knowledge)."""
        node = MemoryNode(
            content=content,
            hierarchy=hierarchy,
            node_type="CONCEPT",
            tags=tags or [],
            confidence=self.CONFIDENCE_LEVELS["CONCEPT"],
        )
        self.nodes[hierarchy] = node
        return node

    def add_instance(self, content: str, hierarchy: str, tags: list = None) -> MemoryNode:
        """Add an instance (specific data/case)."""
        node = MemoryNode(
            content=content,
            hierarchy=hierarchy,
            node_type="INSTANCE",
            tags=tags or [],
            confidence=self.CONFIDENCE_LEVELS["INSTANCE"],
        )
        self.nodes[hierarchy] = node
        return node

    def associate(self, hierarchy1: str, hierarchy2: str) -> bool:
        """Create lateral association between two nodes."""
        if hierarchy1 in self.nodes and hierarchy2 in self.nodes:
            n1 = self.nodes[hierarchy1]
            n2 = self.nodes[hierarchy2]
            if hierarchy2 not in n1.associations:
                n1.associations.append(hierarchy2)
            if hierarchy1 not in n2.associations:
                n2.associations.append(hierarchy1)
            return True
        return False

    def query(self, query_text: str, max_depth: int = 5) -> list[dict]:
        """Axiomatic search - navigate hierarchy from axioms downward,
        with fallback to direct search across all node types."""
        results = []
        keywords = self._extract_keywords(query_text)
        
        # Phase 1: Find relevant axioms (priority entry points)
        axiom_matches = []
        for hierarchy, node in self.nodes.items():
            if node.node_type == "AXIOM":
                score = self._relevance_score(node, keywords)
                if score > 0:
                    axiom_matches.append((hierarchy, node, score))
        
        # Phase 2: Navigate downward from matched axioms
        seen = set()
        for axiom_hier, axiom_node, base_score in sorted(axiom_matches, key=lambda x: -x[2]):
            seen.add(axiom_hier)
            results.append({
                "node_type": axiom_node.node_type,
                "hierarchy": axiom_hier,
                "content": axiom_node.content,
                "score": base_score * 1.2,  # Axiom bonus
                "associations": axiom_node.associations,
            })
            
            # Navigate hierarchy children
            self._traverse_hierarchy(axiom_hier, keywords, results, max_depth, seen)
        
        # Phase 3: Direct search across ALL node types (fallback + complement)
        for hierarchy, node in self.nodes.items():
            if hierarchy in seen:
                continue
            score = self._relevance_score(node, keywords)
            if score > 0.1:
                seen.add(hierarchy)
                results.append({
                    "node_type": node.node_type,
                    "hierarchy": hierarchy,
                    "content": node.content,
                    "score": score,
                    "associations": node.associations,
                })
        
        # Phase 4: Follow lateral associations
        assoc_set = set()
        for r in results[:10]:  # Limit association search to top-10 hits
            hier = r.get("hierarchy", "")
            if hier in self.nodes:
                for assoc_hier in self.nodes[hier].associations:
                    if assoc_hier not in assoc_set and assoc_hier in self.nodes:
                        assoc_set.add(assoc_hier)
                        assoc_node = self.nodes[assoc_hier]
                        assoc_score = self._relevance_score(assoc_node, keywords) * 0.7
                        if assoc_score > 0.1:
                            results.append({
                                "node_type": assoc_node.node_type,
                                "hierarchy": assoc_hier,
                                "content": assoc_node.content,
                                "score": assoc_score,
                                "associations": assoc_node.associations,
                            })
        
        # Sort by score descending
        results.sort(key=lambda x: -x["score"])
        results = results[:20]  # Limit to top-20

        # Telemetría: registrar el recall (consulta) — el usage lo registra
        # el consumidor con mark_used() cuando integra los resultados.
        self.telemetry_data["recalls"] += 1
        self.telemetry_data["recall_log"].append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "query": query_text[:100],
            "hits": [r["hierarchy"] for r in results[:5]],
        })
        self.telemetry_data["recall_log"] = self.telemetry_data["recall_log"][-200:]
        self._save_telemetry()

        return results

    def retrieve_by_hierarchy(self, hierarchy_prefix: str) -> list[dict]:
        """Retrieve all nodes under a hierarchy path."""
        results = []
        for hierarchy, node in self.nodes.items():
            if hierarchy.startswith(hierarchy_prefix):
                results.append(node.to_dict())
        return results

    def retrieve_axioms(self, domain: str = None) -> list[dict]:
        """List all axioms, optionally filtered by domain."""
        results = []
        for hierarchy, node in self.nodes.items():
            if node.node_type == "AXIOM":
                if domain is None or hierarchy.startswith(domain):
                    results.append(node.to_dict())
        return results

    def tree(self, domain: str = None) -> dict:
        """Visualize hierarchical tree."""
        tree = {}
        for hierarchy, node in self.nodes.items():
            parts = hierarchy.split(".")
            current = tree
            for i, part in enumerate(parts):
                if part not in current:
                    sub_hier = ".".join(parts[:i+1])
                    confidence = node.confidence if sub_hier == hierarchy else None
                    current[part] = {
                        "_node": {
                            "type": node.node_type if sub_hier == hierarchy else None,
                            "confidence": confidence,
                            "content_preview": node.content[:50] if sub_hier == hierarchy else None,
                        },
                    }
                current = current[part]
        return tree

    def stats(self) -> dict:
        """Return engine statistics."""
        counts = {"AXIOM": 0, "CONCEPT": 0, "INSTANCE": 0}
        for node in self.nodes.values():
            if node.node_type in counts:
                counts[node.node_type] += 1
        return {
            "total_nodos": len(self.nodes),
            "axiomas": counts["AXIOM"],
            "conceptos": counts["CONCEPT"],
            "instancias": counts["INSTANCE"],
            "memory_dir": str(self.memory_dir),
        }

    def save(self, path: str = None):
        """Save memory to JSON file."""
        if path is None:
            path = str(self.memory_dir / "unified_memory.json")
        
        data = {hierarchy: node.to_dict() for hierarchy, node in self.nodes.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str = None):
        """Load memory from JSON file."""
        if path is None:
            path = str(self.memory_dir / "unified_memory.json")
        
        if not os.path.exists(path):
            return False
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for hierarchy, node_data in data.items():
            self.nodes[hierarchy] = MemoryNode.from_dict(node_data)
        return True

    # --- Internal methods ---

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from query text."""
        text_lower = text.lower()
        # Remove common stopwords and punctuation
        stopwords = {"el", "la", "los", "las", "un", "una", "de", "en", "que", 
                     "con", "por", "para", "es", "del", "al", "y", "o", "como",
                     "what", "how", "the", "a", "an", "is", "are", "and", "or"}
        words = re.findall(r'\b\w{2,}\b', text_lower)
        return [w for w in words if w not in stopwords]

    def _relevance_score(self, node: MemoryNode, keywords: list[str]) -> float:
        """Calculate relevance score between node and query keywords."""
        content_lower = node.content.lower()
        hierarchy_lower = node.hierarchy.lower()
        tags_lower = [t.lower() for t in node.tags]
        
        score = 0.0
        # Content match (highest weight)
        for kw in keywords:
            if kw in content_lower:
                score += 2.0
            if kw in hierarchy_lower:
                score += 1.5
            for tag in tags_lower:
                if kw in tag:
                    score += 1.0
        
        return score / max(len(keywords), 1)

    def _traverse_hierarchy(self, axiom_hier: str, keywords: list[str], 
                           results: list, max_depth: int, seen: set = None):
        """Traverse hierarchy downward from axiom."""
        if seen is None:
            seen = set()
        prefix = axiom_hier.split(".")
        for hierarchy, node in self.nodes.items():
            if hierarchy in seen:
                continue
            parts = hierarchy.split(".")
            # Check if this node is a child of the axiom
            if len(parts) > len(prefix) and all(
                p == pp for p, pp in zip(parts[:len(prefix)], prefix)
            ):
                depth = len(parts) - len(prefix)
                if depth <= max_depth:
                    score = self._relevance_score(node, keywords) * (1.0 / (depth + 1))
                    if score > 0.05:
                        seen.add(hierarchy)
                        results.append({
                            "node_type": node.node_type,
                            "hierarchy": hierarchy,
                            "content": node.content,
                            "score": score,
                            "associations": node.associations,
                        })
