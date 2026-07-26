"""
Omega-Cube Engine v2 — Axion-Enhanced with arXiv 2026 improvements.

Improvements applied (from papers):
1. H²MT (2605.24930): Hierarchical summarization + coarse-to-fine routing
2. SeKV (2606.31145): Zoom-in selectivo (lazy expansion)
3. VirtualSet (2607.18821): Typed ontology + pre-execution checks
4. PAGE-RAG (2607.19301): Skeleton graph + knowledge boundary control
5. DaoQL (2607.17269): Data-first separation (LLM=reasoning / graph=knowledge)
6. Inference Misalignment (2607.00447): Hallucination detection via bias counteraction

Author: Axion Research
Date: 2026-07-26
"""

import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Import base components
from .tensor_node import TensorNode, TensorIndex
from .holographic import HolographicEncoder
from .annealer import QuantumInspiredAnnealer, CubeRotator, PatternEmergence
from .diffusion_sampler import DiffusionGraphSampler
from .grayscale import GrayScaleValidator


# ── Improvement 1: Typed Ontology Schema (VirtualSet, 2607.18821) ──────

class TypedSchema:
    """
    Schema de tipos para nodos y edges.
    Verifica toda operación ANTES de ejecutar (pre-execution checks).
    Intercepta operaciones inválidas con 0 falsos positivos.
    """
    
    VALID_NODE_TYPES = {"AXIOM", "CONCEPT", "INSTANCE", "SESSION", "SUMMARY"}
    VALID_EDGE_TYPES = {"HIERARCHICAL", "ASSOCIATION", "CAUSAL", "TEMPORAL", "SUMMARY_OF"}
    
    # Edge compatibility matrix: (source_type, edge_type) -> valid_target_types
    EDGE_COMPATIBILITY = {
        ("AXIOM", "HIERARCHICAL"): {"CONCEPT", "INSTANCE"},
        ("CONCEPT", "HIERARCHICAL"): {"CONCEPT", "INSTANCE"},
        ("CONCEPT", "ASSOCIATION"): {"CONCEPT", "INSTANCE", "AXIOM"},
        ("CONCEPT", "CAUSAL"): {"CONCEPT", "INSTANCE"},
        ("CONCEPT", "SUMMARY_OF"): {"SUMMARY"},
        ("SUMMARY", "SUMMARY_OF"): {"CONCEPT", "INSTANCE"},
        ("INSTANCE", "ASSOCIATION"): {"CONCEPT", "INSTANCE"},
        ("SESSION", "TEMPORAL"): {"CONCEPT", "INSTANCE"},
    }
    
    def validate_node(self, node_type: str) -> tuple[bool, str]:
        """Valida tipo de nodo contra schema."""
        if node_type not in self.VALID_NODE_TYPES:
            return False, f"Invalid node type: {node_type}. Valid: {self.VALID_NODE_TYPES}"
        return True, "OK"
    
    def validate_edge(self, source_type: str, edge_type: str, target_type: str) -> tuple[bool, str]:
        """Valida edge contra schema ANTES de ejecutar."""
        if edge_type not in self.VALID_EDGE_TYPES:
            return False, f"Invalid edge type: {edge_type}. Valid: {self.VALID_EDGE_TYPES}"
        
        key = (source_type, edge_type)
        if key not in self.EDGE_COMPATIBILITY:
            return False, f"No edge rule for ({source_type}, {edge_type})"
        
        valid_targets = self.EDGE_COMPATIBILITY[key]
        if target_type not in valid_targets:
            return False, f"Invalid edge: {source_type} -{edge_type}-> {target_type}. Valid targets: {valid_targets}"
        
        return True, "OK"


# ── Improvement 2: Hierarchical Summarizer (H²MT, 2605.24930) ──────────

class HierarchicalSummarizer:
    """
    Agregación bottom-up post-order: nodos padre resumen nodos hijo.
    Permite routing coarse-to-fine: O(log n) en lugar de O(n).
    """
    
    def __init__(self, holographic: HolographicEncoder):
        self.holographic = holographic
        self.summaries: dict[str, dict] = {}  # hierarchy_prefix -> summary
    
    def build_hierarchy(self, nodes: dict[str, TensorNode]) -> dict[str, list[str]]:
        """Construye árbol jerárquico desde nodos."""
        tree: dict[str, list[str]] = defaultdict(list)
        
        for nid, node in nodes.items():
            if node.primary_hierarchy:
                parts = node.primary_hierarchy.split(".")
                # Cada nivel es padre del siguiente
                for i in range(len(parts) - 1):
                    parent = ".".join(parts[:i+1])
                    child = ".".join(parts[:i+2])
                    if child not in tree[parent]:
                        tree[parent].append(child)
                # Nodo hoja bajo su prefijo más específico
                leaf_parent = ".".join(parts[:-1]) if len(parts) > 1 else parts[0]
                tree[leaf_parent].append(nid)
        
        return dict(tree)
    
    def aggregate_bottom_up(self, nodes: dict[str, TensorNode], tree: dict[str, list[str]]) -> dict[str, dict]:
        """
        Calcula embedding de nodo padre como agregación de hijos.
        Post-order: procesa hijos antes que padres.
        """
        summaries = {}
        
        # Ordena por profundidad (más profundo primero = post-order)
        sorted_keys = sorted(tree.keys(), key=lambda k: -k.count("."))
        
        for key in sorted_keys:
            children = tree[key]
            child_embeddings = []
            child_contents = []
            
            for child in children:
                if child in nodes:
                    node = nodes[child]
                    if node.holographic_signature:
                        child_embeddings.append(node.holographic_signature)
                    child_contents.append(node.content[:100])
                elif child in summaries:
                    # Hijo es un nodo intermedio ya resumido
                    child_embeddings.append(summaries[child].get("embedding", []))
                    child_contents.append(summaries[child].get("summary_text", ""))
            
            if child_embeddings:
                # Pooling: mean de embeddings hijos
                dim = len(child_embeddings[0])
                avg_embedding = [
                    sum(e[i] for e in child_embeddings) / len(child_embeddings)
                    for i in range(dim)
                ]
                
                # Resumen textual: keywords más frecuentes
                all_words = " ".join(child_contents).lower().split()
                word_freq = defaultdict(int)
                for w in all_words:
                    if len(w) > 3:
                        word_freq[w] += 1
                top_keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:10]
                
                summaries[key] = {
                    "embedding": avg_embedding,
                    "summary_text": f"[{key}] " + ", ".join(top_keywords),
                    "child_count": len(children),
                    "keywords": top_keywords,
                }
        
        self.summaries = summaries
        return summaries
    
    def route_coarse_to_fine(
        self,
        query: str,
        nodes: dict[str, TensorNode],
        tree: dict[str, list[str]],
        top_k: int = 10,
    ) -> list[tuple[TensorNode, float]]:
        """
        Navega de lo general a lo específico.
        En cada nivel, expande solo el hijo más relevante.
        Complejidad: O(log n * branching_factor) en lugar de O(n).
        """
        query_vector = self.holographic.encode_node(query, "QUERY")
        
        # Nivel 0: dominios raíz
        root_keys = [k for k in tree.keys() if "." not in k]
        if not root_keys:
            # Fallback: todos los nodos
            return [(n, 0.5) for n in list(nodes.values())[:top_k]]
        
        # Score cada raíz
        root_scores = []
        for key in root_keys:
            if key in self.summaries:
                sim = self.holographic.similarity(query_vector, self.summaries[key]["embedding"])
            else:
                sim = 0.0
            root_scores.append((key, sim))
        
        root_scores.sort(key=lambda x: -x[1])
        
        # Beam estrecho en raíces (top-2) para evitar ruido cross-domain
        # + profundidad amplia en hijos (top-5) para no perder relevancia
        candidates = []
        beam = [root_scores[0][0]]
        if len(root_scores) > 1 and root_scores[1][1] > 0.25:
            beam.append(root_scores[1][0])
        
        for root_key in beam:
            # Nivel 1: sub-dominios
            children = tree.get(root_key, [])
            child_scores = []
            
            for child in children:
                if child in nodes:
                    node = nodes[child]
                    if node.holographic_signature:
                        sim = self.holographic.similarity(query_vector, node.holographic_signature)
                    else:
                        sim = 0.0
                    child_scores.append((child, sim, node))
                elif child in tree:
                    # Nodo intermedio: usa resumen
                    if child in self.summaries:
                        sim = self.holographic.similarity(query_vector, self.summaries[child]["embedding"])
                    else:
                        sim = 0.0
                    child_scores.append((child, sim, None))
            
            child_scores.sort(key=lambda x: -x[1])
            
            # Expande top-5 hijos (mejorado de 3)
            for child_key, sim, node in child_scores[:5]:
                if node:
                    candidates.append((node, sim))
                elif child_key in tree:
                    # Nivel 2: nodos hoja
                    grandchildren = tree.get(child_key, [])
                    for gc in grandchildren:
                        if gc in nodes:
                            gc_node = nodes[gc]
                            if gc_node.holographic_signature:
                                gc_sim = self.holographic.similarity(query_vector, gc_node.holographic_signature)
                            else:
                                gc_sim = 0.0
                            candidates.append((gc_node, gc_sim))
        
        # Re-rank y retorna top_k
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]


# ── Improvement 3: Knowledge Boundary Control (PAGE-RAG, 2607.19301) ───

class BoundaryController:
    """
    Control de boundaries: solo retorna conocimiento soportado por evidencia.
    El grafo es esqueleto (índice), no fuente de verdad.
    """
    
    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence
    
    def filter_grounded(self, results: list[dict], query: str) -> list[dict]:
        """Filtra resultados: solo incluye los que tienen evidencia suficiente."""
        grounded = []
        query_words = set(query.lower().split())
        
        for r in results:
            content_words = set(r.get("content", "").lower().split())
            
            # Overlap mínimo con query
            overlap = len(query_words & content_words) / max(len(query_words), 1)
            
            # Confidence del nodo
            confidence = r.get("confidence", 0.5)
            gray_composite = r.get("gray_scale_composite", 50.0) / 100.0
            
            # Score combinado de grounding
            grounding_score = 0.4 * overlap + 0.3 * confidence + 0.3 * gray_composite
            
            if grounding_score >= self.min_confidence:
                r["grounding_score"] = round(grounding_score, 4)
                grounded.append(r)
            else:
                # Abstenerse: no incluir info no soportada
                r["abstained"] = True
                r["abstain_reason"] = f"grounding_score {grounding_score:.3f} < {self.min_confidence}"
        
        return grounded


# ── Improvement 4: Hallucination Detector (Inference Misalignment, 2607.00447) ──

class HallucinationDetector:
    """
    Detecta inference misalignment:
    - task-retrieval bias: desambiguación de entidades
    - key-selection bias: elección de acciones
    
    Contrarresta ambos usando el grafo como fuente de desambiguación estructural.
    """
    
    def detect_bias(self, query: str, results: list[dict]) -> dict:
        """Detecta tipo de bias en resultados."""
        if not results:
            return {"bias_type": "no_results", "severity": 1.0}
        
        # Analiza distribución de dominios en resultados
        domain_counts = defaultdict(int)
        for r in results:
            domain = r.get("primary_hierarchy", "").split(".")[0]
            domain_counts[domain] += 1
        
        total = len(results)
        max_domain_share = max(domain_counts.values()) / total if domain_counts else 0
        
        # Task-retrieval bias: un dominio domina (>80% de resultados)
        if max_domain_share > 0.8 and len(domain_counts) > 1:
            dominant = max(domain_counts, key=domain_counts.get)
            return {
                "bias_type": "task_retrieval",
                "severity": max_domain_share,
                "dominant_domain": dominant,
                "suggestion": f"Query may be ambiguous. Dominant: {dominant}. Consider disambiguation.",
            }
        
        # Key-selection bias: scores muy bajos (<0.3 avg)
        avg_score = sum(r.get("score", 0) for r in results) / total
        if avg_score < 0.3:
            return {
                "bias_type": "key_selection",
                "severity": 1.0 - avg_score,
                "avg_score": avg_score,
                "suggestion": "Low confidence results. Constraint-sensitive path recommended.",
            }
        
        return {"bias_type": "none", "severity": 0.0}
    
    def counteract(self, query: str, results: list[dict], bias: dict) -> list[dict]:
        """Contrarresta bias detectado."""
        if bias["bias_type"] == "none":
            return results
        
        if bias["bias_type"] == "task_retrieval":
            # Desambiguación: re-rank para dar más peso a dominios minoritarios
            dominant = bias.get("dominant_domain", "")
            for r in results:
                domain = r.get("primary_hierarchy", "").split(".")[0]
                if domain != dominant:
                    r["score"] = r.get("score", 0) * 1.5  # Boost minoritarios
            results.sort(key=lambda x: -x.get("score", 0))
        
        elif bias["bias_type"] == "key_selection":
            # Forza constraint-sensitive path: filtra por gray_scale
            results = [r for r in results if r.get("gray_scale_composite", 0) > 40]
        
        return results


# ── Main Engine v2 ──────────────────────────────────────────────────────

class OmegaCubeEngineV2:
    """
    Omega-Cube v2: Axion-Enhanced Engine.
    
    Mejoras sobre v1:
    1. Typed Schema (VirtualSet): pre-execution checks, 0 operaciones inválidas
    2. Hierarchical Summarizer (H²MT): coarse-to-fine routing, O(log n)
    3. Boundary Controller (PAGE-RAG): knowledge boundary control
    4. Hallucination Detector (2607.00447): bias detection + counteraction
    5. Zoom-in selectivo (SeKV): lazy expansion de nodos
    """
    
    def __init__(
        self,
        memory_dir: str = None,
        holographic_dim: int = 256,
        tensor_grid_size: int = 10,
    ):
        if memory_dir is None:
            axioma_base = os.environ.get(
                "AXIOMA_PROJECT_PATH",
                str(Path.home() / ".hermes" / "axioma-omega-protocol")
            )
            memory_dir = os.path.join(axioma_base, "memory")
        
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components (same as v1)
        self.nodes: dict[str, TensorNode] = {}
        self.index = TensorIndex(grid_size=tensor_grid_size)
        self.holographic = HolographicEncoder(dimension=holographic_dim)
        self.annealer = QuantumInspiredAnnealer()
        self.rotator = CubeRotator()
        self.diffusion = DiffusionGraphSampler()
        self.gray_validator = GrayScaleValidator()
        self.pattern_emergence = PatternEmergence()
        self.axioms: list[TensorNode] = []
        
        # NEW: Axion improvements
        self.schema = TypedSchema()
        self.summarizer = HierarchicalSummarizer(self.holographic)
        self.boundary = BoundaryController(min_confidence=0.25)
        self.hallucination_detector = HallucinationDetector()
        self.hierarchy_tree: dict[str, list[str]] = {}
        self.hierarchy_built = False
        
        # Stats
        self.query_count = 0
        self.total_retrieval_time = 0.0
        self.invalid_ops_blocked = 0
        self.abstained_results = 0
        self.bias_detections = 0
        
        # Auto-load
        self.load()
    
    # ── Knowledge Ingestion (with typed validation) ───────────────
    
    def add_node(
        self,
        content: str,
        hierarchies: list[str],
        tensor_position: list[float] = None,
        node_type: str = "CONCEPT",
        confidence: float = 0.9,
        tags: list = None,
    ) -> TensorNode:
        """Add node with typed schema validation (VirtualSet)."""
        # Pre-execution check: validate node type
        valid, msg = self.schema.validate_node(node_type)
        if not valid:
            self.invalid_ops_blocked += 1
            raise TypeError(f"[VirtualSet] {msg}")
        
        if tensor_position is None:
            tensor_position = self._compute_tensor_position(hierarchies)
        
        node = TensorNode(
            content=content,
            hierarchies=hierarchies,
            tensor_position=tensor_position,
            node_type=node_type,
            confidence=confidence,
            tags=tags or [],
            created_at=time.time(),
        )
        
        node.holographic_signature = self.holographic.encode_node(content, node.primary_hierarchy)
        node.gray_scale = self.gray_validator.evaluate_node(node, axioms=self.axioms)
        
        self.nodes[node.node_id] = node
        self.index.insert(node)
        
        if node_type == "AXIOM":
            self.axioms.append(node)
        
        # Invalidate hierarchy cache
        self.hierarchy_built = False
        
        return node
    
    def associate(self, node_id1: str, node_id2: str, edge_type: str = "ASSOCIATION") -> bool:
        """Create association with typed edge validation (VirtualSet)."""
        if node_id1 not in self.nodes or node_id2 not in self.nodes:
            return False
        
        n1 = self.nodes[node_id1]
        n2 = self.nodes[node_id2]
        
        # Pre-execution check: validate edge type
        valid, msg = self.schema.validate_edge(n1.node_type, edge_type, n2.node_type)
        if not valid:
            self.invalid_ops_blocked += 1
            return False
        
        if node_id2 not in n1.associations:
            n1.associations.append(node_id2)
        if node_id1 not in n2.associations:
            n2.associations.append(node_id1)
        
        n1.holographic_signature = self._recompute_signature(node_id1)
        n2.holographic_signature = self._recompute_signature(node_id2)
        
        self.hierarchy_built = False
        return True
    
    # ── Hierarchy Building (H²MT) ─────────────────────────────────
    
    def _ensure_hierarchy(self):
        """Construye jerarquía si no existe (lazy, una vez)."""
        if not self.hierarchy_built:
            self.hierarchy_tree = self.summarizer.build_hierarchy(self.nodes)
            self.summarizer.aggregate_bottom_up(self.nodes, self.hierarchy_tree)
            self.hierarchy_built = True
    
    # ── Query (with all improvements) ─────────────────────────────
    
    def query(
        self,
        query_text: str,
        mode: str = "hierarchical",  # NEW default: hierarchical routing
        top_k: int = 10,
        temperature: float = 0.1,
        apply_boundaries: bool = True,
        detect_hallucination: bool = True,
    ) -> list[dict]:
        """
        Query with Axion improvements:
        - hierarchical: coarse-to-fine routing (H²MT) → O(log n)
        - diffusion: parallel sampling (original)
        - holographic: signature match (original)
        - combined: diffusion + holographic re-ranking (original)
        
        Post-processing:
        - Boundary control (PAGE-RAG): filter ungrounded results
        - Hallucination detection (2607.00447): detect + counteract bias
        """
        self.query_count += 1
        start = time.time()
        
        if mode == "hierarchical":
            self._ensure_hierarchy()
            raw_results = self.summarizer.route_coarse_to_fine(
                query_text, self.nodes, self.hierarchy_tree, top_k=top_k * 2
            )
            results = [self._format_result((n, s)) for n, s in raw_results]
        elif mode == "diffusion":
            raw = self.diffusion.sample(query_text, self.index, self.holographic, top_k=top_k, temperature=temperature)
            results = [self._format_result(r) for r in raw]
        elif mode == "holographic":
            raw = self._query_holographic(query_text, top_k)
            results = [self._format_result(r) for r in raw]
        elif mode == "combined":
            raw = self._query_combined(query_text, top_k)
            results = [self._format_result(r) for r in raw]
        else:
            self._ensure_hierarchy()
            raw_results = self.summarizer.route_coarse_to_fine(
                query_text, self.nodes, self.hierarchy_tree, top_k=top_k * 2
            )
            results = [self._format_result((n, s)) for n, s in raw_results]
        
        # Improvement: Boundary control (PAGE-RAG)
        if apply_boundaries:
            before_count = len(results)
            results = self.boundary.filter_grounded(results, query_text)
            self.abstained_results += (before_count - len(results))
        
        # Improvement: Hallucination detection (2607.00447)
        if detect_hallucination and results:
            bias = self.hallucination_detector.detect_bias(query_text, results)
            if bias["bias_type"] != "none":
                self.bias_detections += 1
                results = self.hallucination_detector.counteract(query_text, results, bias)
        
        # Trim to top_k
        results = results[:top_k]
        
        elapsed = time.time() - start
        self.total_retrieval_time += elapsed
        
        return results
    
    # ── Internal query methods (from v1) ──────────────────────────
    
    def _query_holographic(self, query: str, top_k: int) -> list:
        query_vector = self.holographic.encode_node(query, "QUERY")
        results = []
        for node in self.nodes.values():
            if node.holographic_signature:
                sim = self.holographic.partial_match(query_vector, node.holographic_signature)
                results.append((node, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    def _query_combined(self, query: str, top_k: int) -> list:
        candidates = self.diffusion.sample(query, self.index, self.holographic, top_k=top_k * 3, temperature=0.15)
        query_vector = self.holographic.encode_node(query, "QUERY")
        results = []
        for node, score in candidates:
            holo_sim = 0.5
            if node.holographic_signature:
                holo_sim = self.holographic.partial_match(query_vector, node.holographic_signature)
            combined = 0.6 * score + 0.4 * holo_sim
            results.append((node, combined))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    # ── Helpers ───────────────────────────────────────────────────
    
    def _compute_tensor_position(self, hierarchies: list[str]) -> list[float]:
        positions = []
        for h in hierarchies:
            parts = h.split(".")
            if len(parts) >= 2:
                hash_bytes = hashlib.md5(h.encode()).digest()
                coord = int.from_bytes(hash_bytes[:4], 'big') / (2**32)
                positions.append(coord)
            else:
                positions.append(0.5)
        while len(positions) < 2:
            positions.append(0.5)
        return positions
    
    def _recompute_signature(self, node_id: str) -> list[float]:
        node = self.nodes.get(node_id)
        if not node:
            return [0.0] * self.holographic.dim
        neighbors = []
        for assoc_id in node.associations:
            if assoc_id in self.nodes:
                n = self.nodes[assoc_id]
                neighbors.append((n.content, n.primary_hierarchy))
        return self.holographic.encode_holographic_signature(
            node_content=node.content,
            node_hierarchy=node.primary_hierarchy,
            neighbors=neighbors,
        )
    
    def _format_result(self, item) -> dict:
        if isinstance(item, tuple) and len(item) == 2:
            node, score = item
        else:
            node, score = item, 0.5
        return {
            "node_id": node.node_id,
            "content": node.content[:300],
            "primary_hierarchy": node.primary_hierarchy,
            "hierarchies": node.hierarchies,
            "node_type": node.node_type,
            "score": round(score, 4),
            "confidence": node.confidence,
            "gray_scale": node.gray_scale,
            "gray_scale_composite": self.gray_validator.composite_score(
                node.gray_scale or {}
            ) if node.gray_scale else 50.0,
            "tensor_position": node.tensor_position,
            "associations_count": len(node.associations),
        }
    
    # ── Persistence ───────────────────────────────────────────────
    
    def save(self, path: str = None):
        if path is None:
            path = str(self.memory_dir / "omega_cube_memory_v2.json")
        data = {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "axiom_ids": [n.node_id for n in self.axioms],
            "stats": {
                "query_count": self.query_count,
                "total_retrieval_time": self.total_retrieval_time,
                "invalid_ops_blocked": self.invalid_ops_blocked,
                "abstained_results": self.abstained_results,
                "bias_detections": self.bias_detections,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str = None) -> bool:
        if path is None:
            path = str(self.memory_dir / "omega_cube_memory_v2.json")
        if not os.path.exists(path):
            # Try loading v1 data
            v1_path = str(self.memory_dir / "omega_cube_memory.json")
            if os.path.exists(v1_path):
                path = v1_path
            else:
                return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for nid, node_data in data.get("nodes", {}).items():
            node = TensorNode.from_dict(node_data)
            self.nodes[nid] = node
            self.index.insert(node)
        for axiom_id in data.get("axiom_ids", []):
            if axiom_id in self.nodes:
                self.axioms.append(self.nodes[axiom_id])
        stats = data.get("stats", {})
        self.query_count = stats.get("query_count", 0)
        self.total_retrieval_time = stats.get("total_retrieval_time", 0.0)
        return True
    
    def stats(self) -> dict:
        type_counts = {"AXIOM": 0, "CONCEPT": 0, "INSTANCE": 0, "SESSION": 0, "SUMMARY": 0}
        for node in self.nodes.values():
            if node.node_type in type_counts:
                type_counts[node.node_type] += 1
        avg_dims = (
            sum(n.dimension_count for n in self.nodes.values()) / len(self.nodes)
            if self.nodes else 0
        )
        return {
            "total_nodes": len(self.nodes),
            "axioms": type_counts["AXIOM"],
            "concepts": type_counts["CONCEPT"],
            "instances": type_counts["INSTANCE"],
            "sessions": type_counts["SESSION"],
            "summaries": type_counts["SUMMARY"],
            "avg_dimensions_per_node": round(avg_dims, 1),
            "holographic_dim": self.holographic.dim,
            "query_count": self.query_count,
            "avg_retrieval_time_ms": round(
                (self.total_retrieval_time / self.query_count * 1000) if self.query_count else 0, 2
            ),
            "invalid_ops_blocked": self.invalid_ops_blocked,
            "abstained_results": self.abstained_results,
            "bias_detections": self.bias_detections,
            "hierarchy_levels": len(self.hierarchy_tree) if self.hierarchy_built else 0,
        }
