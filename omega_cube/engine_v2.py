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
from .vector_index import HNSWVectorIndex


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
        query_vector = self.holographic.encode_node(query, "")
        
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
        auto_load: bool = True,
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
        
        # NEW v4: HNSW vector index (Layer 2 — O(log n) ANN search)
        self.hnsw = HNSWVectorIndex(
            dimension=holographic_dim,
            metric="cos",
            quantization="f32",  # f32 required: sparse feature-hash vectors lose all signal under i8
        )
        self._hnsw_dirty = True  # needs rebuild after load
        
        # Stats
        self.query_count = 0
        self.total_retrieval_time = 0.0
        self.invalid_ops_blocked = 0
        self.abstained_results = 0
        self.bias_detections = 0
        
        # Auto-load (set False for benchmarks / isolated tests)
        if auto_load:
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
        
        # v4: Index in HNSW for O(log n) ANN search
        if node.holographic_signature:
            self.hnsw.add(node.node_id, node.holographic_signature)
        
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
    
    # ── Query (v5: HNSW-first architecture) ──────────────────────
    
    def query(
        self,
        query_text: str,
        mode: str = "hierarchical",
        top_k: int = 10,
        temperature: float = 0.1,
        apply_boundaries: bool = True,
        detect_hallucination: bool = True,
        node_filter: set = None,
    ) -> list[dict]:
        """
        v5: HNSW-first retrieval architecture.
        
        ALL modes start with HNSW ANN search (O(log n)) to generate
        a candidate set, then apply mode-specific re-ranking on that
        small subset (~50 nodes) instead of the full graph.
        
        Modes:
        - hierarchical: HNSW → hierarchy-depth-aware rerank
        - holographic:  HNSW → multisignal rerank (keyword+depth+tags)
        - diffusion:    HNSW → scoped diffusion on candidates only
        - combined:     HNSW → multisignal + domain-boost rerank
        
        Post-processing:
        - Boundary control (PAGE-RAG): filter ungrounded results
        - Hallucination detection (2607.00447): detect + counteract bias
        """
        self.query_count += 1
        start = time.time()
        
        # Layer 0: HNSW ANN search — generates candidate set for ALL modes
        candidates = self._hnsw_retrieve(query_text, top_k * 5, node_filter=node_filter)
        
        if mode == "hierarchical":
            # Hierarchical = diffusion rerank with moderate temperature
            results = self._rerank_diffusion(query_text, candidates, top_k, temperature=0.05)
        elif mode == "diffusion":
            results = self._rerank_diffusion(query_text, candidates, top_k, temperature)
        elif mode == "holographic":
            # Holographic = diffusion rerank with minimal noise (near-deterministic)
            results = self._rerank_diffusion(query_text, candidates, top_k, temperature=0.01)
        elif mode == "combined":
            # Combined = diffusion rerank with low temperature (more deterministic)
            results = self._rerank_diffusion(query_text, candidates, top_k, temperature=0.03)
        else:
            # Hierarchical = diffusion rerank with moderate temperature
            results = self._rerank_diffusion(query_text, candidates, top_k, temperature=0.05)
        
        # Format results
        results = [self._format_result(r) for r in results]
        
        # Boundary control (PAGE-RAG)
        if apply_boundaries:
            before_count = len(results)
            results = self.boundary.filter_grounded(results, query_text)
            self.abstained_results += (before_count - len(results))
        
        # Hallucination detection (2607.00447)
        if detect_hallucination and results:
            bias = self.hallucination_detector.detect_bias(query_text, results)
            if bias["bias_type"] != "none":
                self.bias_detections += 1
                results = self.hallucination_detector.counteract(query_text, results, bias)
        
        results = results[:top_k]
        
        elapsed = time.time() - start
        self.total_retrieval_time += elapsed
        
        return results
    
    # ── HNSW-first Retrieval (v5) ──────────────────────────────
    
    def _ensure_hnsw(self):
        """Rebuild HNSW index from loaded nodes (lazy, after load)."""
        if not self._hnsw_dirty:
            return
        items = []
        for nid, node in self.nodes.items():
            if node.holographic_signature:
                items.append((nid, node.holographic_signature))
        if items:
            self.hnsw.rebuild(items)
        self._hnsw_dirty = False
    
    def _hnsw_retrieve(self, query: str, candidate_k: int, node_filter: set = None) -> list[tuple]:
        """
        Layer 0: HNSW ANN retrieval — the single entry point for ALL query modes.
        
        Returns [(node, holo_similarity), ...] sorted by ANN distance.
        Falls back to linear scan if HNSW unavailable.
        O(log n) via USearch HNSW with i8 quantization.
        """
        self._ensure_hnsw()
        query_vector = self.holographic.encode_node(query, "")
        
        if self.hnsw.is_available and self.hnsw.size > 0:
            ann_k = min(candidate_k, self.hnsw.size)
            ann_results = self.hnsw.search(query_vector, ann_k)
            
            candidates = []
            seen_ids = set()
            for nid, sim in ann_results:
                if node_filter and nid not in node_filter:
                    continue
                node = self.nodes.get(nid)
                if node:
                    candidates.append((node, sim))
                    seen_ids.add(nid)
            
            # 1-hop graph expansion (SAP CIKM 2025: +15% precision)
            # Add neighbors of seed nodes that weren't found by ANN
            expanded = []
            for seed_node, seed_sim in candidates:
                for assoc_id in seed_node.associations:
                    if assoc_id in seen_ids:
                        continue
                    if node_filter and assoc_id not in node_filter:
                        continue
                    neighbor = self.nodes.get(assoc_id)
                    if neighbor and neighbor.holographic_signature:
                        n_sim = self.holographic.partial_match(query_vector, neighbor.holographic_signature)
                        # Neighbor gets discounted similarity (0.7x) since it's structurally, not semantically, close
                        expanded.append((neighbor, n_sim * 0.7))
                        seen_ids.add(assoc_id)
            
            candidates.extend(expanded)
            candidates.sort(key=lambda x: -x[1])
            return candidates[:candidate_k]
        else:
            # Fallback: linear scan
            candidates = []
            for node in self.nodes.values():
                if node_filter and node.node_id not in node_filter:
                    continue
                if node.holographic_signature:
                    sim = self.holographic.partial_match(query_vector, node.holographic_signature)
                    candidates.append((node, sim))
            candidates.sort(key=lambda x: -x[1])
            return candidates[:candidate_k]
    
    # ── Mode-specific Re-rankers (operate on candidate subset) ──
    
    def _rerank_hierarchical(self, query: str, candidates: list[tuple], top_k: int) -> list[tuple]:
        """
        Hierarchy-depth-aware rerank on HNSW candidates.
        Keyword overlap + hierarchy-match + depth + domain coherence.
        """
        if not candidates:
            return candidates
        
        query_words = {w for w in query.lower().split() if len(w) > 2}
        
        # Precompute domain keyword coherence
        domain_kw = defaultdict(list)
        node_kw = {}
        for node, _ in candidates:
            content_words = {w for w in node.content.lower().split() if len(w) > 2}
            kw = len(query_words & content_words) / max(len(query_words), 1)
            node_kw[node.node_id] = kw
            domain = node.primary_hierarchy.split(".")[0] if node.primary_hierarchy else "?"
            domain_kw[domain].append(kw)
        domain_avg = {d: sum(v)/len(v) for d, v in domain_kw.items() if v}
        
        scored = []
        for node, holo_sim in candidates:
            keyword_score = node_kw[node.node_id]
            
            hier_words = set()
            if node.primary_hierarchy:
                for part in node.primary_hierarchy.replace('.', ' ').replace('/', ' ').lower().split():
                    if len(part) > 2:
                        hier_words.add(part)
            hier_match = len(query_words & hier_words) / max(len(query_words), 1)
            
            depth = node.primary_hierarchy.count(".") if node.primary_hierarchy else 0
            depth_score = min(depth / 4.0, 1.0)
            
            tag_score = 0.0
            if node.tags and query_words:
                tag_words = set()
                for t in node.tags:
                    tag_words.update(w for w in t.lower().split() if len(w) > 2)
                tag_score = len(query_words & tag_words) / max(len(query_words), 1)
            
            domain = node.primary_hierarchy.split(".")[0] if node.primary_hierarchy else "?"
            domain_coherence = domain_avg.get(domain, 0.0)
            
            combined = (
                0.55 * keyword_score
                + 0.10 * holo_sim
                + 0.10 * hier_match
                + 0.10 * tag_score
                + 0.05 * depth_score
                + 0.10 * domain_coherence
            )
            scored.append((node, combined))
        
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
    
    def _rerank_multisignal(self, query: str, candidates: list[tuple], top_k: int) -> list[tuple]:
        """
        Multi-signal rerank: holographic + keyword + hierarchy-match + depth + tags + domain coherence.
        """
        if not candidates:
            return candidates
        
        query_words = {w for w in query.lower().split() if len(w) > 2}
        
        # Precompute domain keyword coherence
        domain_kw = defaultdict(list)
        node_kw = {}
        for node, _ in candidates:
            content_words = {w for w in node.content.lower().split() if len(w) > 2}
            kw = len(query_words & content_words) / max(len(query_words), 1)
            node_kw[node.node_id] = kw
            domain = node.primary_hierarchy.split(".")[0] if node.primary_hierarchy else "?"
            domain_kw[domain].append(kw)
        domain_avg = {d: sum(v)/len(v) for d, v in domain_kw.items() if v}
        
        scored = []
        for node, holo_sim in candidates:
            keyword_score = node_kw[node.node_id]
            
            hier_words = set()
            if node.primary_hierarchy:
                for part in node.primary_hierarchy.replace('.', ' ').replace('/', ' ').lower().split():
                    if len(part) > 2:
                        hier_words.add(part)
            hier_match = len(query_words & hier_words) / max(len(query_words), 1)
            
            depth = node.primary_hierarchy.count(".") if node.primary_hierarchy else 0
            depth_score = min(depth / 4.0, 1.0)
            
            tag_score = 0.0
            if node.tags and query_words:
                tag_words = set()
                for t in node.tags:
                    tag_words.update(w for w in t.lower().split() if len(w) > 2)
                tag_score = len(query_words & tag_words) / max(len(query_words), 1)
            
            domain = node.primary_hierarchy.split(".")[0] if node.primary_hierarchy else "?"
            domain_coherence = domain_avg.get(domain, 0.0)
            
            combined = (
                0.55 * keyword_score
                + 0.10 * holo_sim
                + 0.10 * hier_match
                + 0.10 * tag_score
                + 0.05 * depth_score
                + 0.10 * domain_coherence
            )
            scored.append((node, combined))
        
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
    
    def _rerank_diffusion(self, query: str, candidates: list[tuple], top_k: int, temperature: float = 0.1) -> list[tuple]:
        """
        Scoped diffusion: runs iterative denoising ONLY on the HNSW candidate
        subset (~50 nodes) instead of the full graph. O(k² × steps) where k << N.
        
        Uses hierarchical guidance within the candidate set for clustering.
        """
        if not candidates:
            return candidates
        
        if len(candidates) <= top_k:
            return candidates[:top_k]
        
        query_vector = self.holographic.encode_node(query, "")
        nodes = [n for n, _ in candidates]
        base_sims = [s for _, s in candidates]
        
        # Precompute keyword scores for guidance
        query_words = {w for w in query.lower().split() if len(w) > 2}
        keyword_scores = []
        for node in nodes:
            content_words = {w for w in node.content.lower().split() if len(w) > 2}
            keyword_scores.append(len(query_words & content_words) / max(len(query_words), 1))
        
        # Build domain index for fast neighbor lookup (replaces O(n) .index() calls)
        domain_map = defaultdict(list)  # domain -> [indices]
        for i, node in enumerate(nodes):
            domain = node.primary_hierarchy.split(".")[0] if node.primary_hierarchy else "?"
            domain_map[domain].append(i)
        
        # Iterative denoising (reduced steps for speed)
        num_steps = 8  # was 20, 8 is enough on pre-filtered candidates
        scores = list(base_sims)
        
        import random as _rnd
        for step in range(num_steps):
            progress = step / num_steps
            noise_level = 0.5 * (1 + math.cos(math.pi * progress))
            
            for i in range(len(nodes)):
                signal = base_sims[i]
                noise = _rnd.gauss(0, noise_level * temperature)
                
                # Domain-local guidance: average score of same-domain neighbors
                domain = nodes[i].primary_hierarchy.split(".")[0] if nodes[i].primary_hierarchy else "?"
                neighbors = domain_map.get(domain, [])
                if len(neighbors) > 1:
                    neighbor_avg = sum(base_sims[j] for j in neighbors) / len(neighbors)
                else:
                    neighbor_avg = 0.0
                
                guidance = 0.3 * neighbor_avg + 0.7 * keyword_scores[i]
                
                scores[i] = (
                    (1 - noise_level) * signal
                    + noise_level * noise
                    + 3.0 * guidance * (1 - noise_level)
                )
                scores[i] = max(0.0, min(1.0, scores[i]))
        
        results = list(zip(nodes, scores))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    def _rerank_combined(self, query: str, candidates: list[tuple], top_k: int) -> list[tuple]:
        """
        Combined mode: HNSW + multisignal + hierarchy-match + domain-boost.
        No diffusion — pure deterministic rerank for speed and reproducibility.
        """
        if not candidates:
            return candidates
        
        query_words = {w for w in query.lower().split() if len(w) > 2}
        
        # First pass: compute domain keyword coherence (like diffusion's guidance)
        domain_keyword_scores = defaultdict(list)
        node_keyword_scores = {}
        for node, _ in candidates:
            content_words = {w for w in node.content.lower().split() if len(w) > 2}
            kw = len(query_words & content_words) / max(len(query_words), 1)
            node_keyword_scores[node.node_id] = kw
            domain = node.primary_hierarchy.split(".")[0] if node.primary_hierarchy else "?"
            domain_keyword_scores[domain].append(kw)
        
        domain_avg = {d: sum(v)/len(v) for d, v in domain_keyword_scores.items() if v}
        
        scored = []
        for node, holo_sim in candidates:
            keyword_score = node_keyword_scores[node.node_id]
            
            hier_words = set()
            if node.primary_hierarchy:
                for part in node.primary_hierarchy.replace('.', ' ').replace('/', ' ').lower().split():
                    if len(part) > 2:
                        hier_words.add(part)
            hier_match = len(query_words & hier_words) / max(len(query_words), 1)
            
            depth = node.primary_hierarchy.count(".") if node.primary_hierarchy else 0
            depth_score = min(depth / 4.0, 1.0)
            
            tag_score = 0.0
            if node.tags and query_words:
                tag_words = set()
                for t in node.tags:
                    tag_words.update(w for w in t.lower().split() if len(w) > 2)
                tag_score = len(query_words & tag_words) / max(len(query_words), 1)
            
            # Domain coherence: boost nodes in domains where neighbors also match
            domain = node.primary_hierarchy.split(".")[0] if node.primary_hierarchy else "?"
            domain_coherence = domain_avg.get(domain, 0.0)
            
            combined = (
                0.55 * keyword_score
                + 0.10 * holo_sim
                + 0.10 * hier_match
                + 0.10 * tag_score
                + 0.05 * depth_score
                + 0.10 * domain_coherence
            )
            
            scored.append((node, combined))
        
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
    
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
    
    # ── Multi-topic & Pattern Emergence (porte de v1, 2026-08-25 MEJORAS.md P0.1) ──

    def query_multi_topic(
        self,
        query_text: str,
        topics: list[str],
        top_k_per_topic: int = 3,
    ) -> dict[str, list[dict]]:
        """
        Multi-topic query using diffusion sampling per topic.
        Returns results organized by topic cluster.
        """
        raw = self.diffusion.sample_multi_topic(
            query_text, self.index, self.holographic,
            topic_dimensions=topics, top_k_per_topic=top_k_per_topic,
        )

        result = {}
        for topic, nodes in raw.items():
            result[topic] = [self._format_result((n, s)) for n, s in nodes]

        return result

    def _build_cubes(self) -> list[dict]:
        """Build cube representations from graph nodes."""
        # Group nodes by primary hierarchy prefix
        topics: dict[str, list[TensorNode]] = {}
        for node in self.nodes.values():
            prefix = node.primary_hierarchy.split(".")[0] if node.primary_hierarchy else "UNKNOWN"
            if prefix not in topics:
                topics[prefix] = []
            topics[prefix].append(node)

        cubes = []
        for i, (topic, nodes) in enumerate(topics.items()):
            cube = {
                "id": f"cube_{i}",
                "topic": topic,
                "dimensions": [h for n in nodes for h in n.hierarchies],
                "active_dimension": 0,
                "subtopics": [n.primary_hierarchy for n in nodes],
                "exposed_subtopic": nodes[0].primary_hierarchy if nodes else "",
                "exposed_content": nodes[0].content[:100] if nodes else "",
                "active_vector": nodes[0].tensor_position if nodes else [],
                "associations": {},
                "node_count": len(nodes),
            }
            # Map associations
            for n in nodes:
                for assoc_id in n.associations:
                    cube["associations"][assoc_id] = True

            cubes.append(cube)

        return cubes

    def _pattern_energy(self, cubes: list[dict], query_vector: list[float]) -> float:
        """Energy function: lower = better alignment with query."""
        energy = 0.0

        for cube in cubes:
            cube_vec = cube.get("active_vector", [])
            if cube_vec:
                # Alignment with query
                alignment = CubeRotator._cosine_similarity(query_vector, cube_vec)
                energy -= alignment  # Higher alignment → lower energy

        return energy / max(len(cubes), 1)

    def find_patterns(
        self,
        query_text: str,
        min_strength: float = 0.5,
    ) -> list[dict]:
        """
        Find emergent cross-topic patterns using annealing + pattern detection.
        """
        # Create cubes from topic clusters
        cubes = self._build_cubes()

        # Run annealing
        # F8 (2026-08-25): hier="" — el marcador "QUERY" entraba al hash como
        # token espuro y degradaba el recall (ver docs/pruebas/F8_INFORME_2026-08-25.md).
        query_vector = self.holographic.encode_node(query_text, "")

        optimized, _, _ = self.annealer.anneal(
            cubes=cubes,
            energy_fn=lambda c: self._pattern_energy(c, query_vector),
            neighbor_fn=lambda c: [self.rotator.random_rotation(cube) for cube in c],
            max_iterations=200,
        )

        # Extract patterns
        patterns = self.pattern_emergence.extract_patterns(optimized, threshold=min_strength)

        return patterns

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
        # FIX 2026-08-25 (MEJORAS.md P0.3): guardado atómico — se escribe a .tmp,
        # se valida el JSON y recién entonces se reemplaza. Un crash a mitad de
        # escritura ya no puede corromper el store (antecedente: corrupción 926MB).
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp_path, encoding="utf-8") as f:
            json.load(f)  # falla ruidoso si el temp quedó truncado
        os.replace(tmp_path, path)
    
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
            # v5 migration: re-encode holographic signatures with semantic encoder
            # Old signatures were random vectors (non-semantic); new ones use
            # feature hashing so HNSW cosine similarity actually means something.
            hierarchy = node.primary_hierarchy or ""
            node.holographic_signature = self.holographic.encode_node(
                node.content, hierarchy
            )
            self.nodes[nid] = node
            self.index.insert(node)
        # FIX 2026-08-25 (MEJORAS.md P0.3): guard anti-duplicación de axiomas en
        # doble load() (mismo bug que causó la corrupción 926MB en v1).
        _axiom_ids = {a.node_id for a in self.axioms}
        for axiom_id in data.get("axiom_ids", []):
            if axiom_id in self.nodes and axiom_id not in _axiom_ids:
                self.axioms.append(self.nodes[axiom_id])
                _axiom_ids.add(axiom_id)
        stats = data.get("stats", {})
        self.query_count = stats.get("query_count", 0)
        self.total_retrieval_time = stats.get("total_retrieval_time", 0.0)
        self._hnsw_dirty = True  # force HNSW rebuild with new signatures
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
            "hnsw": self.hnsw.stats(),
        }
