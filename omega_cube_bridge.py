"""
Omega-Cube Bridge — Conecta Omega-Cube con fabric_recall/fabric_search.

Este modulo traduce entre la API de fabric (entries, tags, agent) y las queries
semánticas del motor Omega-Cube usando HolographicEncoder para busqueda O(1).

Uso:
    bridge = OmegaCubeBridge()
    results = bridge.semantic_search("evony generals meta")
    results = bridge.query_fabric(query="como configurar", top_k=5)
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

# Import engine components
from .engine import OmegaCubeEngine
from .holographic import HolographicEncoder


class OmegaCubeBridge:
    """
    Puente entre fabric_recall/fabric_search y Omega-Cube.
    
    Proporciona busqueda semantica real usando HolographicEncoder en vez de
    keyword matching, integrando la memoria persistente de Hermes con el
    motor multi-dimensional.
    """
    
    def __init__(self, memory_dir: str = None):
        """Inicializar bridge con el engine Omega-Cube."""
        if memory_dir is None:
            axioma_base = os.environ.get(
                "AXIOMA_PROJECT_PATH",
                str(Path.home() / ".hermes" / "axioma-omega-protocol")
            )
            memory_dir = os.path.join(axioma_base, "memory")
        
        self.memory_dir = Path(memory_dir)
        self.engine = OmegaCubeEngine(memory_dir=str(self.memory_dir))
        self.holographic = HolographicEncoder()
        
        # Cargar estado existente si hay
        if not self.engine.load():
            print(f"[OmegaCubeBridge] No state found at {self.memory_dir}")
    
    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Busqueda semantica usando HolographicEncoder.
        
        Reemplaza keyword matching de fabric_search con busqueda O(1) por
        similitud holografica en el espacio N-dimensional.
        
        Args:
            query: Consulta natural language
            top_k: Numero de resultados
        
        Returns:
            Lista de dicts con node_id, content, hierarchies, score
        """
        results = self.engine.query(query, mode="holographic", top_k=top_k)
        return results
    
    def query_fabric(self, query: str, agent: str = None, project: str = None, 
                     max_results: int = 5) -> list[dict]:
        """
        Query fabric entries via Omega-Cube semantic search.
        
        Simula fabric_recall() pero usando el motor semantico en vez de
        ranking por keyword overlap.
        
        Args:
            query: Texto a buscar
            agent: Filtrar por agente (opcional)
            project: Filtrar por proyecto (opcional)
            max_results: Max resultados
        
        Returns:
            Lista de entries con scores semanticos
        """
        # Query el engine
        results = self.engine.query(query, mode="combined", top_k=max_results * 2)
        
        # Si hay filtros, aplicar post-query
        if agent or project:
            filtered = []
            for r in results:
                matches = True
                if agent and agent.lower() not in json.dumps(r).lower():
                    matches = False
                if project and project.lower() not in json.dumps(r).lower():
                    matches = False
                if matches:
                    filtered.append(r)
            results = filtered[:max_results]
        
        return results
    
    def index_fabric_entry(self, entry_id: str, content: str, 
                           tags: list[str] = None, agent: str = None,
                           project: str = None) -> bool:
        """
        Indexar una entry de fabric en Omega-Cube.
        
        Convierte la entrada de fabric (task, decision, review, etc.) en un
        TensorNode con jerarquias multi-dimensionales basadas en tipo y tags.
        
        Args:
            entry_id: ID unico de la entry
            content: Contenido completo
            tags: Tags asociados
            agent: Agente que creo la entry
            project: Proyecto relacionado
        
        Returns:
            True si se indexo exitosamente
        """
        # Construir jerarquias multi-dimensionales
        hierarchies = []
        
        if agent:
            hierarchies.append(f"AGENT.{agent.upper()}")
        if project:
            hierarchies.append(f"PROJECT.{project.upper().replace(' ', '_')}")
        
        # Agregar tags como dimensiones adicionales
        for tag in (tags or []):
            hierarchies.append(f"TAG.{tag.upper()}")
        
        # Siempre agregar por tipo de contenido
        content_type = "MEMORY"
        if any(kw in content.lower() for kw in ["decision", "decidido", "elegido"]):
            content_type = "DECISION"
        elif any(kw in content.lower() for kw in ["review", "revisar", "calidad"]):
            content_type = "REVIEW"
        elif any(kw in content.lower() for kw in ["task", "tarea", "trabajo"]):
            content_type = "TASK"
        
        hierarchies.append(f"TYPE.{content_type}")
        
        # Agregar entry_id como jerarquia final
        hierarchies.append(f"ENTRY.{entry_id[:8]}")
        
        try:
            node = self.engine.add_node(
                content=content,
                hierarchies=hierarchies,
                node_type="CONCEPT",
                confidence=0.85,
                tags=tags or [],
            )
            
            # Guardar estado
            self.engine.save()
            return True
            
        except Exception as e:
            print(f"[OmegaCubeBridge] Error indexing entry {entry_id}: {e}")
            return False
    
    def update_entry(self, entry_id: str, new_content: str) -> bool:
        """Actualizar contenido de una entry existente."""
        # Buscar nodo por ID parcial
        target = None
        for nid, node in self.engine.nodes.items():
            if entry_id[:8] in nid or entry_id in node.content:
                target = node
                break
        
        if target:
            target.content = new_content
            # Recalcular signature
            target.holographic_signature = self.holographic.encode_node(
                new_content, target.primary_hierarchy
            )
            self.engine.save()
            return True
        return False
    
    def delete_entry(self, entry_id: str) -> bool:
        """Eliminar una entry del grafo."""
        for nid in list(self.engine.nodes.keys()):
            if entry_id[:8] in nid:
                node = self.engine.nodes.pop(nid)
                # Eliminar de asociaciones
                for other_nid, other_node in self.engine.nodes.items():
                    if nid in other_node.associations:
                        other_node.associations.remove(nid)
                self.engine.save()
                return True
        return False
    
    def get_stats(self) -> dict:
        """Obtener estadisticas del engine."""
        type_counts = {}
        for node in self.engine.nodes.values():
            t = node.node_type
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            "total_nodes": len(self.engine.nodes),
            "type_distribution": type_counts,
            "query_count": self.engine.query_count,
            "avg_retrieval_time_ms": (
                (self.engine.total_retrieval_time / self.engine.query_count * 1000)
                if self.engine.query_count > 0 else 0
            ),
        }


def quick_search(query: str, top_k: int = 5) -> list[dict]:
    """Funcion standalone rapida para busqueda semantica."""
    bridge = OmegaCubeBridge()
    return bridge.semantic_search(query, top_k=top_k)


if __name__ == "__main__":
    # Test rapido
    print("Testing Omega-Cube Bridge...")
    
    bridge = OmegaCubeBridge()
    stats = bridge.get_stats()
    print(f"Nodes loaded: {stats['total_nodes']}")
    print(f"Type distribution: {stats['type_distribution']}")
    
    if stats['total_nodes'] > 0:
        results = bridge.semantic_search("evony", top_k=3)
        print(f"\nQuery 'evony' -> {len(results)} results:")
        for r in results[:3]:
            print(f"  [{r['score']:.2f}] {r['content'][:80]}...")
    else:
        print("No nodes loaded. Run omega_auto_indexer.py first.")
