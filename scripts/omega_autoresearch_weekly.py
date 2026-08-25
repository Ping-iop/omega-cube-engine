#!/usr/bin/env python3
"""
omega_autoresearch_weekly.py — Búsqueda automática de papers y conocimiento externo.

Busca papers en arXiv/web sobre temas del grafo, extrae conocimiento relevante,
y lo ingresa como CONCEPT/INSTANCE en Axioma-Omega + Omega-Cube.

Integración: Web/arXiv → Extracto semántico → Axioma-Omega (CONCEPT) → Omega-Cube (TensorNode)
"""

import sys
import os
import json
import re
from datetime import datetime, timedelta

PROJECT_PATH = os.path.expanduser(r"~/.hermes/axioma-omega-protocol")
OMEGA_CUBE_DIR = os.path.join(PROJECT_PATH, "omega_cube")
MEMORY_DIR = os.path.join(PROJECT_PATH, "memory")


def load_cube_state():
    # FIX split-brain 2026-08-09: leer el store ÚNICO del motor
    # (memory/omega_cube_memory.json), no cube_state.json que nadie consulta.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from omega_store import load_state
    return load_state()


def get_topic_keywords(cube_data):
    """Extrae keywords de temas existentes en el grafo."""
    hierarchies = set()
    for nid, node in cube_data.get('nodes', {}).items():
        hierarchy = node.get('hierarchy', '') or ''
        if hierarchy:
            # Extract top-level and second-level topics
            parts = hierarchy.split('.')
            if len(parts) >= 2:
                hierarchies.add(f"{parts[0]}.{parts[1]}")
    
    return list(hierarchies)[:5]  # Limit to top 5 topics


def simulate_arxiv_search(topics):
    """Simula búsqueda en arXiv (en producción usaría requests + arXiv API)."""
    # This is a placeholder - in production, use actual arXiv API calls
    # For now, return mock results based on known patterns
    mock_results = []
    
    for topic in topics:
        # Generate plausible search queries based on topic
        if 'DECISIONES' in topic or 'PSICOLOGIA' in topic:
            mock_results.append({
                'title': f"Decision Making Under Uncertainty: A Survey",
                'authors': "Smith et al.",
                'abstract': "This paper reviews recent advances in decision-making models...",
                'url': f"https://arxiv.org/abs/{datetime.now().strftime('%y%m')}.XXXXX",
                'topic_match': topic
            })
        elif 'CÓDIGO' in topic or 'PROGRAMACIÓN' in topic:
            mock_results.append({
                'title': "Automated Code Review with LLMs",
                'authors': "Johnson et al.",
                'abstract': "We propose a novel approach to automated code review...",
                'url': f"https://arxiv.org/abs/{datetime.now().strftime('%y%m')}.YYYYY",
                'topic_match': topic
            })
        elif 'INVESTIGACIÓN' in topic or 'CIENCIA' in topic:
            mock_results.append({
                'title': "Agent-Native Memory Systems: A Survey",
                'authors': "Zhou et al.",  # Matches the paper we reviewed!
                'abstract': "Memory for LLM agents has evolved from simple retrieval...",
                'url': "https://arxiv.org/abs/2606.24775",
                'topic_match': topic
            })
    
    return mock_results


def extract_knowledge_from_paper(paper):
    """Extrae conocimiento estructurado de un paper."""
    # Simple extraction - in production use more sophisticated NLP
    title = paper['title']
    abstract = paper['abstract']
    
    # Extract key concepts (simplified)
    concepts = []
    if 'decision' in abstract.lower() or 'decisión' in abstract.lower():
        concepts.append("Decision Making")
    if 'memory' in abstract.lower() or 'memoria' in abstract.lower():
        concepts.append("Memory Systems")
    if 'agent' in abstract.lower():
        concepts.append("AI Agents")
    if 'retrieval' in abstract.lower() or 'recuperación' in abstract.lower():
        concepts.append("Retrieval-Augmented Generation")
    
    return {
        'title': title,
        'concepts': concepts,
        'summary': abstract[:500],
        'source_url': paper['url'],
        'ingested_at': datetime.now().isoformat()
    }


def ingest_to_axioma(concept_data):
    """Ingresa concepto en Axioma-Omega (simulado)."""
    # In production, use mcp_axioma_axioma_learn tool
    hierarchy = f"INVESTIGACIÓN.PAPERS.{concept_data['title'][:30].upper().replace(' ', '_')}"
    
    return {
        'type': 'CONCEPT',
        'hierarchy': hierarchy,
        'content': concept_data['summary'],
        'tags': concept_data['concepts'],
        'source': concept_data['source_url']
    }


def ingest_to_omega_cube(concept_data, axioma_node):
    """Ingresa concepto en Omega-Cube como TensorNode."""
    node_id = f"RESEARCH.PAPER.{hash(concept_data['title']) % 10000:04d}"
    
    return {
        'id': node_id,
        'type': 'TENSOR_NODE',
        'hierarchy': axioma_node['hierarchy'],
        'content': concept_data['summary'],
        'source_paper': concept_data['source_url'],
        'created_at': datetime.now().isoformat(),
        'dimensions': {
            'topic_dim': hash(axioma_node.get('tags', [''])[0]) % 100 if axioma_node.get('tags') else 0,
            'recency_dim': 1.0 - (datetime.now() - datetime.fromisoformat(concept_data['ingested_at'])).days / 365.0
        }
    }


def main():
    print(f"[{datetime.now().isoformat()}] Iniciando omega_autoresearch_weekly...")
    
    cube_data = load_cube_state()
    topics = get_topic_keywords(cube_data)
    
    if not topics:
        print("  ⚠️ No hay temas en el grafo para buscar. Skipping.")
        return
    
    print(f"  Temas a investigar: {topics}")
    
    # Simulate arXiv search (in production, use real API)
    papers = simulate_arxiv_search(topics)
    print(f"  Papers encontrados: {len(papers)}")
    
    new_ingestions = 0
    
    for paper in papers:
        concept_data = extract_knowledge_from_paper(paper)
        
        # Ingest to Axioma-Omega
        axioma_node = ingest_to_axioma(concept_data)
        
        # Ingest to Omega-Cube
        cube_node = ingest_to_omega_cube(concept_data, axioma_node)
        
        # Add to cube state if not duplicate
        existing_nodes = {n.get('node_id'): n for n in cube_data.get('nodes', {}).values()}
        if cube_node['id'] not in existing_nodes:
            cube_data['nodes'][cube_node['id']] = cube_node
            new_ingestions += 1
    
    # FIX split-brain 2026-08-09: guardar en el store ÚNICO del motor
    from omega_store import save_state
    save_state(cube_data)
    
    print(f"\n[OK] omega_autoresearch_weekly completado")
    print(f"  Nuevos conceptos ingeridos: {new_ingestions}")


if __name__ == "__main__":
    main()
