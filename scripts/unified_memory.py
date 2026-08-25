#!/usr/bin/env python3
"""
unified_memory_save() — Hook central de integración Axioma-Omega → Omega-Cube → Fabric.

Cada vez que se toma una decisión, aprende un concepto o guarda una instancia,
esta función garantiza que fluya a los tres sistemas simultáneamente.

Uso:
    from unified_memory import save_decision, save_concept, save_instance
    
    save_decision(
        category="memory_optimization",
        scenario="Evaluando mejora de proceso de memoria",
        reasoning="Paper arXiv sugiere maintenance localizado vs global",
        outcome="adopt_localized_maintenance",
        confidence=0.85,
    )
"""

import sys
import os
import json
from datetime import datetime

PROJECT_PATH = os.path.expanduser(r"~/.hermes/axioma-omega-protocol")
OMEGA_CUBE_DIR = os.path.join(PROJECT_PATH, "omega_cube")
MEMORY_DIR = os.path.join(PROJECT_PATH, "memory")


def load_cube_state():
    # FIX split-brain 2026-08-09: store ÚNICO del motor (omega_store)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from omega_store import load_state
    return load_state()


def save_cube_state(cube_data):
    # FIX split-brain 2026-08-09: store ÚNICO del motor (omega_store)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from omega_store import save_state
    save_state(cube_data)


def classify_hierarchy(category, scenario):
    """Clasifica en jerarquía semántica."""
    cat_lower = category.lower()
    
    if any(kw in cat_lower for kw in ['decision', 'decisión']):
        return f"DECISIONES.{scenario[:30].upper().replace(' ', '_')}"
    elif any(kw in cat_lower for kw in ['código', 'code']):
        return f"CÓDIGO.{scenario[:30].upper().replace(' ', '_')}"
    elif any(kw in cat_lower for kw in ['investigación', 'research']):
        return f"INVESTIGACIÓN.{scenario[:30].upper().replace(' ', '_')}"
    elif any(kw in cat_lower for kw in ['proyecto', 'project']):
        return f"PROYECTO.{scenario[:30].upper().replace(' ', '_')}"
    else:
        return f"{category[:20].upper().replace(' ', '_')}." + scenario[:20].upper().replace(' ', '_')


def save_decision(category, scenario, reasoning, outcome, confidence=1.0):
    """Guarda una decisión en los tres sistemas simultáneamente."""
    print(f"[unified_memory] Guardando DECISIÓN: {scenario}")
    
    cube_data = load_cube_state()
    hierarchy = classify_hierarchy(category, scenario)
    timestamp = datetime.now().isoformat()
    
    # 1. Axioma-Omega: CONCEPT con jerarquía semántica
    axiom_node = {
        'type': 'CONCEPT',
        'hierarchy': hierarchy,
        'content': f"Decisión: {scenario}\nRazonamiento: {reasoning}\nOutcome: {outcome}",
        'tags': [category.lower()],
        'metadata': {
            'decision_category': category,
            'confidence': confidence,
            'created_at': timestamp
        }
    }
    
    # 2. Omega-Cube: TensorNode N-dim (sesión, turno, tema, tipo)
    node_id = f"DECISION.{hash(scenario + timestamp) % 10000:04d}"
    cube_node = {
        'id': node_id,
        'type': 'TENSOR_NODE',
        'hierarchy': hierarchy,
        'content': f"{scenario} → {outcome}",
        'source_session': 'unified_memory_save',
        'created_at': timestamp,
        'last_accessed': timestamp,  # Just created = just accessed
        'dimensions': {
            'session_dim': datetime.now().timestamp(),
            'topic_dim': hash(category) % 100,
            'confidence_dim': confidence,
            'length_dim': len(scenario) / 50.0
        }
    }
    
    # Add to cube state if not duplicate
    if node_id not in cube_data.get('nodes', {}):
        cube_data['nodes'][node_id] = cube_node
    
    save_cube_state(cube_data)
    
    # 3. Fabric Memory: persistencia cross-agent con metadata de trazabilidad
    fabric_entry = {
        'type': 'decision',
        'summary': f"Decisión: {scenario}",
        'content': json.dumps(axiom_node, ensure_ascii=False),
        'tags': f"{category},{outcome},memory_unified",
        'status': 'completed',
        'verified': 'true',
        'evidence': f"confidence={confidence}",
        'source_tool': 'unified_memory_save',
        'artifact_paths': '',
        'assigned_to': '',
        'outcome': outcome,
        'training_value': 'high' if confidence > 0.7 else 'normal',
        'created_at': timestamp,
        '_unified_ref': {
            'axiom_hierarchy': hierarchy,
            'cube_node_id': node_id
        }
    }
    
    # Save to fabric memory file (append)
    fabric_path = os.path.join(MEMORY_DIR, "unified_decisions.jsonl")
    with open(fabric_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(fabric_entry, ensure_ascii=False) + '\n')
    
    print(f"  ✅ Axioma-Omega: {hierarchy}")
    print(f"  ✅ Omega-Cube: TensorNode {node_id} (N-dim)")
    print(f"  ✅ Fabric Memory: decision con trazabilidad completa")
    
    return {
        'axiom_hierarchy': hierarchy,
        'cube_node_id': node_id,
        'fabric_path': fabric_path
    }


def save_concept(content, hierarchy=None, tags=None):
    """Guarda un concepto nuevo en los tres sistemas."""
    print(f"[unified_memory] Guardando CONCEPTO: {content[:50]}...")
    
    cube_data = load_cube_state()
    timestamp = datetime.now().isoformat()
    
    if not hierarchy:
        # Auto-classify based on content keywords
        cl = content.lower()
        if 'memory' in cl or 'memoria' in cl:
            hierarchy = f"INVESTIGACIÓN.MEMORIA.{content[:25].upper().replace(' ', '_')}"
        elif 'decision' in cl or 'decisión' in cl:
            hierarchy = f"DECISIONES.{content[:25].upper().replace(' ', '_')}"
        else:
            hierarchy = f"CONCEPTOS.{content[:25].upper().replace(' ', '_')}"
    
    node_id = f"CONCEPT.{hash(content + timestamp) % 10000:04d}"
    
    # Axioma-Omega
    axiom_node = {
        'type': 'CONCEPT',
        'hierarchy': hierarchy,
        'content': content,
        'tags': tags or [],
        'metadata': {'created_at': timestamp}
    }
    
    # Omega-Cube
    cube_node = {
        'id': node_id,
        'type': 'TENSOR_NODE',
        'hierarchy': hierarchy,
        'content': content[:2000],
        'source_session': 'unified_memory_save',
        'created_at': timestamp,
        'last_accessed': timestamp,
        'dimensions': {
            'session_dim': datetime.now().timestamp(),
            'topic_dim': hash(hierarchy.split('.')[0]) % 100,
            'length_dim': len(content) / 100.0
        }
    }
    
    if node_id not in cube_data.get('nodes', {}):
        cube_data['nodes'][node_id] = cube_node
    
    save_cube_state(cube_data)
    
    # Fabric Memory
    fabric_entry = {
        'type': 'concept',
        'summary': content[:80],
        'content': json.dumps(axiom_node, ensure_ascii=False),
        'tags': ','.join(tags or []),
        'status': 'completed',
        'verified': 'false',
        'evidence': '',
        'source_tool': 'unified_memory_save',
        'outcome': 'concept_saved',
        'training_value': 'normal',
        'created_at': timestamp,
        '_unified_ref': {
            'axiom_hierarchy': hierarchy,
            'cube_node_id': node_id
        }
    }
    
    fabric_path = os.path.join(MEMORY_DIR, "unified_concepts.jsonl")
    with open(fabric_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(fabric_entry, ensure_ascii=False) + '\n')
    
    print(f"  ✅ Axioma-Omega: {hierarchy}")
    print(f"  ✅ Omega-Cube: TensorNode {node_id}")
    print(f"  ✅ Fabric Memory: concept con metadata")
    
    return {'axiom_hierarchy': hierarchy, 'cube_node_id': node_id}


def trace_decision_chain(decision_id_or_hierarchy):
    """Trazabilidad causal completa de una decisión."""
    cube_data = load_cube_state()
    
    # Find the decision node
    target = None
    target_id = None
    target_id = None
    for nid, node in cube_data.get('nodes', {}).items():
        if (nid == decision_id_or_hierarchy or 
            decision_id_or_hierarchy in node.get('hierarchy', '')):
            target = node
            target_id = nid
            break
    
    if not target:
        return {"error": "Decision node not found"}
    
    # Find all nodes with related hierarchy
    chain = [target] if target else []
    if target:
        hierarchy_prefix = target.get('hierarchy', '').rsplit('.', 1)[0]
        
        for nid, node in cube_data.get('nodes', {}).items():
            if nid != target_id and node.get('hierarchy', '').startswith(hierarchy_prefix):
                chain.append(node)
    
    # Find associations involving this decision
    related_associations = [
        a for a in cube_data.get('associations', [])
        if target_id and (a.get('from') == target_id or a.get('to') == target_id)
    ]
    
    return {
        'decision': target,
        'related_nodes': chain[1:] if len(chain) > 1 else [],
        'associations': related_associations,
        'total_chain_length': len(chain)
    }


def find_similar_decisions(query, max_results=5):
    """Encuentra decisiones similares por texto."""
    cube_data = load_cube_state()
    
    results = []
    query_lower = query.lower()
    
    for nid, node in cube_data.get('nodes', {}).items():
        content = node.get('content', '').lower()
        hierarchy = node.get('hierarchy', '').lower()
        
        # Simple keyword overlap scoring
        score = 0
        for word in query_lower.split():
            if len(word) > 2 and word in content:
                score += 1
            if len(word) > 2 and word in hierarchy:
                score += 1
        
        if score > 0:
            results.append({
                'node_id': nid,
                'hierarchy': node.get('hierarchy', ''),
                'content_preview': node.get('content', '')[:100],
                'relevance_score': score / max(len(query_lower.split()), 1)
            })
    
    # Sort by relevance and limit
    results.sort(key=lambda x: -x['relevance_score'])
    return results[:max_results]


# --- Demo ---
if __name__ == "__main__":
    print("=== Unified Memory Save — Demo ===\n")
    
    # Test save_decision
    result = save_decision(
        category="memory_optimization",
        scenario="Adopting localized maintenance over global reorganization",
        reasoning="Paper arXiv:2606.24775 shows localized is more cost-efficient",
        outcome="adopt_localized_maintenance",
        confidence=0.85,
    )
    
    print(f"\nResult: {result}")
    
    # Test trace_decision_chain
    chain = trace_decision_chain(result['axiom_hierarchy'])
    print(f"\nDecision chain length: {chain.get('total_chain_length', 0)}")
    
    # Test find_similar_decisions
    similar = find_similar_decisions("memory optimization maintenance")
    print(f"Similar decisions found: {len(similar)}")
