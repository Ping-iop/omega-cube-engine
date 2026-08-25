#!/usr/bin/env python3
"""
memory_maintenance.py — Maintenance localizado basado en arXiv paper findings.

Implementa el patrón de "localized maintenance" vs "global reorganization" 
descubierto en el paper arXiv:2606.24775. Más eficiente en costo que reorganizar todo.

Integración: Fabric Memory → Omega-Cube (detectar gaps) → Axioma-Omega (reforzar asociaciones)
"""

import sys
import os
import json
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


def find_stale_entries(cube_data, days=7):
    """Detecta nodos sin uso reciente (>X días sin last_accessed)."""
    stale = []
    
    for nid, node in cube_data.get('nodes', {}).items():
        created = node.get('created_at', '') or ''
        last_accessed = node.get('last_accessed')
        
        if not last_accessed:
            # Node never accessed - check creation date
            try:
                created_dt = datetime.fromisoformat(created)
                age_days = (datetime.now() - created_dt).days
                if age_days > days:
                    stale.append({
                        'id': nid,
                        'age_days': age_days,
                        'hierarchy': node.get('hierarchy', '')
                    })
            except (ValueError, TypeError):
                continue
    
    return stale


def find_duplicate_entries(cube_data):
    """Detecta entradas duplicadas o muy similares."""
    duplicates = []
    content_fingerprints = {}
    
    for nid, node in cube_data.get('nodes', {}).items():
        content = node.get('content', '')[:150]  # First 150 chars as fingerprint
        if not content:
            continue
            
        h = hash(content)
        if h in content_fingerprints:
            duplicates.append({
                'original_id': content_fingerprints[h],
                'duplicate_id': nid,
                'similarity_score': 0.95  # High similarity on first 150 chars
            })
        else:
            content_fingerprints[h] = nid
    
    return duplicates


def find_weak_associations(cube_data):
    """Detecta asociaciones débiles o sin peso."""
    weak = []
    
    for assoc in cube_data.get('associations', []):
        weight = assoc.get('weight', 1.0)
        if weight < 0.5:
            weak.append({
                'from': assoc.get('from', ''),
                'to': assoc.get('to', ''),
                'current_weight': weight,
                'suggested_action': 'strengthen'
            })
    
    return weak


def archive_unused_nodes(cube_data, threshold_days=30):
    """Archiva nodos sin uso en >X días (no los elimina)."""
    archived = []
    
    for nid, node in cube_data.get('nodes', {}).items():
        created = node.get('created_at', '') or ''
        last_accessed = node.get('last_accessed')
        
        if not last_accessed:
            try:
                created_dt = datetime.fromisoformat(created)
                age_days = (datetime.now() - created_dt).days
                
                if age_days > threshold_days:
                    archived.append(nid)
                    # Mark as archived instead of deleting
                    node['archived'] = True
                    node['archived_at'] = datetime.now().isoformat()
            except (ValueError, TypeError):
                continue
    
    return archived


def strengthen_weak_associations(cube_data):
    """Refuerza asociaciones débiles basadas en co-ocurrencia."""
    strengthened = []
    
    # Simple heuristic: if two nodes share a hierarchy prefix, strengthen their association
    node_hierarchies = {}
    for nid, node in cube_data.get('nodes', {}).items():
        hierarchy = node.get('hierarchy', '') or ''
        if hierarchy:
            parts = hierarchy.split('.')
            if len(parts) >= 2:
                parent_hierarchy = '.'.join(parts[:2])
                if parent_hierarchy not in node_hierarchies:
                    node_hierarchies[parent_hierarchy] = []
                node_hierarchies[parent_hierarchy].append(nid)
    
    # For each hierarchy with multiple nodes, ensure associations exist
    for parent_hier, node_ids in node_hierarchies.items():
        if len(node_ids) > 1:
            for i, id1 in enumerate(node_ids):
                for id2 in node_ids[i+1:]:
                    # Check if association exists
                    assoc_exists = any(
                        (a.get('from') == id1 and a.get('to') == id2) or
                        (a.get('from') == id2 and a.get('to') == id1)
                        for a in cube_data.get('associations', [])
                    )
                    
                    if not assoc_exists:
                        # Create new association with default weight
                        new_assoc = {
                            'from': id1,
                            'to': id2,
                            'weight': 0.7,  # Default moderate strength
                            'created_at': datetime.now().isoformat()
                        }
                        if 'associations' not in cube_data:
                            cube_data['associations'] = []
                        cube_data['associations'].append(new_assoc)
                        strengthened.append({
                            'id_1': id1,
                            'id_2': id2,
                            'hierarchy': parent_hier
                        })
    
    return strengthened


def main():
    print(f"[{datetime.now().isoformat()}] Iniciando memory_maintenance (localized)...")
    
    cube_data = load_cube_state()
    
    # Ensure 'associations' key exists (Omega-Cube format doesn't always have it)
    if 'associations' not in cube_data:
        cube_data['associations'] = []
    
    # 1. Find stale entries (>7 days without access)
    stale = find_stale_entries(cube_data, days=7)
    if stale:
        print(f"  ⚠️ Nodos posiblemente stale (>7 días sin uso): {len(stale)}")
        for s in stale[:3]:
            print(f"     - {s['id']}: {s['age_days']} días, hier: {s['hierarchy'][:50]}")
    
    # 2. Find duplicates
    duplicates = find_duplicate_entries(cube_data)
    if duplicates:
        print(f"  ⚠️ Duplicados detectados: {len(duplicates)}")
        for d in duplicates[:3]:
            print(f"     - {d['original_id']} ↔ {d['duplicate_id']} (similarity: {d['similarity_score']})")
    
    # 3. Find weak associations
    weak = find_weak_associations(cube_data)
    if weak:
        print(f"  🌉 Asociaciones débiles detectadas: {len(weak)}")
    
    # 4. Archive unused nodes (>30 days without access)
    archived = archive_unused_nodes(cube_data, threshold_days=30)
    if archived:
        print(f"  📦 Nodos archivados (>30 días sin uso): {len(archived)}")
    
    # 5. Strengthen weak associations (key insight: proactive bridging)
    strengthened = strengthen_weak_associations(cube_data)
    if strengthened:
        print(f"  💪 Asociaciones reforzadas: {len(strengthened)}")
        for s in strengthened[:3]:
            print(f"     - {s['id_1']} ↔ {s['id_2']} (hier: {s['hierarchy']})")
    
    # FIX split-brain 2026-08-09: guardar en el store ÚNICO del motor
    from omega_store import save_state
    save_state(cube_data)
    
    print(f"\n[OK] memory_maintenance completado")
    print(f"  Resumen:")
    print(f"    - Stale candidates: {len(stale)}")
    print(f"    - Duplicates: {len(duplicates)}")
    print(f"    - Weak associations: {len(weak)}")
    print(f"    - Archived nodes: {len(archived)}")
    print(f"    - Strengthened associations: {len(strengthened)}")


if __name__ == "__main__":
    main()
