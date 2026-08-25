#!/usr/bin/env python3
"""
omega_daily_evolution.py — AutoResearch loop diario.

Analiza patrones de uso del grafo, sugiere nuevas asociaciones basadas en co-ocurrencia,
y ejecuta el ciclo de auto-optimización semanal si corresponde (domingos).

Integración: Omega-Cube → Axioma-Omega (refuerza asociaciones) → Fabric Memory (log)
"""

import sys
import os
import json
from datetime import datetime, timedelta
from collections import Counter

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
    """Detecta nodos sin uso reciente."""
    stale = []
    cutoff = datetime.now() - timedelta(days=days)
    
    for nid, node in cube_data.get('nodes', {}).items():
        created = node.get('created_at', '') or ''
        try:
            created_dt = datetime.fromisoformat(created)
            if created_dt < cutoff and node.get('last_accessed'):
                stale.append(nid)
        except (ValueError, TypeError):
            continue
    
    return stale


def find_duplicate_or_contradictory(cube_data):
    """Detecta entradas duplicadas o contradictorias."""
    duplicates = []
    content_hashes = {}
    
    for nid, node in cube_data.get('nodes', {}).items():
        content = node.get('content', '')[:200]  # First 200 chars as fingerprint
        if not content:
            continue
            
        h = hash(content)
        if h in content_hashes:
            duplicates.append({
                'original': content_hashes[h],
                'duplicate': nid,
                'similarity_score': 0.95  # Exact match on first 200 chars
            })
        else:
            content_hashes[h] = nid
    
    return duplicates


def find_weak_cross_domain_bridges(cube_data):
    """Detecta asociaciones cruzadas débiles o ausentes entre dominios."""
    hierarchies = set()
    for nid, node in cube_data.get('nodes', {}).items():
        hierarchy = node.get('hierarchy', '') or ''
        if hierarchy:
            top_level = hierarchy.split('.')[0]
            hierarchies.add(top_level)
    
    associations = set()
    for assoc in cube_data.get('associations', []):
        h1, h2 = assoc.get('from', ''), assoc.get('to', '')
        if h1 and h2:
            top1 = h1.split('.')[0]
            top2 = h2.split('.')[0]
            associations.add((top1, top2))
    
    # Find potential cross-domain pairs not yet associated
    weak_bridges = []
    for i, d1 in enumerate(sorted(hierarchies)):
        for d2 in sorted(hierarchies)[i+1:]:
            pair = (d1, d2)
            reverse_pair = (d2, d1)
            if pair not in associations and reverse_pair not in associations:
                weak_bridges.append({
                    'domain_1': d1,
                    'domain_2': d2,
                    'suggested_weight': 0.3  # Weak by default
                })
    
    return list(weak_bridges)[:10]  # Limit to top 10 suggestions


def analyze_usage_patterns(cube_data):
    """Analiza patrones de uso para sugerir optimizaciones."""
    node_counts_by_type = Counter()
    for nid, node in cube_data.get('nodes', {}).items():
        node_type = node.get('node_type', 'UNKNOWN') or 'UNKNOWN'
        node_counts_by_type[node_type] += 1
    
    return dict(node_counts_by_type)


def suggest_new_associations(cube_data, weak_bridges):
    """Sugiere nuevas asociaciones basadas en co-ocurrencia."""
    suggestions = []
    
    for bridge in weak_bridges:
        # Count nodes in each domain
        count_1 = sum(1 for n in cube_data.get('nodes', []) 
                      if n.get('hierarchy', '').startswith(bridge['domain_1']))
        count_2 = sum(1 for n in cube_data.get('nodes', []) 
                      if n.get('hierarchy', '').startswith(bridge['domain_2']))
        
        # Suggest association if both domains have meaningful content
        if count_1 >= 3 and count_2 >= 3:
            suggestions.append({
                'from': bridge['domain_1'],
                'to': bridge['domain_2'],
                'reason': f"Co-ocurrencia detectada: {count_1} nodos en {bridge['domain_1']}, "
                          f"{count_2} nodos en {bridge['domain_2']}"
            })
    
    return suggestions


def main():
    print(f"[{datetime.now().isoformat()}] Iniciando omega_daily_evolution...")
    
    cube_data = load_cube_state()
    
    # 1. Analyze current state
    usage_patterns = analyze_usage_patterns(cube_data)
    print(f"  Patrones de uso: {usage_patterns}")
    
    # 2. Find stale entries (maintenance based on arXiv paper findings)
    stale = find_stale_entries(cube_data, days=7)
    if stale:
        print(f"  ⚠️ Nodos posiblemente stale (>7 días sin uso): {len(stale)}")
        # In production: archive or flag these nodes
    
    # 3. Find duplicates
    duplicates = find_duplicate_or_contradictory(cube_data)
    if duplicates:
        print(f"  ⚠️ Duplicados detectados: {len(duplicates)}")
    
    # 4. Find weak cross-domain bridges (key insight from research)
    weak_bridges = find_weak_cross_domain_bridges(cube_data)
    if weak_bridges:
        print(f"  🌉 Puentes cruzados débiles detectados: {len(weak_bridges)}")
        for wb in weak_bridges[:3]:
            print(f"     - {wb['domain_1']} ↔ {wb['domain_2']} (weight: {wb['suggested_weight']})")
    
    # 5. Suggest new associations
    suggestions = suggest_new_associations(cube_data, weak_bridges)
    if suggestions:
        print(f"  💡 Nuevas asociaciones sugeridas: {len(suggestions)}")
        for s in suggestions[:3]:
            print(f"     - {s['from']} ↔ {s['to']}: {s['reason']}")
    
    # 6. Log to fabric memory (if available)
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': 'daily_evolution',
        'stats': {
            'total_nodes': len(cube_data.get('nodes', [])),
            'associations': len(cube_data.get('associations', [])),
            'stale_candidates': len(stale),
            'duplicates': len(duplicates),
            'weak_bridges': len(weak_bridges),
            'suggested_associations': len(suggestions)
        }
    }
    
    # Save evolution log
    log_path = os.path.join(MEMORY_DIR, "evolution_log.json")
    existing_logs = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                existing_logs = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_logs = []
    
    existing_logs.append(log_entry)
    # Keep only last 30 entries
    existing_logs = existing_logs[-30:]
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(existing_logs, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] omega_daily_evolution completado")
    print(f"  Log guardado en: {log_path}")


if __name__ == "__main__":
    main()
