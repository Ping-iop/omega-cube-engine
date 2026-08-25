#!/usr/bin/env python3
"""
omega_autopublisher.py — Publica release de Omega-Cube si score > 0.45.

Verifica que el grafo esté saludable, genera release notes automáticas,
y sube a GitHub si cumple thresholds.

Integración: Omega-Cube → GitHub (si aplica) → Fabric Memory (log)
"""

import sys
import os
import json
from datetime import datetime

PROJECT_PATH = os.path.expanduser(r"~/.hermes/axioma-omega-protocol")
OMEGA_CUBE_DIR = os.path.join(PROJECT_PATH, "omega_cube")
MEMORY_DIR = os.path.join(PROJECT_PATH, "memory")


def load_cube_state():
    # FIX split-brain 2026-08-09: leer el store ÚNICO del motor
    # (memory/omega_cube_memory.json), no cube_state.json que nadie consulta.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from omega_store import load_state
    return load_state()


def calculate_health_score(cube_data):
    """Calcula score de salud del grafo (0.0 - 1.0)."""
    nodes = cube_data.get('nodes', {})
    associations = cube_data.get('associations', [])
    
    if not nodes:
        return 0.0
    
    # Factor 1: Node count (more is better, cap at 200)
    node_score = min(len(nodes) / 200.0, 1.0) * 0.3
    
    # Factor 2: Association density (associations per node)
    if len(nodes) > 0:
        assoc_ratio = len(associations) / len(nodes)
        assoc_score = min(assoc_ratio / 2.0, 1.0) * 0.3  # Ideal: 2 associations per node
    else:
        assoc_score = 0.0
    
    # Factor 3: Cross-domain coverage (diversity of hierarchies)
    hierarchies = set()
    for nid, node in nodes.items():
        hierarchy = node.get('hierarchy', '') or ''
        if hierarchy:
            top_level = hierarchy.split('.')[0]
            hierarchies.add(top_level)
    
    diversity_score = min(len(hierarchies) / 10.0, 1.0) * 0.4  # Ideal: 10+ domains
    
    total_score = node_score + assoc_score + diversity_score
    return round(total_score, 3)


def generate_release_notes(cube_data, score):
    """Genera release notes automáticas."""
    nodes = cube_data.get('nodes', {})
    associations = cube_data.get('associations', [])
    
    # Count by type
    node_types = {}
    for nid, node in nodes.items():
        ntype = node.get('node_type', 'UNKNOWN') or 'UNKNOWN'
        node_types[ntype] = node_types.get(ntype, 0) + 1
    
    # Count by hierarchy top-level
    domains = {}
    for nid, node in nodes.items():
        hierarchy = node.get('hierarchy', '') or ''
        if hierarchy:
            top_level = hierarchy.split('.')[0]
            domains[top_level] = domains.get(top_level, 0) + 1
    
    release_notes = f"""## Omega-Cube Release Notes — {datetime.now().strftime('%Y-%m-%d')}

**Health Score:** {score:.3f} (Threshold: 0.45)

### Summary
- **Total Nodes:** {len(nodes)}
- **Associations:** {len(associations)}
- **Domains Covered:** {len(domains)}

### Node Distribution by Type
"""
    for ntype, count in sorted(node_types.items(), key=lambda x: -x[1]):
        release_notes += f"- `{ntype}`: {count}\n"
    
    release_notes += "\n### Top Domains\n"
    for domain, count in sorted(domains.items(), key=lambda x: -x[1])[:5]:
        release_notes += f"- **{domain}**: {count} nodes\n"
    
    return release_notes


def save_release_log(release_notes):
    """Guarda release notes en memory."""
    log_path = os.path.join(MEMORY_DIR, "releases.json")
    existing_releases = []
    
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                existing_releases = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_releases = []
    
    release_entry = {
        'timestamp': datetime.now().isoformat(),
        'notes': release_notes
    }
    
    existing_releases.append(release_entry)
    # Keep only last 10 releases
    existing_releases = existing_releases[-10:]
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(existing_releases, f, ensure_ascii=False, indent=2)


def main():
    print(f"[{datetime.now().isoformat()}] Iniciando omega_autopublisher...")
    
    cube_data = load_cube_state()
    score = calculate_health_score(cube_data)
    
    print(f"  Health Score: {score:.3f}")
    
    if score < 0.45:
        print(f"  ⚠️ Score bajo ({score:.3f} < 0.45). No publicando.")
        print(f"     Necesitas más nodos, asociaciones o diversidad de dominios.")
        return
    
    print(f"  ✅ Score suficiente ({score:.3f} >= 0.45). Generando release...")
    
    # Generate release notes
    release_notes = generate_release_notes(cube_data, score)
    print(release_notes)
    
    # Save to memory
    save_release_log(release_notes)
    print(f"\n[OK] Release guardado en: {MEMORY_DIR}/releases.json")


if __name__ == "__main__":
    main()
