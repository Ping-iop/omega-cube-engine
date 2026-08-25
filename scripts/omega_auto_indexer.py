#!/usr/bin/env python3
"""
omega_auto_indexer.py — Indexa conversaciones recientes en Omega-Cube.

Lee sesiones del DB de Hermes, trocea por tema y las ingresa como TensorNodes N-dim.
Integración: Axioma-Omega → Omega-Cube → Fabric Memory.

Uso standalone: python omega_auto_indexer.py
Uso via cron (no_agent=True): el script se ejecuta directamente.
"""

import sys
import os
import json
import hashlib
from datetime import datetime, timedelta

# Add project paths
PROJECT_PATH = os.path.expanduser(r"~/.hermes/axioma-omega-protocol")
OMEGA_CUBE_DIR = os.path.join(PROJECT_PATH, "omega_cube")
MEMORY_DIR = os.path.join(PROJECT_PATH, "memory")

sys.path.insert(0, PROJECT_PATH)  # Add project root to path

try:
    from omega_cube.engine import OmegaCubeEngine
except ImportError as e:
    print(f"ERROR: No se pudo importar OmegaCubeEngine ({e})")
    sys.exit(1)


# Archivos de ESTADO del motor — NUNCA ingerirlos como sesiones.
# Bug 2026-08-09: el indexer se auto-ingería (dumps de cube_state/unified_memory
# terminaban como nodos "GENERAL", generando 95% de basura duplicada).
STATE_FILES = {
    'omega_cube_memory.json', 'omega_cube_memory_v2.json', 'unified_memory.json',
    'evolution_log.json', 'releases.json', 'cube_state.json', 'context_state.json',
    'telemetry.json',  # FIX 2026-08-09 (22:20): se auto-ingirió como 11 nodos basura
    'semantic_embeddings.json',  # P1.10 2026-08-25: cache de embeddings se auto-ingirió como nodo basura
}


def get_recent_sessions(limit=20):
    """Simula session_search trayendo sesiones recientes por fecha."""
    # En producción, esto usaría fabric_recall o session_search directamente
    # Por ahora, buscamos en el directorio de memoria existente
    sessions = []
    
    if not os.path.exists(MEMORY_DIR):
        return sessions
    
    for root, dirs, files in os.walk(MEMORY_DIR):
        dirs[:] = [d for d in dirs if d != 'backups']  # no re-ingerir backups
        for f in files:
            if f.endswith('.json') and f not in STATE_FILES:
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                        sessions.append({
                            'file': filepath,
                            'data': data,
                            'mtime': os.path.getmtime(filepath)
                        })
                except (json.JSONDecodeError, IOError):
                    continue
    
    # Ordenar por fecha más reciente
    sessions.sort(key=lambda x: x['mtime'], reverse=True)
    return sessions[:limit]


def segment_by_topic(text, max_segments=5):
    """Trocea texto largo en segmentos temáticos."""
    if not text or len(text.strip()) < 50:
        return []
    
    # Split by natural boundaries (newlines, paragraphs)
    lines = text.split('\n')
    segments = []
    current_segment = []
    current_topic = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect topic change (simple heuristic: all-caps or numbered headers)
        is_new_topic = (
            line.startswith(('##', '###', '#')) or 
            line.isupper() or
            len(line) < 30 and ':' in line
        )
        
        if is_new_topic and current_segment:
            segments.append({
                'topic': current_topic,
                'text': '\n'.join(current_segment[-5:]) if len(current_segment) > 5 else '\n'.join(current_segment)
            })
            current_segment = [line]
            current_topic = line[:50]
        else:
            current_segment.append(line)
    
    if current_segment:
        segments.append({
            'topic': current_topic,
            'text': '\n'.join(current_segment[-5:]) if len(current_segment) > 5 else '\n'.join(current_segment)
        })
    
    return segments[:max_segments]


def classify_hierarchy(topic_text):
    """Clasifica texto en jerarquía semántica."""
    topic_lower = topic_text.lower()
    
    # Simple keyword-based classification (expand as needed)
    if any(kw in topic_lower for kw in ['decision', 'decisión', 'elige', 'elijo']):
        return f"DECISIONES.{topic_text[:30].upper().replace(' ', '_')}"
    elif any(kw in topic_lower for kw in ['código', 'code', 'programa', 'debug']):
        return f"CÓDIGO.{topic_text[:30].upper().replace(' ', '_')}"
    elif any(kw in topic_lower for kw in ['investigación', 'research', 'paper', 'arxiv']):
        return f"INVESTIGACIÓN.{topic_text[:30].upper().replace(' ', '_')}"
    elif any(kw in topic_lower for kw in ['proyecto', 'project', 'tarea', 'task']):
        return f"PROYECTO.{topic_text[:30].upper().replace(' ', '_')}"
    else:
        return f"GENERAL.{topic_text[:30].upper().replace(' ', '_')}"


# === Repos locales: la fuente de verdad son los docs del repo, Omega es solo índice ===
# Fase 4 (2026-08-10): expandido de 1 repo a 9 — todos con docs reales verificables.
_APPS = r"K:/Proyectos/DESARROLLO_APPS"  # FIX 2026-08-25: repos movidos a K:/Proyectos (antes ~/Documents/GEMINI)
REPO_DOCS = [
    {'repo': os.path.join(_APPS, "EvonyBot-Pro"), 'name': 'EVONYBOT',
     'paths': ['ISSUES.md', 'PLAN.md', 'PLAN_MEJORAS_BOTS.md', 'dev', '.sessions']},
    {'repo': os.path.join(_APPS, "H-Bit"), 'name': 'HBIT',
     'paths': ['README.md']},
    {'repo': os.path.join(_APPS, "Blastron"), 'name': 'BLASTRON',
     'paths': ['PLAN.md']},
    {'repo': os.path.join(_APPS, "MixClaw-ai-V2"), 'name': 'MIXCLAW',
     'paths': ['README.md']},
    {'repo': os.path.join(_APPS, "MASTERFLUX"), 'name': 'MASTERFLUX',
     'paths': ['README.md']},
    {'repo': os.path.join(_APPS, "NEURALLIANCE - DOCKER"), 'name': 'NEURALLIANCE',
     'paths': ['README.md']},
    {'repo': os.path.join(_APPS, "INVESTIGACION"), 'name': 'INVESTIGACION',
     'paths': ['README.md']},
    {'repo': os.path.join(_APPS, "hermes-intention-middleware"), 'name': 'INTENTION_MIDDLEWARE',
     'paths': ['README.md', 'ROADMAP.md']},
    {'repo': os.path.join(_APPS, "TESTFI"), 'name': 'TESTFI',
     'paths': ['README.md']},
]


def engine_put(engine, by_id, node_id, content, hierarchies, node_type, confidence, tags):
    """Inserta/actualiza un nodo en el engine con node_id ESTABLE.

    add_node() genera node_id por hash de contenido, pero el indexer necesita
    ids estables (por ruta de archivo / sesión) para dedup entre corridas.
    """
    import time as _time
    from omega_cube.tensor_node import TensorNode

    # INVARIANTE P1.10 (2026-08-25): todo nodo no-AXIOM nace con cadena de tono
    # verificable. El dominio (1er segmento) DEBE tener axioma; si no, el nodo
    # sería huérfano → se re-hogar bajo DEV.TOOLS.* (DEV tiene axioma) en vez
    # de rechazar: conserva el dato Y garantiza la cadena.
    segs = hierarchies[0].split(".") if hierarchies else []
    if segs:
        dom = segs[0]
        if not any(a.primary_hierarchy.split(".")[0] == dom for a in engine.axioms):
            resto = ".".join(segs[1:]) or segs[0]
            hierarchies = [f"DEV.TOOLS.{resto}"]

    existing = by_id.get(node_id)
    if existing is not None:
        existing.content = content
        existing.hierarchies = hierarchies
        existing.tags = tags
        existing.holographic_signature = engine.holographic.encode_node(content, hierarchies[0])
        return 'updated'

    node = TensorNode(
        content=content,
        hierarchies=hierarchies,
        tensor_position=engine._compute_tensor_position(hierarchies),
        node_type=node_type,
        confidence=confidence,
        tags=tags,
        node_id=node_id,
        created_at=_time.time(),
    )
    node.holographic_signature = engine.holographic.encode_node(content, hierarchies[0])
    engine.nodes[node_id] = node
    engine.index.insert(node)
    by_id[node_id] = node
    return 'added'


def index_repo_docs(engine, by_id):
    """Indexa documentación de repos locales en Omega-Cube (store único del motor).

    Semántica de actualización: cada archivo se identifica por ruta (node_id estable)
    y lleva un content_hash en tags. Si el hash cambió, el nodo se actualiza in-place.
    """
    new_docs = 0
    updated_docs = 0

    for repo_cfg in REPO_DOCS:
        repo_root = repo_cfg['repo']
        repo_name = repo_cfg['name']
        if not os.path.isdir(repo_root):
            print(f"  [SKIP] Repo no existe: {repo_root}")
            continue

        # Recolectar .md de las rutas configuradas (archivo suelto o directorio)
        md_files = []
        for rel in repo_cfg['paths']:
            full = os.path.join(repo_root, rel)
            if os.path.isfile(full) and full.endswith('.md'):
                md_files.append(full)
            elif os.path.isdir(full):
                for root, dirs, files in os.walk(full):
                    for f in files:
                        if f.endswith('.md'):
                            md_files.append(os.path.join(root, f))

        for filepath in md_files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
                    content = fh.read()
            except IOError:
                continue

            if len(content.strip()) < 50:
                continue

            rel_path = os.path.relpath(filepath, repo_root).replace(os.sep, '/')
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
            node_id = f"REPO.{repo_name}.DOC.{rel_path.replace('/', '.')}"

            existing = by_id.get(node_id)
            hash_tag = f"content_hash:{content_hash}"
            if existing is not None and hash_tag in existing.tags:
                continue  # sin cambios desde la última indexación

            # Segmentar docs largos por secciones (## headers) para búsqueda granular
            segments = segment_by_topic(content, max_segments=8)
            summary = content[:2000] if not segments else '\n\n'.join(
                s['text'] for s in segments if s['text'])[:2000]

            result = engine_put(
                engine, by_id, node_id, summary,
                hierarchies=[f"PROYECTO.{repo_name}.{classify_hierarchy(rel_path)}"],
                node_type='INSTANCE',  # DOCUMENT no existe en el enum del engine
                confidence=1.0,  # fuente de verdad del repo, no inferencia
                tags=['repo_doc', repo_name.lower(), rel_path, hash_tag],
            )
            if result == 'updated':
                updated_docs += 1
            else:
                new_docs += 1

    return new_docs, updated_docs


def main():
    print(f"[{datetime.now().isoformat()}] Iniciando omega_auto_indexer...")

    # === UN STORE ÚNICO: el motor consultable (omega_cube_memory.json) ===
    # FIX split-brain 2026-08-09: antes se escribía a cube_state.json, que el
    # motor MCP NUNCA lee. Ahora todo pasa por OmegaCubeEngine.add_node()+save().
    # FIX 2026-08-09 (22:05): el constructor YA auto-carga el store; el
    # engine.load() extra duplicaba axiomas en cada corrida (→ corrupción 926MB).
    from omega_cube.engine import OmegaCubeEngine
    engine = OmegaCubeEngine()
    print(f"  Motor cargado: {len(engine.nodes)} nodos")

    # Índice auxiliar por node_id para dedup estable
    by_id = {n.node_id: n for n in engine.nodes.values()}

    context_state_path = os.path.join(OMEGA_CUBE_DIR, "context_state.json")
    
    # Get recent sessions
    sessions = get_recent_sessions(limit=10)
    print(f"  Encontradas {len(sessions)} sesiones recientes")

    # === Indexar docs de repos locales (fuente de verdad) ===
    new_docs, updated_docs = index_repo_docs(engine, by_id)
    print(f"  Repos: {new_docs} docs nuevos, {updated_docs} actualizados")
    
    new_nodes = 0
    
    for session in sessions:
        data = session['data']
        
        # Guard: nunca ingerir dumps de estado del motor (grafos, logs de evolución)
        if isinstance(data, dict) and ('nodes' in data or 'stats' in data):
            continue

        # Extract meaningful content (skip empty/metadata-only files)
        content = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)
        
        if len(content.strip()) < 100:
            continue
            
        # Segment by topic
        segments = segment_by_topic(content, max_segments=3)
        
        for segment in segments:
            topic = segment['topic'] or "UNKNOWN"
            text = segment['text']
            
            if not text or len(text.strip()) < 20:
                continue
            
            # Classify hierarchy
            hierarchy = classify_hierarchy(topic)
            
            # FIX: hash() de Python es aleatorio entre procesos (PYTHONHASHSEED),
            # así que el mismo texto generaba un node_id NUEVO en cada corrida del
            # cron y la dedup nunca funcionaba. sha256 es estable entre procesos.
            topic_hash = hashlib.sha256(text[:50].encode('utf-8')).hexdigest()[:8]
            session_name = session['file'].split(os.sep)[-1].replace('.json', '')
            node_id = f"CONVERSATION.SESSION.{session_name}.TOPIC.{topic_hash}"
            
            # Check if already indexed (dedup estable entre corridas)
            if node_id in by_id:
                continue
            
            engine_put(
                engine, by_id, node_id, text[:2000],
                hierarchies=[hierarchy],
                node_type='CONCEPT',
                confidence=0.8,
                tags=['conversation', 'auto_indexed', os.path.basename(session['file'])],
            )
            new_nodes += 1
    
    # P1.10: repintar cadenas de tono ANTES de persistir — todo nodo sale
    # con hue_origin verificable (invariante permanente, 0 huérfanos).
    from omega_cube.color_chain import ColorChain
    _chain = ColorChain(engine)
    _chain.assign_axiom_hues()
    _prop = _chain.propagate()
    print(f"  Cadenas de tono: {len(_prop['colored'])} coloreados, "
          f"{len(_prop['orphans'])} huerfanos")

    # Guardar en el STORE ÚNICO del motor consultable (omega_cube_memory.json).
    # engine.save() ya no persiste firmas holográficas (compactación 2026-08-09).
    engine.save()
    
    print(f"  Nuevos nodos creados: {new_nodes}")
    print(f"  Total nodos en Omega-Cube: {len(engine.nodes)}")
    print("[OK] omega_auto_indexer completado")


if __name__ == "__main__":
    main()
