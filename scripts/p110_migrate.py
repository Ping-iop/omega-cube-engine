"""Migracion P1.10: huerfanos a 0.

1) Eliminar 2 nodos basura (cache de embeddings ingerido + conversacion efimera).
2) Re-hogar 3 nodos HERRAMIENTAS.* bajo DEV.TOOLS.* (DEV tiene axioma).
3) Repintar todo el grafo con ColorChain.propagate().
4) Verificar invariante: 0 nodos no-AXIOM sin hue_origin.
"""
import sys
sys.path.insert(0, '.')
from omega_cube.engine import OmegaCubeEngine
from omega_cube.color_chain import ColorChain

TRASH = [
    'CONVERSATION.SESSION.semantic_embeddings.TOPIC.c50d1bfa',
    '81a9e4f94bafd503',  # turno CONVERSATION.SESSION.20260611... (efimero)
]
REHOME = {
    'HERRAMIENTAS.VIDEO.KRILLINAI': 'DEV.TOOLS.VIDEO.KRILLINAI',
    'HERRAMIENTAS.3D.MODLY': 'DEV.TOOLS.3D.MODLY',
    'HERRAMIENTAS.DISENO.SKILLUI': 'DEV.TOOLS.DISENO.SKILLUI',
}

eng = OmegaCubeEngine()
eng.load()
print('antes:', len(eng.nodes), 'nodos')

# --- 1) basura ---
for nid in TRASH:
    if nid in eng.nodes:
        del eng.nodes[nid]
        print('eliminado:', nid)

# --- 2) re-home ---
for nid, n in eng.nodes.items():
    if n.primary_hierarchy in REHOME:
        nuevo = REHOME[n.primary_hierarchy]
        n.hierarchies = [nuevo]
        n.holographic_signature = eng.holographic.encode_node(n.content, nuevo)
        print(f're-hogado: {n.primary_hierarchy} -> {nuevo}')

# --- 3) repintar ---
chain = ColorChain(eng)
chain.assign_axiom_hues()
res = chain.propagate()
print('coloreados:', len(res['colored']), '| huerfanos:', res['orphans'])

# --- 4) invariante ---
huerfanos = [nid for nid, n in eng.nodes.items()
             if n.node_type != 'AXIOM' and not getattr(n, 'hue_origin', None)]
print('invariante post-migracion: huerfanos =', len(huerfanos), huerfanos[:5])

if len(huerfanos) == 0:
    eng.save()
    print('SAVE_OK')
else:
    print('NO_SAVE: quedan huerfanos')
