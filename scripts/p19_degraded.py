"""Smoke P1.9 camino degradado con ensamblado NO vacio (tau=0)."""
import sys
sys.path.insert(0, '.')
from omega_cube.engine import OmegaCubeEngine
from omega_cube.cube_move import CubeMover

eng = OmegaCubeEngine()
eng.load()
print('nodos:', len(eng.nodes))

# Sin embedder -> 100% holografico (todos los nodos marcados degradados)
mover = CubeMover(eng)
res = mover.cube_move('calcular la derivada de x al cuadrado', tau=0.0)
print('embed_source:', res.embed_source)
print('seleccion:', len(res.all_node_ids()), 'nodos')
print('verdict:', res.verdict)
print('reason:', res.verdict_reason)

flags = sum(1 for n in eng.nodes.values()
            if getattr(n, '_degraded_embedding', False))
print('nodos marcados _degraded_embedding:', flags)
assert res.embed_source == 'holographic_fallback'
assert len(res.all_node_ids()) > 0, 'ensamblado deberia ser no-vacio con tau=0'
assert 'degradados' in res.verdict_reason, 'el reporte de degradacion debe aparecer'
print('P1.9 DEGRADADO OK')
