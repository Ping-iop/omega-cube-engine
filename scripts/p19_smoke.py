import sys
sys.path.insert(0, '.')
from omega_cube.engine import OmegaCubeEngine
from omega_cube.cube_move import CubeMover

eng = OmegaCubeEngine()
eng.load()
print('nodos:', len(eng.nodes))

mover = CubeMover(eng)
res = mover.cube_move('calcular la derivada de x al cuadrado')
print('embed_source:', res.embed_source)
print('verdict:', res.verdict, '|', res.verdict_reason)
print('seleccion:', len(res.all_node_ids()), 'nodos')
