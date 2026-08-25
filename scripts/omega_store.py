"""omega_store.py — Store ÚNICO y compartido para los scripts de mantenimiento.

FIX split-brain 2026-08-09: antes cada script leía/escribía cube_state.json,
un almacén que el motor consultable (omega_cube_mcp_server / OmegaCubeEngine)
NUNCA lee. Todo contenido escrito ahí era invisible para las queries.

Ahora TODOS los scripts usan este módulo, que opera sobre
memory/omega_cube_memory.json — el mismo archivo que carga el engine.

El formato es compatible: dict de nodos keyed por node_id con los campos de
TensorNode.to_dict(). Los campos extra que añaden los scripts (repo_path,
content_hash, etc.) se ignoran al cargar en el engine (from_dict los descarta),
por eso los datos que deben sobrevivir (hashes, rutas) van en `tags`.
"""

import json
import os
from pathlib import Path

PROJECT_PATH = os.path.expanduser(r"~/.hermes/axioma-omega-protocol")
MEMORY_PATH = Path(PROJECT_PATH) / "memory" / "omega_cube_memory.json"


def load_state() -> dict:
    """Carga el estado del motor desde el store único."""
    if not MEMORY_PATH.exists():
        return {"nodes": {}, "axiom_ids": [], "stats": {}, "associations": []}
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalizar estructura esperada por los scripts de mantenimiento
    data.setdefault("nodes", {})
    data.setdefault("associations", [])
    data.setdefault("stats", {})
    return data


def save_state(state: dict) -> None:
    """Guarda el estado en el store único (sin firmas holográficas)."""
    # Las firmas son derivables de content+hierarchy; el engine las recalcula
    # en load(). No persistirlas ahorra ~90% del tamaño del archivo.
    for node in state.get("nodes", {}).values():
        node.pop("holographic_signature", None)
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
