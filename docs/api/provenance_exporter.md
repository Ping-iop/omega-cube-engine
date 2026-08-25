# ProvenanceExporter API Reference

**Ubicación:** `omega_cube/provenance_export.py`  
**Versión:** 1.0.0  
**Última actualización:** 2026-06-29  

---

## Resumen

`ProvenanceExporter` exporta proveniencia de nodos en formatos estandarizados (JSON-LD, CSV, JSON simple) conforme a W3C PROV-O para interoperabilidad con sistemas externos.

---

## Inicialización

```python
from omega_cube.provenance_export import ProvenanceExporter

exporter = ProvenanceExporter()
```

---

## Métodos Principales

### `export_prov_o_jsonld()` — Exportación W3C PROV-O JSON-LD

**Firma:**
```python
def export_prov_o_jsonld(
    self,
    node_id: str,
    node_data: dict,
    activity_data: dict = None
) -> dict  # JSON-LD conforme a W3C PROV-O
```

Genera documento de proveniencia en formato W3C PROV-O JSON-LD.

**Parámetros:**

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `node_id` | `str` | Sí | ID del nodo |
| `node_data` | `dict` | Sí | Datos completos del nodo (desde `to_dict()`) |
| `activity_data` | `dict` | No | Metadata de actividad opcional (timestamp, agente) |

**Estructura del output:**
```python
{
    "@context": "https://www.w3.org/ns/prov.jsonld",
    "@type": "prov:Entity",
    "@id": "<node_id>",
    "prov:type": "<category o node_type>",
    "prov:value": {
        "content": "...",
        "hierarchies": [...],
        ...
    },
    "prov:generatedAtTime": "<ISO timestamp>"
}
```

**Ejemplo:**
```python
node = engine.nodes["a1b2c3d4"]
prov_jsonld = exporter.export_prov_o_jsonld(
    node_id=node.node_id,
    node_data=node.to_dict()
)
# → Dict JSON-LD listo para consumo externo
```

---

### `export_csv()` — Exportación Múltiple a CSV

**Firma:**
```python
def export_csv(
    self,
    nodes: list[dict],
    output_path="provenance_export.csv"
) -> str  # Path del archivo o "No data to export"
```

Exporta múltiples nodos a un archivo CSV con columnas estándar de proveniencia.

**Parámetros:**

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `nodes` | `list[dict]` | Sí | Lista de dicts (desde `to_dict()`) |
| `output_path` | `str` | No | Ruta del archivo CSV (default: "provenance_export.csv") |

**Columnas CSV generadas:**
```
node_id,content,hierarchies,node_type,confidence,tags,outcome,category,scenario
```

**Ejemplo:**
```python
nodes_data = [n.to_dict() for n in engine.nodes.values()]
path = exporter.export_csv(nodes_data, "my_provenance.csv")
print(f"Exportado a: {path}")
# → Exportado a: my_provenance.csv (18 rows)
```

---

### `export_simple_json()` — Exportación Simple para Debugging

**Firma:**
```python
def export_simple_json(
    self,
    node_id: str,
    node_data: dict,
    output_path: str = None
) -> str  # JSON string o path del archivo
```

Exporta un nodo individual en formato JSON simple (sin contexto PROV-O). Útil para debugging y desarrollo.

**Parámetros:**

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `node_id` | `str` | Sí | ID del nodo |
| `node_data` | `dict` | Sí | Datos del nodo |
| `output_path` | `str` | No | Si se provee, escribe a archivo; si no, retorna string JSON |

**Ejemplo:**
```python
# Retorna string JSON directamente
json_str = exporter.export_simple_json("a1b2c3d4", node_data)

# O escribe a archivo
path = exporter.export_simple_json("a1b2c3d4", node_data, "debug_node.json")
```

---

## Ejemplo Completo: Exportación Multi-Formato

```python
from omega_cube.provenance_export import ProvenanceExporter
import json

exporter = ProvenanceExporter()

# Obtener datos de todos los nodos
all_nodes = [n.to_dict() for n in engine.nodes.values()]

# 1. Exportar primer nodo a JSON-LD (W3C PROV-O)
node0 = all_nodes[0]
prov_jsonld = exporter.export_prov_o_jsonld(node0["id"], node0)
print(json.dumps(prov_jsonld, indent=2))

# 2. Exportar todos los nodos a CSV
csv_path = exporter.export_csv(all_nodes, "all_provenance.csv")
print(f"CSV exportado: {csv_path}")

# 3. Debugging de un nodo específico
node_debug = exporter.export_simple_json(
    all_nodes[0]["id"], 
    all_nodes[0],
    output_path="debug_node.json"
)
```

---

**Ver también:** [DecisionNode API](decision_node.md), [TensorNode API](tensor_node.md)
