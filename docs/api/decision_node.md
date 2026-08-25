# DecisionNode API Reference

**Ubicación:** `omega_cube/decision_node.py` (106 líneas, 3,348 bytes)  
**Autor:** Omega-Cube Research  
**Última actualización:** 2026-06-29  

## Resumen

`DecisionNode` extiende `TensorNode` con metadata estructurada para decisiones: categoría, escenario, cadena de razonamiento y resultado. Genera trazabilidad W3C PROV-O JSON-LD y serialización extendida. Importa `ConflictDetectorV2` como alias `ConflictDetector` para compatibilidad hacia atrás.

## Clase: `DecisionNode(TensorNode)`

### Constructor

```python
from omega_cube.decision_node import DecisionNode

decision = DecisionNode(
    category="vendor_selection",           # Categoría de decisión (obligatorio)
    scenario="Choose cloud provider for HIPAA workload",  # Descripción del escenario
    reasoning="AWS offers BAA, mature HIPAA tooling...",     # Cadena de razonamiento
    outcome="selected_aws",                # Resultado codificado (obligatorio)
    confidence=0.93,                       # Default: 0.9
    hierarchies=[f"DECISIONS.{category.upper()}"],  # Default: ["DECISIONS.VENDOR_SELECTION"]
    tags=["cloud", "hipaa", "vendor"],     # Default: []
    metadata={...},                        # Metadata extendida opcional
)
```

**Nota:** `content` se establece internamente como el valor de `outcome`. Las `hierarchies` por defecto son `[f"DECISIONS.{category.upper()}"]`.

### Atributos Extendidos (además de TensorNode)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `metadata` | `dict[str, Any]` | Metadata extendida con todos los campos de decisión |
| `content` | `str` | Heredado de TensorNode — igual a `outcome` |
| `hierarchies` | `list[str]` | Heredado — default `[f"DECISIONS.{category.upper()}"]` |

### Metadata Interna (almacenada en `self.metadata`)

```python
{
    "decision_type": str,                  # category pasado al constructor
    "scenario_description": str,           # scenario pasado al constructor
    "reasoning_chain": list[str],          # reasoning.split("\n") — líneas separadas
    "outcome_traceability": str,           # f"Selected {outcome} based on: {reasoning}"
    "confidence_score": float,             # confidence del constructor
    "timestamp": str,                      # datetime.utcnow().isoformat()
    "provenance_type": str,                # fijo: "w3c_prov_o"
}
```

### Método: `trace_chain()` — Trazabilidad W3C PROV-O

```python
def trace_chain(self) -> dict  # JSON-LD conforming to W3C PROV-O
```

Genera documento de proveniencia en formato W3C PROV-O JSON-LD.

**Estructura del output:**
```python
{
    "@context": "https://www.w3.org/ns/prov.jsonld",
    "@type": "prov:Entity",
    "prov:id": f"urn:uuid:{self.node_id}",
    "prov:value": self.content,  # = outcome
    "prov:generatedAtTime": self.metadata["timestamp"],
}
```

### Método: `to_decision_dict()` — Serialización Extendida

```python
def to_decision_dict(self) -> dict  # Estructura de decisión completa
```

Serializa a diccionario con todos los campos extendidos.

**Estructura del output:**
```python
{
    "id": str,                       # self.node_id (UUID)
    "category": str,                 # metadata["decision_type"]
    "scenario": str,                 # metadata["scenario_description"]
    "reasoning": str,                # "\n".join(metadata["reasoning_chain"])
    "outcome": str,                  # self.content (= outcome del constructor)
    "confidence": float,             # self.confidence
    "timestamp": str,                # metadata["timestamp"]
    "provenance": dict,              # resultado de trace_chain()
}
```

### Método Estático: `from_dict()` — Deserialización

```python
@classmethod
def from_dict(cls, data: dict) -> DecisionNode
```

Reconstruye un DecisionNode desde su representación en diccionario. Extrae automáticamente los campos extendidos de decisión del dict.

**Ejemplo:**
```python
# Desde JSON guardado o exportado
restored = DecisionNode.from_dict(json.loads(json_str))
print(restored.category)  # → "vendor_selection"
print(restored.outcome)   # → "selected_aws"
```

## Integración con ConflictDetector v2

`DecisionNode` se integra nativamente con `ConflictDetectorV2`:

| Método | Descripción |
|--------|-------------|
| `decision_node.trace_chain()` | Genera trazabilidad W3C PROV-O JSON-LD |
| `decision_node.to_decision_dict()` | Serializa a dict extendido |
| `DecisionNode.from_dict(d)` | Reconstruye desde dict (estático) |

El detector analiza automáticamente el campo `outcome` del DecisionNode para detectar contradicciones como "selected_aws" vs "rejected_azure".

## Uso Completo: Flujo de Decisión

```python
from omega_cube.decision_node import DecisionNode
from omega_cube.conflict_detector_v2 import ConflictDetectorV2

# 1. Crear decisión con razonamiento completo
decision = DecisionNode(
    category="vendor_selection",
    scenario="Choose cloud provider for HIPAA workload",
    reasoning=(
        "1. AWS ofrece BAA para compliance HIPAA\n"
        "2. Herramientas maduras de seguridad\n"
        "3. Escalabilidad enterprise probada"
    ),
    outcome="selected_aws",
    confidence=0.93,
)

# 2. Trazabilidad W3C PROV-O
prov = decision.trace_chain()
print(prov["@type"])  # → "prov:Entity"
print(prov["prov:id"])  # → "urn:uuid:<node_id>"

# 3. Serialización para persistencia
d = decision.to_decision_dict()
import json
json_str = json.dumps(d, indent=2)

# 4. Reconstruir desde dict
restored = DecisionNode.from_dict(json.loads(json_str))
print(restored.category)  # → "vendor_selection"
print(restored.outcome)   # → "selected_aws"

# 5. Detectar conflictos con otras decisiones existentes
detector = ConflictDetectorV2()
conflicts = detector.detect_conflicts(
    new_node=restored,
    existing_nodes=[existing_decision1, existing_decision2],
)
for c in conflicts:
    print(f"Conflicto detectado: {c['type']} ({c['severity']})")
```

---

**Ver también:** [ConflictDetector v2 API](conflict_detector_v2.md), [TensorNode API](tensor_node.md)
