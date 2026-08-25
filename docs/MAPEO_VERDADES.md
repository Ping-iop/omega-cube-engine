# MAPEO OFICIAL: Capas de certeza del protocolo ↔ estructuras del motor

> Fuente de verdad de la correspondencia entre el **Axioma-Omega Protocol**
> (`src/core/`: `Axiom`, `AxiomLayer`, `ValidationVerdict`) y el **motor Omega-Cube**
> (`omega_cube/`: `TensorNode.node_type`, `ColorChain`, `ValidityGate`).
>
> Decisión P2.D (2026-08-25): se documenta el mapeo en vez de extraer dataclasses
> compartidas. Razón: son ~10 definiciones estables repartidas en dos código bases
> desplegadas por separado; un módulo compartido añadiría un acoplamiento de
> empaquetado sin beneficio real hoy.

## 1. Capas de certeza ↔ tipos de nodo y saturación

| Protocolo (`AxiomLayer`) | Motor (`node_type`) | Saturación (`ColorChain`) | Confianza |
|---|---|---|---|
| `ATOMIC = 0` | `AXIOM` (raíz de cadena) | `sat = λ⁰ = 1.0` (origen) | obligatoria `1.0` |
| `DOMAIN = 1` | `CONCEPT` a profundidad 1 del axioma | `sat = λ¹` | `0.0–1.0` verificada |
| `SITUATIONAL = 2` | `INSTANCE` / nodos contextuales (profundidad ≥ 2) | `sat = λ^depth`, piso `sat_min` | depende de entorno |
| `CREATIVE = 3` | híbridos (`create_hybrid`, concentración < 1) | `sat = concentración` del resultante | especulativa |

Donde:
- `depth = max(0, niveles_jerárquicos_nodo − niveles_axioma)` (distancia al axioma raíz).
- `λ` es el factor de decaimiento de `ColorChain` (por defecto 0.85); trunca en `sat_min`.

## 2. Veredictos (identidad exacta de strings)

| Protocolo (`ValidationVerdict`) | Motor (`GateVerdict.verdict`) | Semántica |
|---|---|---|
| `APPROVED` | `"APPROVED"` | compatible con axiomas / linaje verificado |
| `VETOED`   | `"VETOED"`   | contradicción con Capa 0 → bloqueado |
| `FLAGGED`  | `"FLAGGED"`  | permitido con advertencia (baja alineación / embeddings degradados) |

Los strings son idénticos en ambos lados a propósito: cualquier cambio debe
reflejarse aquí primero.

## 3. NO-equivalencias explícitas (no confundir)

- `confidence` (protocolo) ≠ `saturation` (motor): la primera es **certeza
  epistémica** declarada con fuentes; la segunda es **pureza cromática** derivada
  de la profundidad derivacional. No son intercambiables ni se calculan una de otra.
- `hue_origin` (id del axioma raíz de cadena) no tiene equivalente en el protocolo;
  es infraestructura de trazabilidad del motor.
- Un nodo `CONCEPT` puede sustentar una afirmación `DOMAIN` pero la capa la decide
  el protocolo al registrar el axioma, nunca la posición en el grafo.

## 4. Regla de sincronización

Si cambias capas, veredictos o la fórmula de saturación en cualquiera de los dos
lados, actualiza esta tabla en el mismo commit en ambos repos:
- Protocolo: `Ping-iop/Axioma-Omega_Protocol`
- Motor: `Ping-iop/omega-cube-engine`
