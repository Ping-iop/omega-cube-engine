# GrayScaleValidator — Multi-bit Truth Assessment

Inspired by H-Bit steganography principles. Instead of binary 0/1 verification, each graph node carries a **multi-bit "gray scale"** encoding degrees of truth across multiple dimensions. A node scoring 75% on "factuality" and 90% on "relevance" is treated differently than one scoring 90% on both — the gray scale preserves nuance that binary classification loses.

## Class: `GrayScaleValidator`

**Source:** `omega_cube/grayscale.py` (239 lines, 8,639 bytes)  
**Author:** Omega-Cube Research  
**Date:** 2026-06-11

### Overview

Evaluates nodes along 6 truth dimensions, producing a "gray scale profile" — a dict mapping dimension name → score (0–100). This replaces boolean true/false with a multi-bit confidence spectrum.

```python
from omega_cube.grayscale import GrayScaleValidator

validator = GrayScaleValidator(default_scale=50)  # 50 = neutral

profile = validator.evaluate_node(
    node=my_tensor_node,
    query="evony generals",
    axioms=[axiom1, axiom2],
    related_nodes=[node_a, node_b],
)
# → {"factuality": 85.0, "relevance": 92.0, "recency": 78.0, ...}

composite = validator.composite_score(profile)
# → 86.3 (weighted average across dimensions)
```

### Dimensions

| Dimension | What it measures | Default weight |
|-----------|-----------------|----------------|
| `factuality` | Anchored to known axioms | 0.35 |
| `relevance` | Query alignment via keyword overlap | 0.25 |
| `recency` | Temporal freshness (half-life: 30 days) | 0.10 |
| `coherence` | Internal consistency with related nodes | 0.15 |
| `provenance` | Traceability to source (tags check) | 0.10 |
| `specificity` | Detail granularity (length + specificity indicators) | 0.05 |

### Key Methods

#### `evaluate_node(node, query="", axioms=None, related_nodes=None) → dict[str, float]`

Evaluates a node across all 6 dimensions. Returns `{dimension: score}` where each score is 0–100.

```python
profile = validator.evaluate_node(
    node=engine.nodes[node_id],
    query="how to optimize SDXL prompts",
    axioms=[axiom_nodes],
)
# → {"factuality": 72.0, "relevance": 95.0, "recency": 88.0,
#    "coherence": 67.0, "provenance": 40.0, "specificity": 73.0}
```

#### `composite_score(gray_profile, weights=None) → float`

Combines gray-scale dimensions into a single confidence score (0–100). Uses weighted average; custom weights can override defaults.

```python
# Default weights: factuality=0.35, relevance=0.25, recency=0.10, coherence=0.15, provenance=0.10, specificity=0.05
score = validator.composite_score(profile)  # → 86.3

# Custom weights (e.g., prioritize factuality for safety-critical queries)
custom = {"factuality": 0.50, "relevance": 0.20, "recency": 0.05, "coherence": 0.15, "provenance": 0.05, "specificity": 0.05}
score = validator.composite_score(profile, weights=custom)
```

#### `partial_evidence_score(gray_profile, available_dimensions) → float`

H-Bit principle: you don't need all dimensions for a useful assessment. Computes confidence from partial evidence only.

```python
# Even with just factuality + relevance, we get actionable confidence
partial = validator.partial_evidence_score(profile, ["factuality", "relevance"])
# → 83.5 (average of available dimensions)
```

#### `verify_against_axioms(node, axioms, threshold=60.0) → tuple[bool, float, str]`

Verifies a node's factual grounding against known axioms. Returns `(is_verified, confidence, explanation)`. Uses keyword overlap (>0.3 threshold) for matching.

```python
verified, confidence, explanation = validator.verify_against_axioms(
    node=my_node,
    axioms=[axiom_nodes],
    threshold=60.0,
)
# → (True, 78.5, "Matched 3 axioms (avg: 78.5%)")
```

#### `compute_gray_scale_hash(gray_profile) → str`

Creates a SHA-256 hash fingerprint of the gray-scale profile. Quantizes each dimension to 5-bit precision (32 levels), packs into integer, hashes. Enables quick comparison without storing full profiles.

```python
hash1 = validator.compute_gray_scale_hash(profile_a)  # "a3f29b01c4e7"
hash2 = validator.compute_gray_scale_hash(profile_b)  # "d8e1f502a6b3"
# Different hashes → different truth profiles (even if composite scores are similar)
```

### Internal Assessors

| Method | Signal | Algorithm |
|--------|--------|-----------|
| `_assess_factuality` | Axiom grounding | Keyword overlap with axioms, penalized by 50% if unverified |
| `_assess_relevance` | Query alignment | Keyword Jaccard similarity between node content and query |
| `_assess_recency` | Temporal freshness | Exponential decay with 30-day half-life: `100 * (0.5)^(age_days/30)` |
| `_assess_coherence` | Internal consistency | Average keyword overlap with related nodes |
| `_assess_provenance` | Source traceability | Tag-based scoring: "axiom"→95%, "verified"→85%, "source"/"citation"→70%, else→40% |
| `_assess_specificity` | Detail granularity | Average of (content_length/5, specificity_indicators×20) where indicators are numbers/dates/paths/config references |

### Integration with Engine

The engine creates a `GrayScaleValidator` instance and attaches it as `engine.gray_validator`. It's used in:
- **MCP server** (`omega_cube_mcp_server.py`): `omega_cube_verify` tool recomputes gray-scale per node on demand
- **Diffusion sampler**: Gray scale composites rank results
- **AutoResearch loop**: Uses composite scores to identify nodes needing re-evaluation

### H-Bit Connection

The gray scale design is directly inspired by H-Bit steganography:
- **Multi-bit encoding**: Like H-Bit's multi-bit truth assessment, each dimension captures a different aspect of "truth"
- **Partial evidence principle**: Even checking 2 of 6 dimensions provides actionable confidence — same as H-Bit working with partial file checks
- **Hash fingerprinting**: Gray scale hash enables quick deduplication/comparison without full profile storage
