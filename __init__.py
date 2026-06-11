"""
Omega-Cube Engine — Multi-dimensional hierarchical graph memory.

A quantum leap beyond vector databases and flat knowledge graphs:
- Tensor Hierarchies: nodes exist in N-dimensional hierarchy spaces
- Holographic Encoding: compressed signatures for O(1) approximate search
- Quantum-Inspired Annealing: dynamic topology optimization
- Diffusion Graph Sampling: parallel non-autoregressive retrieval
- Gray-Scale Validation: multi-bit truth assessment (H-Bit inspired)

Usage:
    from omega_cube import OmegaCubeEngine
    
    engine = OmegaCubeEngine()
    engine.add_node("SDXL is better for quality", 
                    hierarchies=["COMFYUI.WORKFLOWS.GENERATION",
                                "QUALITY.IMAGE_RESOLUTION.HIGH"],
                    tensor_position=[0.7, 0.8])
    
    results = engine.query("best image generation workflow", mode="diffusion")
    patterns = engine.find_patterns("multi-topic query")
"""

from .engine import OmegaCubeEngine
from .tensor_node import TensorNode, TensorIndex
from .holographic import HolographicEncoder
from .annealer import QuantumInspiredAnnealer, CubeRotator, PatternEmergence
from .diffusion_sampler import DiffusionGraphSampler
from .grayscale import GrayScaleValidator
from .autoresearch import AutoResearchLoop

__version__ = "1.0.0"
__author__ = "Omega-Cube Research"
__all__ = [
    "OmegaCubeEngine",
    "TensorNode",
    "TensorIndex",
    "HolographicEncoder",
    "QuantumInspiredAnnealer",
    "CubeRotator",
    "PatternEmergence",
    "DiffusionGraphSampler",
    "GrayScaleValidator",
    "AutoResearchLoop",
]
