"""
DecisionNode - First-class decision objects with W3C PROV-O provenance.

Extends TensorNode with decision-specific metadata and traceability.
Adapted from Semantica's Decision Intelligence pattern.

Author: Omega-Cube Research  
Date: 2026-06-29
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from .tensor_node import TensorNode


class DecisionNode(TensorNode):
    """
    A node representing a decision with full traceability.
    
    Key features:
    - Structured metadata (category, scenario, reasoning chain)
    - W3C PROV-O provenance tracking
    - Confidence scoring
    - Conflict detection support
    
    Usage:
        decision = DecisionNode(
            category="vendor_selection",
            scenario="Choose cloud provider for HIPAA workload",
            reasoning="AWS offers BAA, mature HIPAA tooling...",
            outcome="selected_aws",
            confidence=0.93
        )
    """

    def __init__(
        self,
        category: str,
        scenario: str,
        reasoning: str,
        outcome: str,
        confidence: float = 0.9,
        hierarchies=None,
        tags=None,
        metadata=None,
    ):
        if hierarchies is None:
            hierarchies = [f"DECISIONS.{category.upper()}"]

        super().__init__(
            content=outcome,
            hierarchies=hierarchies,
            tensor_position=[],
            node_type="CONCEPT",
            confidence=confidence,
            tags=tags or [],
            associations=[],
        )

        self.metadata = metadata or {}
        self.metadata.update({
            "decision_type": category,
            "scenario_description": scenario,
            "reasoning_chain": reasoning.split("\n"),
            "outcome_traceability": f"Selected {outcome} based on: {reasoning}",
            "confidence_score": confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "provenance_type": "w3c_prov_o",
        })

    def trace_chain(self) -> Dict[str, Any]:
        return {
            "@context": "https://www.w3.org/ns/prov.jsonld",
            "@type": "prov:Entity",
            "prov:id": f"urn:uuid:{self.node_id}",
            "prov:value": self.content,
            "prov:generatedAtTime": self.metadata.get("timestamp"),
        }

    def to_decision_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "category": self.metadata["decision_type"],
            "scenario": self.metadata["scenario_description"],
            "reasoning": "\n".join(self.metadata["reasoning_chain"]),
            "outcome": self.content,
            "confidence": self.confidence,
            "timestamp": self.metadata.get("timestamp"),
            "provenance": self.trace_chain(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionNode":
        return cls(
            category=data["category"],
            scenario=data["scenario"],
            reasoning=data["reasoning"],
            outcome=data["outcome"],
            confidence=data.get("confidence", 0.9),
            hierarchies=[f"DECISIONS.{data['category'].upper()}"],
        )


# ConflictDetector v2 is in conflict_detector_v2.py
# Import it for backward compatibility
from .conflict_detector_v2 import ConflictDetectorV2 as ConflictDetector
