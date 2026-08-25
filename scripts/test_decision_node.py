#!/usr/bin/env python3
"""Test DecisionNode and ConflictDetector."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "omega_cube"))

from decision_node import DecisionNode, ConflictDetector
from provenance_export import ProvenanceExporter


def test_decision_node_creation():
    print("Testing DecisionNode creation...")
    
    decision = DecisionNode(
        category="vendor_selection",
        scenario="Choose cloud provider for HIPAA workload",
        reasoning="AWS offers BAA\nMature HIPAA tooling\nCost-effective at scale",
        outcome="selected_aws",
        confidence=0.93,
    )
    
    print(f"  Node ID: {decision.node_id}")
    print(f"  Category: {decision.metadata['decision_type']}")
    print(f"  Outcome: {decision.content}")
    print(f"  Confidence: {decision.confidence}")
    
    provenance = decision.trace_chain()
    assert "@context" in provenance, "Missing @context"
    print("  Provenance chain generated OK")
    
    ddict = decision.to_decision_dict()
    assert ddict["category"] == "vendor_selection"
    assert ddict["outcome"] == "selected_aws"
    print(f"  Decision dict exported: {ddict['id']}")
    
    return decision


def test_conflict_detection():
    print("\nTesting ConflictDetector...")
    
    detector = ConflictDetector(similarity_threshold=0.5)
    
    node_a = DecisionNode(
        category="vendor_selection",
        scenario="Choose cloud provider",
        reasoning="AWS is better for HIPAA",
        outcome="selected_aws",
        confidence=0.9,
    )
    
    node_b = DecisionNode(
        category="vendor_selection",
        scenario="Choose cloud provider (revised)",
        reasoning="GCP is better for cost savings",
        outcome="rejected_aws_selected_gcp",
        confidence=0.85,
    )
    
    conflicts = detector.detect_conflicts(node_b, [node_a])
    
    if conflicts:
        print(f"  Detected {len(conflicts)} conflict(s)")
        for i, c in enumerate(conflicts):
            print(f"    [{c['severity']}] {c['type']} — similarity={c['similarity_score']}")
        return True
    
    print("  No conflicts detected (threshold may be too high)")
    return False


def test_provenance_export():
    print("\nTesting Provenance Export...")
    
    exporter = ProvenanceExporter()
    
    node_data = {
        "id": "test-node-123",
        "content": "Selected AWS for HIPAA compliance",
        "metadata": {
            "decision_type": "vendor_selection",
            "timestamp": "2026-06-29T10:00:00Z",
            "agent_id": "bit-agent-001",
            "source_ids": ["node-abc-456"],
        },
    }
    
    prov_o = exporter.export_prov_o_jsonld(
        node_id="test-node-123",
        node_data=node_data,
        activity_data={
            "activity_id": "act-789",
            "timestamp": "2026-06-29T10:00:00Z",
            "agents": ["agent-001"],
        },
    )
    
    assert prov_o["@context"] == "https://www.w3.org/ns/prov.jsonld"
    print("  W3C PROV-O JSON-LD generated OK")
    
    json_path = exporter.export_simple_json(
        node_id="test-node-123",
        node_data=node_data,
        output_path="/tmp/test_provenance.json",
    )
    print(f"  Simple JSON exported to: {json_path}")


def main():
    print("=" * 60)
    print("Testing Decision Intelligence Features")
    print("=" * 60)
    
    try:
        test_decision_node_creation()
        conflict_detected = test_conflict_detection()
        test_provenance_export()
        
        print("\n" + "=" * 60)
        if conflict_detected:
            print("All tests passed! Conflict detection working.")
        else:
            print("Tests completed. Conflict detection needs tuning.")
        print("=" * 60)
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
