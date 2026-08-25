"""
Provenance Export - W3C PROV-O compliant export for Omega-Cube nodes.

Exports node provenance in standardized formats (JSON-LD, CSV, RDF).
Adapted from Semantica's audit trail pattern.

Author: Omega-Cube Research  
Date: 2026-06-29
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


class ProvenanceExporter:
    """
    Export Omega-Cube node provenance in multiple formats.
    
    Supports:
    - W3C PROV-O JSON-LD (standard)
    - CSV (for spreadsheets/analysis)
    - Simple JSON (for debugging)
    """
    
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else None
    
    def export_prov_o_jsonld(
        self, 
        node_id, 
        node_data,
        activity_data=None
    ):
        """Export node provenance in W3C PROV-O JSON-LD format."""
        entity = {
            "@context": "https://www.w3.org/ns/prov.jsonld",
            "@type": "prov:Entity",
            "prov:id": f"urn:uuid:{node_id}",
            "prov:value": node_data.get("content", ""),
            "prov:generatedAtTime": node_data.get(
                "timestamp", datetime.utcnow().isoformat()
            )
        }
        
        if activity_data:
            entity["prov:wasGeneratedBy"] = [
                {
                    "@type": "prov:Activity",
                    "prov:id": f"urn:uuid:{activity_data.get('activity_id', 'unknown')}",
                    "prov:endedAtTime": activity_data.get("timestamp"),
                    "prov:wasAssociatedWith": [
                        {
                            "@type": "prov:Agent",
                            "prov:id": f"urn:uuid:{aid}"
                        }
                        for aid in activity_data.get("agents", [])
                    ]
                }
            ]
        
        if node_data.get("source_ids"):
            entity["prov:wasDerivedFrom"] = [
                {
                    "@type": "prov:Entity",
                    "prov:id": f"urn:uuid:{sid}"
                }
                for sid in node_data["source_ids"]
            ]
        
        return entity
    
    def export_csv(self, nodes, output_path="provenance_export.csv"):
        """Export multiple nodes to CSV format."""
        if not nodes:
            return "No data to export"
        
        columns = [
            "node_id", "content", "category", "decision_type",
            "outcome", "confidence", "timestamp", "agent_id",
            "reasoning_summary"
        ]
        
        csv_rows = []
        for node in nodes:
            row = {
                "node_id": node.get("id", ""),
                "content": self._escape_csv(node.get("content", "")),
                "category": node.get("metadata", {}).get("decision_type", ""),
                "decision_type": node.get("metadata", {}).get("decision_type", ""),
                "outcome": (node.get("content") or "")[:200],
                "confidence": node.get("confidence", ""),
                "timestamp": node.get("metadata", {}).get("timestamp", ""),
                "agent_id": node.get("metadata", {}).get("agent_id", ""),
                "reasoning_summary": self._escape_csv(
                    "\n".join(
                        node.get("metadata", {}).get("reasoning_chain", [])
                    )[:100]
                )
            }
            csv_rows.append(row)
        
        if self.output_dir:
            output_path = str(self.output_dir / output_path)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(",".join(columns) + "\n")
            for row in csv_rows:
                values = [self._escape_csv(str(row[col])) for col in columns]
                f.write(",".join(values) + "\n")
        
        return output_path
    
    def export_simple_json(self, node_id, node_data, output_path=None):
        """Export node as simple JSON (for debugging)."""
        export_data = {
            "node_id": node_id,
            "content": node_data.get("content"),
            "metadata": node_data.get("metadata", {}),
            "provenance_chain": self._build_simple_provenance(node_data)
        }
        
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            return output_path
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def _build_simple_provenance(self, node_data):
        """Build simple provenance chain for debugging."""
        chain = []
        meta = node_data.get("metadata", {})
        
        chain.append({
            "type": "GENERATION",
            "timestamp": meta.get("timestamp"),
            "agent": meta.get("agent_id")
        })
        
        for source_id in meta.get("source_ids", []):
            chain.append({
                "type": "DERIVATION_FROM",
                "source_node": source_id
            })
        
        return chain
    
    def _escape_csv(self, text):
        """Escape text for CSV format."""
        if not text:
            return ""
        if any(char in text for char in [",", "\n", '"']):
            return f'"{text.replace(chr(34), chr(34)+chr(34))}"'
        return str(text)
