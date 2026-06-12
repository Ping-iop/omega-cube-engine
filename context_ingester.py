#!/usr/bin/env python3
"""
Omega-Cube Context Ingester — Trocea conversaciones en grafos jerárquicos.

ALTO NIVEL:
  En lugar de meter todo el historial en el prompt (y perder calidad por
  límite de tokens), este script:
  
  1. Trocea la conversación en segmentos por tema
  2. Cada segmento → nodo en Omega-Cube con jerarquía multi-dimensional
  3. Los nodos se conectan por co-ocurrencia (grafo)
  4. Al buscar, Omega-Cube devuelve solo los N nodos más relevantes
  5. = contexto perfecto sin perder detalle

Esto permite conversaciones de 100k+ tokens donde Omega-Cube mantiene
el contexto relevante en menos de 4k tokens de inyección.

ARQUITECTURA:
  Conversación → TopicDetector → Chunker → OmegaCubeEngine.add_node()
                                                    ↓
  Query → PredictiveContextSearch → Top-5 nodos → prompt inyectado
                                                    ↓
                                                  MARP Worker responde

USO:
  python context_ingester.py "raw conversation text"
  python context_ingester.py --session-id abc123 --from-file chat_log.txt
  python context_ingester.py --interactive
"""

import sys, json, re, time, hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, str(Path.home() / ".hermes" / "axioma-omega-protocol"))
sys.path.insert(0, str(Path.home() / ".hermes" / "axioma-omega-protocol" / "omega_cube"))

# ─── Omega-Cube imports ──────────────────────────────────────────
from omega_cube.engine import OmegaCubeEngine
from omega_cube.predictive_search import PredictiveContextSearch

LOG_DIR = Path.home() / ".hermes" / "logs" / "marp_router"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Topic signals ───────────────────────────────────────────────
TOPIC_SIGNALS = {
    "math":          ["math","calculus","derivative","integral","algebra","equation","theorem","probability"],
    "code":          ["python","javascript","rust","function","class","api","docker","kubernetes","async","sql"],
    "science":       ["quantum","physics","chemistry","biology","dna","gene","neuron","evolution","crispr"],
    "engineering":   ["circuit","bridge","mechanical","sensor","motor","structural","pid","controller"],
    "law":           ["contract","nda","patent","copyright","court","statute","compliance","liability"],
    "medical":       ["diagnosis","symptom","treatment","surgery","disease","diabetes","cancer","therapy"],
    "business":      ["revenue","profit","investment","npv","startup","valuation","market","strategy","roi"],
    "philosophy":    ["ethics","kant","free will","consciousness","morality","existential","determinism"],
    "gaming":        ["game","rpg","roguelike","moba","player","mechanic","mmorpg","quest"],
    "language":      ["translate","grammar","syntax","linguistics","poem","essay","literature","spanish"],
    "omega-cube":    ["omega-cube","marp","tensor","holographic","hierarchical","predictive","cube"],
    "h-bit":         ["h-bit","steganography","spectrum","verification","payload","stego"],
    "evony":         ["evony","march","rally","ranged","mounted","monarch","boss"],
    "hermes":        ["hermes","agent","cron","mcp","skill","session","tool"],
}


class ConversationChunker:
    """Trocea conversación en segmentos por tema."""

    def chunk_by_topic(self, text: str, max_chars: int = 500) -> list[dict]:
        """Divide texto en segmentos temáticos.
        Returns: [{topic, hierarchy, content, confidence, timestamp}]
        """
        # Split by speaker turns or paragraphs
        segments = []
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_topic = "general"
        current_content = ""
        current_hierarchy = "general"
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Detect topic shift
            detected = self._detect_topic(para)
            new_topic = detected["topic"]
            new_hierarchy = detected["hierarchy"]
            
            # If topic changed or content too long, flush
            if (new_topic != current_topic and current_content) or \
               (len(current_content) + len(para) > max_chars):
                if current_content:
                    segments.append({
                        "topic": current_topic,
                        "hierarchy": current_hierarchy,
                        "content": current_content.strip(),
                        "confidence": 0.7,
                        "timestamp": datetime.now().isoformat(),
                        "id": hashlib.md5(current_content.encode()).hexdigest()[:12],
                    })
                current_topic = new_topic
                current_hierarchy = new_hierarchy
                current_content = para
            else:
                current_content += "\n\n" + para
        
        # Flush last
        if current_content:
            segments.append({
                "topic": current_topic,
                "hierarchy": current_hierarchy,
                "content": current_content.strip(),
                "confidence": 0.7,
                "timestamp": datetime.now().isoformat(),
                "id": hashlib.md5(current_content.encode()).hexdigest()[:12],
            })
        
        return segments

    def _detect_topic(self, text: str) -> dict:
        """Detect the most likely topic for a text segment."""
        text_lower = text.lower()
        scores = {}
        
        for topic, keywords in TOPIC_SIGNALS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                scores[topic] = count
        
        if not scores:
            return {"topic": "general", "hierarchy": "general"}
        
        best_topic = max(scores, key=scores.get)
        hierarchy = f"{best_topic}.{best_topic}_{int(time.time()) % 1000}"
        return {"topic": best_topic, "hierarchy": hierarchy}


class OmegaContextIngester:
    """Ingiere contexto de conversación en Omega-Cube."""

    def __init__(self):
        self.engine = OmegaCubeEngine()
        self.pcs = PredictiveContextSearch()
        self.chunker = ConversationChunker()
        self.stats = {"nodes_added": 0, "chunks_processed": 0}

    def ingest(self, text: str, source: str = "conversation") -> list[dict]:
        """Ingiere texto en Omega-Cube con jerarquías detectadas."""
        chunks = self.chunker.chunk_by_topic(text)
        added_nodes = []

        for chunk in chunks:
            # Add to Omega-Cube engine
            tensor_pos = [self._hash_to_float(chunk["id"]),
                         0.5,
                         self._confidence_to_pos(chunk["confidence"])]
            
            node_info = self.engine.add_node(
                content=chunk["content"],
                hierarchies=[chunk["hierarchy"], f"source.{source}"],
                tensor_position=tensor_pos,
                node_type=chunk["topic"].upper(),
                confidence=chunk["confidence"],
            )
            
            # Add to PredictiveContextSearch trie
            prefix = chunk["topic"][:4]
            self.pcs.trie.insert(
                text=f"ctx_{chunk['id']}",
                domain=chunk["topic"],
                node_id=str(node_info) if not isinstance(node_info, str) else node_info,
            )
            
            added_nodes.append(chunk)
            self.stats["nodes_added"] += 1
        
        self.stats["chunks_processed"] = len(chunks)
        return added_nodes

    def search_context(self, query: str, top_k: int = 5) -> list[dict]:
        """Busca los N nodos más relevantes para una consulta."""
        # First: PCS prefix search
        pcs_results = self.pcs.predict(query[:8])
        
        # Then: engine query for ranking
        engine_results = self.engine.query(
            query_text=query,
            mode="holographic",
            top_k=top_k,
        )
        
        # Merge: PCS para dominio, engine para ranking
        seen = set()
        results = []
        
        for r in engine_results:
            if r.get("content") and r["content"] not in seen:
                seen.add(r["content"])
                results.append(r)
                if len(results) >= top_k:
                    break
        
        return results

    def to_prompt_context(self, query: str, top_k: int = 5) -> str:
        """Genera contexto en formato prompt para inyectar al worker."""
        nodes = self.search_context(query, top_k)
        if not nodes:
            return ""
        
        parts = []
        parts.append("<context from omega-cube>")
        for i, node in enumerate(nodes, 1):
            content = node.get("content", "")[:300]
            hierarchies = node.get("hierarchies", [])
            if hierarchies:
                parts.append(f"[{i}] ({', '.join(hierarchies)}): {content}")
            else:
                parts.append(f"[{i}]: {content}")
        parts.append("</context>")
        
        return "\n".join(parts)

    def _hash_to_float(self, h: str) -> float:
        return float(int(hashlib.md5(h.encode()).hexdigest()[:8], 16)) / 0xFFFFFFFF

    def _confidence_to_pos(self, conf: float) -> float:
        return 0.3 + conf * 0.5

    def save(self, path: str = None):
        if path is None:
            path = str(Path.home() / ".hermes" / "axioma-omega-protocol" / "omega_cube" / "context_state.json")
        self.engine.save(path)

    def load(self, path: str = None):
        if path is None:
            path = str(Path.home() / ".hermes" / "axioma-omega-protocol" / "omega_cube" / "context_state.json")
        p = Path(path)
        if p.exists():
            self.engine.load(path)


# ─── CLI ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Omega-Cube Context Ingester")
    parser.add_argument("text", nargs="?", help="Text to ingest (or read from stdin/file)")
    parser.add_argument("--session-id", help="Session identifier")
    parser.add_argument("--from-file", help="Read text from file")
    parser.add_argument("--query", help="Search context for this query")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--stats", action="store_true", help="Show engine stats")
    args = parser.parse_args()

    ingester = OmegaContextIngester()
    ingester.load()

    if args.stats:
        stats = ingester.engine.stats()
        print(json.dumps(stats, indent=2))
        sys.exit(0)

    if args.query:
        context = ingester.to_prompt_context(args.query, top_k=4)
        print(context)
        sys.exit(0)

    if args.interactive:
        print("Omega-Cube Context Ingester — Interactive")
        print("Commands: /search <q>  /stats  /save  /load  quit")
        while True:
            try:
                line = input("\n> ").strip()
                if line.lower() in ("quit", "exit"):
                    break
                if line.startswith("/search"):
                    ctx = ingester.to_prompt_context(line[8:].strip())
                    print(ctx)
                elif line == "/stats":
                    print(json.dumps(ingester.engine.stats(), indent=2))
                elif line == "/save":
                    ingester.save()
                    print("Saved")
                elif line:
                    chunks = ingester.ingest(line)
                    print(f"Ingested {len(chunks)} chunks")
            except KeyboardInterrupt:
                break
        sys.exit(0)

    source = args.text or ""
    if args.from_file:
        with open(args.from_file) as f:
            source = f.read()
    elif not source and not sys.stdin.isatty():
        source = sys.stdin.read()

    if source:
        chunks = ingester.ingest(source, source=args.session_id or "cli")
        print(json.dumps({"chunks": len(chunks), "nodes": ingester.stats["nodes_added"]}))
        ingester.save()
    else:
        print("Omega-Cube Context Ingester")
        print(f"Usage: echo 'your text' | {sys.argv[0]}")
        print(f"       {sys.argv[0]} --interactive")
        print(f"       {sys.argv[0]} --query 'what topics'")
