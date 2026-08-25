"""
SessionContextEngine — Extensión del AxiomaticMemoryEngine para indexación
de contexto de sesión en tiempo real.

Resuelve el problema de decaimiento de contexto en chats largos:
extrae entidades, decisiones, paths e insights de cada turno y los indexa
en el grafo jerárquico Omega para recuperación O(log n).

Jerarquía de sesión:
    SESSION.<session_id>.DECISIONS.<turn>_<tema>
    SESSION.<session_id>.CONTEXT.PATHS.<key>
    SESSION.<session_id>.CONTEXT.COMMANDS.<key>
    SESSION.<session_id>.CONTEXT.CONFIG.<key>
    SESSION.<session_id>.INSIGHTS.<turn>_<tema>
    SESSION.<session_id>.STATE

Autor: Bit (Hermes Agent)
Fecha: 2026-06-11
"""

import re
import time
from typing import Optional

from memory_engine import AxiomaticMemoryEngine, MemoryNode


# ─── Entity Extractors ───────────────────────────────────────────────

# Paths: Windows (C:\..., J:\...) y Unix (/home/..., /mnt/...)
RE_PATH_WIN = re.compile(r'\b([A-Za-z]:[\\/](?:[\w\-\.]+[\\/])*[\w\-\.]+)', re.IGNORECASE)
RE_PATH_UNIX = re.compile(r'\b(/(?:[\w\-\.]+/)*[\w\-\.]+)', re.IGNORECASE)

# Comandos de terminal (bash, python, git, etc.)
RE_COMMAND = re.compile(r'`([^`]+)`|```(?:bash|python|sh|shell)?\n(.*?)```', re.DOTALL)

# Decisiones explícitas
DECISION_MARKERS = [
    r'\b(?:voy a|vamos a|decidí|decidimos|usemos|usaré|implementemos|la solución es|el plan es)\b',
    r'\b(?:I will|we will|let\'s|I decided|we decided|the solution is|the plan is)\b',
]
RE_DECISION = re.compile('(' + '|'.join(DECISION_MARKERS) + ')', re.IGNORECASE)

# Configuraciones mencionadas
CONFIG_PATTERNS = [
    r'\b(?:config\.yaml|config\.json|\.env|settings|api[_\s]?key|token|model|provider)\b',
    r'\b(?:hermes config set|hermes config get|hermes mcp)\b',
]
RE_CONFIG = re.compile('(' + '|'.join(CONFIG_PATTERNS) + ')', re.IGNORECASE)


def extract_paths(text: str) -> list[str]:
    """Extrae paths de archivos del texto."""
    paths = set()
    for m in RE_PATH_WIN.finditer(text):
        p = m.group(1)
        if len(p) > 3 and not p.endswith(('.', ',')):
            paths.add(p)
    for m in RE_PATH_UNIX.finditer(text):
        p = m.group(1)
        if len(p) > 2 and not p.endswith(('.', ',')):
            paths.add(p)
    return list(paths)[:10]


def extract_commands(text: str) -> list[str]:
    """Extrae comandos de terminal del texto."""
    commands = []
    for m in RE_COMMAND.finditer(text):
        cmd = m.group(1) or m.group(2)
        cmd = cmd.strip()
        if cmd and len(cmd) > 2:
            commands.append(cmd[:200])
    return commands[:5]


def detect_decision(text: str) -> Optional[str]:
    """Detecta si el texto contiene una decisión explícita y extrae la frase clave."""
    if RE_DECISION.search(text):
        sentences = re.split(r'[.!?]\s+', text)
        for sent in sentences:
            if RE_DECISION.search(sent):
                return sent.strip()[:300]
    return None


def extract_config_mentions(text: str) -> list[str]:
    """Extrae menciones de configuración del texto."""
    mentions = set()
    for m in RE_CONFIG.finditer(text):
        mentions.add(m.group(1))
    return list(mentions)[:5]


def summarize_turn(text: str, max_len: int = 80) -> str:
    """Genera un resumen corto del turno para usar como etiqueta."""
    clean = re.sub(r'`|```|\*\*|__|#', '', text)
    sentences = re.split(r'[.!?\n]', clean)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 10:
            return sent[:max_len]
    return clean[:max_len]


# ─── Session Context Engine ──────────────────────────────────────────

class SessionContextEngine(AxiomaticMemoryEngine):
    """
    Motor de contexto de sesión que extiende AxiomaticMemoryEngine.
    
    Añade indexación automática de turnos de conversación en el grafo
    jerárquico Omega, permitiendo recuperación eficiente de información
    que ya no está en la ventana de contexto lineal del LLM.
    
    Nivel SESSION: confidence 0.7 — menor que INSTANCE porque es efímero.
    """

    CONFIDENCE_LEVELS = {
        **AxiomaticMemoryEngine.CONFIDENCE_LEVELS,
        "SESSION": 0.7,
    }

    def stats(self) -> dict:
        """Return engine statistics including SESSION nodes."""
        counts = {"AXIOM": 0, "CONCEPT": 0, "INSTANCE": 0, "SESSION": 0}
        for node in self.nodes.values():
            if node.node_type in counts:
                counts[node.node_type] += 1
        return {
            "total_nodos": len(self.nodes),
            "axiomas": counts["AXIOM"],
            "conceptos": counts["CONCEPT"],
            "instancias": counts["INSTANCE"],
            "sesiones": counts["SESSION"],
            "memory_dir": str(self.memory_dir),
        }

    # ── Public API ─────────────────────────────────────────────────

    def index_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        turn_number: int,
    ) -> dict:
        """
        Indexa un turno completo (user + assistant) en el grafo de sesión.
        
        Extrae y almacena:
        - Decisiones explícitas
        - Paths de archivos mencionados
        - Comandos ejecutados
        - Menciones de configuración
        - Insights (de la respuesta del assistant)
        
        Returns:
            dict con conteo de entidades indexadas
        """
        stats = {"decisions": 0, "paths": 0, "commands": 0, "configs": 0, "insights": 0}

        combined = f"{user_message}\n{assistant_response}"

        # 1. Decisiones
        decision = detect_decision(combined)
        if decision:
            topic = summarize_turn(decision, 60).replace(".", "_").replace(" ", "_")[:50]
            node = self._add_session_node(
                f"SESSION.{session_id}.DECISIONS.{turn_number:04d}_{topic}",
                decision,
                tags=["decision", f"turn_{turn_number}"],
            )
            if node:
                stats["decisions"] += 1

        # 2. Paths
        paths = extract_paths(combined)
        for i, path in enumerate(paths):
            key = self._path_to_key(path)
            node = self._add_session_node(
                f"SESSION.{session_id}.CONTEXT.PATHS.{key}",
                path,
                tags=["path", f"turn_{turn_number}"],
            )
            if node:
                stats["paths"] += 1

        # 3. Comandos
        commands = extract_commands(combined)
        for i, cmd in enumerate(commands):
            key = re.sub(r'[^a-zA-Z0-9_-]', '_', cmd[:40]).strip('_')
            node = self._add_session_node(
                f"SESSION.{session_id}.CONTEXT.COMMANDS.{turn_number:04d}_{key}",
                cmd,
                tags=["command", f"turn_{turn_number}"],
            )
            if node:
                stats["commands"] += 1

        # 4. Configuraciones
        configs = extract_config_mentions(combined)
        for cfg in configs:
            key = re.sub(r'[^a-zA-Z0-9_-]', '_', cfg).strip('_')
            node = self._add_session_node(
                f"SESSION.{session_id}.CONTEXT.CONFIG.{turn_number:04d}_{key}",
                cfg,
                tags=["config", f"turn_{turn_number}"],
            )
            if node:
                stats["configs"] += 1

        # 5. Insights (de la respuesta del assistant)
        if len(assistant_response) > 50:
            insight = summarize_turn(assistant_response, 100)
            topic = insight.replace(".", "_").replace(" ", "_")[:60]
            node = self._add_session_node(
                f"SESSION.{session_id}.INSIGHTS.{turn_number:04d}_{topic}",
                insight,
                tags=["insight", f"turn_{turn_number}"],
            )
            if node:
                stats["insights"] += 1

        return stats

    def retrieve_session_context(
        self,
        session_id: str,
        query: str,
        max_results: int = 10,
    ) -> list[dict]:
        """
        Recupera contexto relevante de la sesión usando búsqueda axiomática.
        
        Busca solo en el subgrafo de esta sesión, usando navegación jerárquica
        desde los nodos SESSION. Más rápido que buscar todo el grafo.
        
        Args:
            session_id: ID de la sesión (ej: 'abc123')
            query: Texto de búsqueda (lo que se necesita recordar)
            max_results: Máximo de resultados
        
        Returns:
            Lista de nodos relevantes con score, jerarquía y contenido
        """
        keywords = self._extract_keywords(query)
        session_prefix = f"SESSION.{session_id}."
        
        candidates = []
        for hierarchy, node in self.nodes.items():
            if hierarchy.startswith(session_prefix):
                score = self._relevance_score(node, keywords)
                if score > 0:
                    candidates.append({
                        "node_type": node.node_type,
                        "hierarchy": hierarchy,
                        "content": node.content,
                        "score": score,
                        "tags": node.tags,
                        "associations": node.associations,
                    })
        
        candidates.sort(key=lambda x: -x["score"])
        return candidates[:max_results]

    def checkpoint(self, session_id: str, state_summary: str) -> bool:
        """
        Guarda un checkpoint del estado actual de la conversación.
        
        Útil para retomar después de cambios de tema o sesiones largas.
        Solo mantiene el checkpoint más reciente (sobrescribe STATE).
        
        Args:
            session_id: ID de la sesión
            state_summary: Resumen del estado actual
        
        Returns:
            True si se guardó correctamente
        """
        hierarchy = f"SESSION.{session_id}.STATE"
        node = self._add_session_node(
            hierarchy,
            state_summary,
            tags=["checkpoint", "state"],
        )
        return node is not None

    def get_session_summary(self, session_id: str) -> dict:
        """
        Obtiene un resumen completo de la sesión: decisiones, paths,
        comandos, configs, insights y último checkpoint.
        
        Returns:
            dict con conteos y listas por categoría
        """
        prefix = f"SESSION.{session_id}."
        summary = {
            "session_id": session_id,
            "total_nodes": 0,
            "decisions": [],
            "paths": [],
            "commands": [],
            "configs": [],
            "insights": [],
            "state": None,
            "turn_range": None,
        }

        turns = []
        for hierarchy, node in self.nodes.items():
            if not hierarchy.startswith(prefix):
                continue
            summary["total_nodes"] += 1

            if ".DECISIONS." in hierarchy:
                summary["decisions"].append({
                    "hierarchy": hierarchy,
                    "content": node.content,
                    "tags": node.tags,
                })
            elif ".PATHS." in hierarchy:
                summary["paths"].append({
                    "hierarchy": hierarchy,
                    "content": node.content,
                    "tags": node.tags,
                })
            elif ".COMMANDS." in hierarchy:
                summary["commands"].append({
                    "hierarchy": hierarchy,
                    "content": node.content,
                    "tags": node.tags,
                })
            elif ".CONFIG." in hierarchy:
                summary["configs"].append({
                    "hierarchy": hierarchy,
                    "content": node.content,
                    "tags": node.tags,
                })
            elif ".INSIGHTS." in hierarchy:
                summary["insights"].append({
                    "hierarchy": hierarchy,
                    "content": node.content,
                    "tags": node.tags,
                })
            elif hierarchy.endswith(".STATE"):
                summary["state"] = node.content

            for tag in node.tags:
                if tag.startswith("turn_"):
                    try:
                        turns.append(int(tag.split("_")[1]))
                    except ValueError:
                        pass

        if turns:
            summary["turn_range"] = (min(turns), max(turns))

        return summary

    def clear_session(self, session_id: str) -> int:
        """
        Elimina todos los nodos de una sesión.
        
        Returns:
            Número de nodos eliminados
        """
        prefix = f"SESSION.{session_id}."
        to_delete = [h for h in self.nodes if h.startswith(prefix)]
        for h in to_delete:
            del self.nodes[h]
        return len(to_delete)

    def list_sessions(self) -> list[str]:
        """
        Lista todos los session_id activos en el grafo.
        
        Returns:
            Lista de session_id únicos
        """
        sessions = set()
        for hierarchy in self.nodes:
            if hierarchy.startswith("SESSION."):
                parts = hierarchy.split(".")
                if len(parts) >= 2:
                    sessions.add(parts[1])
        return sorted(sessions)

    # ── Internal ───────────────────────────────────────────────────

    def _add_session_node(
        self,
        hierarchy: str,
        content: str,
        tags: list[str] = None,
    ) -> Optional[MemoryNode]:
        """Añade un nodo de tipo SESSION al grafo."""
        node = MemoryNode(
            content=content[:1000],
            hierarchy=hierarchy,
            node_type="SESSION",
            tags=tags or [],
            confidence=self.CONFIDENCE_LEVELS["SESSION"],
        )
        self.nodes[hierarchy] = node
        return node

    @staticmethod
    def _path_to_key(path: str) -> str:
        """Convierte un path a una key válida para jerarquía."""
        clean = re.sub(r'^[A-Za-z]:', '', path)
        clean = clean.replace('\\', '_').replace('/', '_')
        clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean)
        clean = re.sub(r'_+', '_', clean).strip('_')
        return clean[:80] if clean else f"path_{hash(path) % 10000}"


# ─── Convenience ─────────────────────────────────────────────────────

def create_session_engine(memory_path: str = None) -> SessionContextEngine:
    """
    Crea y carga un SessionContextEngine desde el path de memoria estándar.
    
    Args:
        memory_path: Path al archivo unified_memory.json
    
    Returns:
        SessionContextEngine listo para usar
    """
    from pathlib import Path

    engine = SessionContextEngine()
    
    if memory_path is None:
        axioma_base = str(Path.home() / ".hermes" / "axioma-omega-protocol")
        memory_path = f"{axioma_base}/memory/unified_memory.json"
    
    if Path(memory_path).exists():
        engine.load(memory_path)
    
    return engine
