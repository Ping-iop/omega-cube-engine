"""
AXION Telemetry — Protocol efficiency measurement and problem detection.

Logs every routing decision, engine query, and system event to structured
JSON logs. Provides efficiency reports, anomaly detection, and health checks.

Usage:
    from omega_cube.telemetry import Telemetry

    telemetry = Telemetry(log_dir="logs")
    telemetry.log_routing(query, domain, latency_ms, confidence, correct)
    telemetry.log_query(query, mode, results_count, latency_ms)
    telemetry.log_error(component, error_type, message)
    report = telemetry.efficiency_report()
    problems = telemetry.detect_problems()
"""

from __future__ import annotations

import json
import os
import time
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RoutingEvent:
    """A single routing decision."""
    timestamp: str
    query: str
    predicted_domain: str
    expected_domain: Optional[str]
    confidence: float
    latency_ms: float
    correct: Optional[bool]
    hierarchical: bool
    context_nodes: int
    boundary_filtered: int
    bias_detected: bool
    session_id: str = ""


@dataclass
class QueryEvent:
    """A single engine query."""
    timestamp: str
    query: str
    mode: str  # hierarchical, holographic, diffusion, combined
    results_count: int
    latency_ms: float
    top_score: float
    session_id: str = ""


@dataclass
class ErrorEvent:
    """A system error or anomaly."""
    timestamp: str
    component: str  # router, engine, scheduler, protocol
    error_type: str  # timeout, crash, low_accuracy, bias, memory
    message: str
    severity: str = "warning"  # info, warning, error, critical
    session_id: str = ""


@dataclass
class SessionStats:
    """Aggregated stats for a session."""
    session_id: str
    start_time: str
    end_time: str = ""
    total_queries: int = 0
    total_routings: int = 0
    correct_routings: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    errors: int = 0
    bias_detections: int = 0
    boundary_filtered_total: int = 0
    domains_used: dict = field(default_factory=dict)
    modes_used: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Telemetry Engine
# ═══════════════════════════════════════════════════════════════════

class Telemetry:
    """AXION Protocol telemetry — logs, measures, detects problems.

    Writes structured JSON logs to a configurable directory.
    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        session_id: Optional[str] = None,
        auto_flush: bool = True,
        flush_interval: int = 50,  # events
    ):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._auto_flush = auto_flush
        self._flush_interval = flush_interval
        self._lock = threading.Lock()

        # In-memory buffers
        self._routing_events: list[RoutingEvent] = []
        self._query_events: list[QueryEvent] = []
        self._error_events: list[ErrorEvent] = []
        self._event_count = 0

        # Session tracking
        self._session_start = datetime.now().isoformat()
        self._latencies: list[float] = []
        self._domain_counter: Counter = Counter()
        self._mode_counter: Counter = Counter()

        # Log files
        self._routing_log = self._log_dir / f"routing_{self._session_id}.jsonl"
        self._query_log = self._log_dir / f"queries_{self._session_id}.jsonl"
        self._error_log = self._log_dir / f"errors_{self._session_id}.jsonl"
        self._session_log = self._log_dir / f"session_{self._session_id}.json"

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    # ── Logging methods ──

    def log_routing(
        self,
        query: str,
        predicted_domain: str,
        latency_ms: float,
        confidence: float = 0.0,
        expected_domain: Optional[str] = None,
        correct: Optional[bool] = None,
        hierarchical: bool = False,
        context_nodes: int = 0,
        boundary_filtered: int = 0,
        bias_detected: bool = False,
    ) -> RoutingEvent:
        """Log a routing decision."""
        event = RoutingEvent(
            timestamp=datetime.now().isoformat(),
            query=query[:200],  # truncate for storage
            predicted_domain=predicted_domain,
            expected_domain=expected_domain,
            confidence=confidence,
            latency_ms=round(latency_ms, 3),
            correct=correct,
            hierarchical=hierarchical,
            context_nodes=context_nodes,
            boundary_filtered=boundary_filtered,
            bias_detected=bias_detected,
            session_id=self._session_id,
        )

        with self._lock:
            self._routing_events.append(event)
            self._latencies.append(latency_ms)
            self._domain_counter[predicted_domain] += 1
            self._event_count += 1

            if self._auto_flush and self._event_count % self._flush_interval == 0:
                self._flush_routing()

        return event

    def log_query(
        self,
        query: str,
        mode: str,
        results_count: int,
        latency_ms: float,
        top_score: float = 0.0,
    ) -> QueryEvent:
        """Log an engine query."""
        event = QueryEvent(
            timestamp=datetime.now().isoformat(),
            query=query[:200],
            mode=mode,
            results_count=results_count,
            latency_ms=round(latency_ms, 3),
            top_score=round(top_score, 4),
            session_id=self._session_id,
        )

        with self._lock:
            self._query_events.append(event)
            self._mode_counter[mode] += 1
            self._event_count += 1

            if self._auto_flush and self._event_count % self._flush_interval == 0:
                self._flush_queries()

        return event

    def log_error(
        self,
        component: str,
        error_type: str,
        message: str,
        severity: str = "warning",
    ) -> ErrorEvent:
        """Log a system error or anomaly."""
        event = ErrorEvent(
            timestamp=datetime.now().isoformat(),
            component=component,
            error_type=error_type,
            message=message[:500],
            severity=severity,
            session_id=self._session_id,
        )

        with self._lock:
            self._error_events.append(event)
            self._event_count += 1

        # Errors always flush immediately
        self._append_jsonl(self._error_log, asdict(event))
        return event

    # ── Analysis methods ──

    def efficiency_report(self) -> dict:
        """Generate a comprehensive efficiency report."""
        with self._lock:
            total_routings = len(self._routing_events)
            total_queries = len(self._query_events)
            total_errors = len(self._error_events)

            # Routing accuracy
            labeled = [e for e in self._routing_events if e.correct is not None]
            correct = sum(1 for e in labeled if e.correct)
            accuracy = correct / len(labeled) * 100 if labeled else 0.0

            # Latency stats
            latencies = sorted(self._latencies)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            p50 = latencies[len(latencies) // 2] if latencies else 0.0
            p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
            p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0
            max_latency = max(latencies) if latencies else 0.0

            # Confidence
            confidences = [e.confidence for e in self._routing_events]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Hierarchical usage
            hierarchical_pct = (
                sum(1 for e in self._routing_events if e.hierarchical)
                / max(total_routings, 1) * 100
            )

            # Context quality
            avg_context = (
                sum(e.context_nodes for e in self._routing_events)
                / max(total_routings, 1)
            )
            total_boundary = sum(e.boundary_filtered for e in self._routing_events)

            # Bias
            bias_count = sum(1 for e in self._routing_events if e.bias_detected)

            # Error breakdown
            error_by_type = Counter(e.error_type for e in self._error_events)
            error_by_severity = Counter(e.severity for e in self._error_events)

            # Domain distribution
            domain_dist = dict(self._domain_counter.most_common())

            # Mode distribution
            mode_dist = dict(self._mode_counter.most_common())

            # Query latency by mode
            query_latency_by_mode: dict[str, list[float]] = defaultdict(list)
            for qe in self._query_events:
                query_latency_by_mode[qe.mode].append(qe.latency_ms)

            mode_stats = {}
            for mode, lats in query_latency_by_mode.items():
                mode_stats[mode] = {
                    "count": len(lats),
                    "avg_ms": round(sum(lats) / len(lats), 3),
                    "p95_ms": round(sorted(lats)[int(len(lats) * 0.95)], 3) if lats else 0,
                }

        report = {
            "session_id": self._session_id,
            "generated_at": datetime.now().isoformat(),
            "duration_seconds": round(
                (datetime.now() - datetime.fromisoformat(self._session_start)).total_seconds(), 1
            ),
            "totals": {
                "routings": total_routings,
                "queries": total_queries,
                "errors": total_errors,
            },
            "routing_efficiency": {
                "accuracy_pct": round(accuracy, 1),
                "correct": correct,
                "labeled": len(labeled),
                "avg_confidence": round(avg_confidence, 3),
                "hierarchical_pct": round(hierarchical_pct, 1),
                "avg_context_nodes": round(avg_context, 1),
                "boundary_filtered_total": total_boundary,
                "bias_detections": bias_count,
            },
            "latency": {
                "avg_ms": round(avg_latency, 3),
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p99_ms": round(p99, 3),
                "max_ms": round(max_latency, 3),
            },
            "query_modes": mode_stats,
            "domain_distribution": domain_dist,
            "mode_distribution": mode_dist,
            "errors": {
                "by_type": dict(error_by_type),
                "by_severity": dict(error_by_severity),
            },
            "health_score": self._compute_health_score(
                accuracy, avg_latency, p95, total_errors, total_routings, bias_count
            ),
        }

        # Save report
        report_path = self._log_dir / f"report_{self._session_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def detect_problems(self) -> list[dict]:
        """Detect problems and anomalies in the protocol."""
        problems = []

        with self._lock:
            labeled = [e for e in self._routing_events if e.correct is not None]
            if len(labeled) >= 10:
                correct = sum(1 for e in labeled if e.correct)
                accuracy = correct / len(labeled) * 100

                if accuracy < 50:
                    problems.append({
                        "type": "low_accuracy",
                        "severity": "critical",
                        "message": f"Routing accuracy {accuracy:.1f}% < 50% threshold",
                        "suggestion": "Check keyword rules and graph coverage",
                    })
                elif accuracy < 70:
                    problems.append({
                        "type": "low_accuracy",
                        "severity": "warning",
                        "message": f"Routing accuracy {accuracy:.1f}% < 70% target",
                        "suggestion": "Add more domain keywords or graph nodes",
                    })

            # Latency problems
            if self._latencies:
                avg = sum(self._latencies) / len(self._latencies)
                p95 = sorted(self._latencies)[int(len(self._latencies) * 0.95)]

                if avg > 50:
                    problems.append({
                        "type": "high_latency",
                        "severity": "critical",
                        "message": f"Avg latency {avg:.1f}ms > 50ms target",
                        "suggestion": "Check graph size, reduce query depth",
                    })
                elif p95 > 100:
                    problems.append({
                        "type": "latency_spike",
                        "severity": "warning",
                        "message": f"P95 latency {p95:.1f}ms > 100ms",
                        "suggestion": "Investigate slow queries, check for GC pauses",
                    })

            # Error rate
            total_events = len(self._routing_events) + len(self._query_events)
            if total_events > 0:
                error_rate = len(self._error_events) / total_events * 100
                if error_rate > 10:
                    problems.append({
                        "type": "high_error_rate",
                        "severity": "critical",
                        "message": f"Error rate {error_rate:.1f}% > 10%",
                        "suggestion": "Check error log for root causes",
                    })
                elif error_rate > 5:
                    problems.append({
                        "type": "elevated_errors",
                        "severity": "warning",
                        "message": f"Error rate {error_rate:.1f}% > 5%",
                        "suggestion": "Monitor error trends",
                    })

            # Bias problems
            bias_count = sum(1 for e in self._routing_events if e.bias_detected)
            if bias_count > 0 and len(self._routing_events) > 0:
                bias_rate = bias_count / len(self._routing_events) * 100
                if bias_rate > 20:
                    problems.append({
                        "type": "high_bias",
                        "severity": "warning",
                        "message": f"Bias detection rate {bias_rate:.1f}% > 20%",
                        "suggestion": "Review domain classification rules",
                    })

            # Domain imbalance
            if self._domain_counter:
                total = sum(self._domain_counter.values())
                for domain, count in self._domain_counter.items():
                    pct = count / total * 100
                    if pct > 60 and len(self._domain_counter) > 2:
                        problems.append({
                            "type": "domain_imbalance",
                            "severity": "info",
                            "message": f"Domain '{domain}' handles {pct:.0f}% of queries",
                            "suggestion": "May indicate routing bias or genuine usage pattern",
                        })

            # Context quality
            if self._routing_events:
                avg_ctx = sum(e.context_nodes for e in self._routing_events) / len(self._routing_events)
                if avg_ctx < 1.0:
                    problems.append({
                        "type": "low_context",
                        "severity": "warning",
                        "message": f"Avg context nodes {avg_ctx:.1f} < 1.0",
                        "suggestion": "Graph may be too sparse, add more nodes",
                    })

        return problems

    def _compute_health_score(
        self, accuracy, avg_latency, p95, errors, total, bias
    ) -> dict:
        """Compute a 0-100 health score with breakdown."""
        # Accuracy component (40%)
        acc_score = min(accuracy / 80 * 40, 40)  # 80% accuracy = full marks

        # Latency component (30%)
        if avg_latency <= 5:
            lat_score = 30
        elif avg_latency <= 20:
            lat_score = 20
        elif avg_latency <= 50:
            lat_score = 10
        else:
            lat_score = 0

        # Error component (20%)
        error_rate = errors / max(total, 1) * 100
        if error_rate == 0:
            err_score = 20
        elif error_rate <= 2:
            err_score = 15
        elif error_rate <= 5:
            err_score = 10
        elif error_rate <= 10:
            err_score = 5
        else:
            err_score = 0

        # Bias component (10%)
        bias_rate = bias / max(total, 1) * 100
        if bias_rate == 0:
            bias_score = 10
        elif bias_rate <= 5:
            bias_score = 7
        elif bias_rate <= 15:
            bias_score = 4
        else:
            bias_score = 0

        total_score = round(acc_score + lat_score + err_score + bias_score, 1)

        return {
            "total": total_score,
            "grade": (
                "A" if total_score >= 90 else
                "B" if total_score >= 75 else
                "C" if total_score >= 60 else
                "D" if total_score >= 40 else
                "F"
            ),
            "breakdown": {
                "accuracy": f"{acc_score:.0f}/40",
                "latency": f"{lat_score:.0f}/30",
                "errors": f"{err_score:.0f}/20",
                "bias": f"{bias_score:.0f}/10",
            },
        }

    # ── Persistence ──

    def flush(self):
        """Flush all buffers to disk."""
        with self._lock:
            self._flush_routing()
            self._flush_queries()
            self._save_session()

    def _flush_routing(self):
        for event in self._routing_events:
            self._append_jsonl(self._routing_log, asdict(event))
        self._routing_events.clear()

    def _flush_queries(self):
        for event in self._query_events:
            self._append_jsonl(self._query_log, asdict(event))
        self._query_events.clear()

    def _save_session(self):
        session = {
            "session_id": self._session_id,
            "start_time": self._session_start,
            "end_time": datetime.now().isoformat(),
            "total_routings": len(self._routing_events),
            "total_queries": len(self._query_events),
            "total_errors": len(self._error_events),
            "domains": dict(self._domain_counter),
            "modes": dict(self._mode_counter),
        }
        with open(self._session_log, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _append_jsonl(path: Path, data: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # ── Historical analysis ──

    @classmethod
    def load_reports(cls, log_dir: str = "logs") -> list[dict]:
        """Load all historical reports from a log directory."""
        log_path = Path(log_dir)
        reports = []
        for f in sorted(log_path.glob("report_*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    reports.append(json.load(fh))
            except (json.JSONDecodeError, OSError):
                continue
        return reports

    @classmethod
    def trend_analysis(cls, log_dir: str = "logs") -> dict:
        """Analyze trends across multiple sessions."""
        reports = cls.load_reports(log_dir)
        if not reports:
            return {"sessions": 0, "message": "No historical data"}

        accuracies = [r["routing_efficiency"]["accuracy_pct"] for r in reports if r.get("routing_efficiency")]
        latencies = [r["latency"]["avg_ms"] for r in reports if r.get("latency")]
        health_scores = [r["health_score"]["total"] for r in reports if r.get("health_score")]
        error_counts = [r["totals"]["errors"] for r in reports if r.get("totals")]

        return {
            "sessions": len(reports),
            "accuracy_trend": {
                "first": accuracies[0] if accuracies else None,
                "last": accuracies[-1] if accuracies else None,
                "avg": round(sum(accuracies) / len(accuracies), 1) if accuracies else None,
                "improving": accuracies[-1] > accuracies[0] if len(accuracies) > 1 else None,
            },
            "latency_trend": {
                "first": latencies[0] if latencies else None,
                "last": latencies[-1] if latencies else None,
                "avg": round(sum(latencies) / len(latencies), 3) if latencies else None,
                "improving": latencies[-1] < latencies[0] if len(latencies) > 1 else None,
            },
            "health_trend": {
                "first": health_scores[0] if health_scores else None,
                "last": health_scores[-1] if health_scores else None,
                "avg": round(sum(health_scores) / len(health_scores), 1) if health_scores else None,
            },
            "total_errors": sum(error_counts) if error_counts else 0,
        }
