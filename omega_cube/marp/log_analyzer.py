"""
log_analyzer.py — Daily MARP router log analysis.

Extracted from router_service.py during P1.6 consolidation (2026-08-25):
the SmolLM2 service was removed (superseded by QwenGPURouter in gpu_router.py),
but this log reader remains useful (marp_analyze_today.py consumes it).

Log source: ~/.hermes/logs/marp_router/marp_YYYYMMDD.jsonl (written by
gpu_router.log_classification).
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

LOG_DIR = Path.home() / ".hermes" / "logs" / "marp_router"


class LogAnalyzer:
    """Analyze MARP router logs for metrics."""

    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or LOG_DIR

    def analyze_today(self) -> dict:
        """Analyze today's logs."""
        today = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"marp_{today}.jsonl"
        if not log_file.exists():
            return {"error": f"No logs for {today}"}

        entries = []
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not entries:
            return {"error": "No valid entries"}

        latencies = [e["latency_us"] for e in entries]
        latencies.sort()
        domains = Counter()
        models = Counter()
        for e in entries:
            for d in e["domains"]:
                domains[d] += 1
            models[e["model_used"]] += 1

        return {
            "date": today,
            "total_queries": len(entries),
            "avg_latency_us": round(sum(latencies) / len(latencies), 1),
            "p50_latency_us": latencies[len(latencies)//2],
            "p95_latency_us": latencies[int(len(latencies)*0.95)],
            "p99_latency_us": latencies[int(len(latencies)*0.99)],
            "top_domains": domains.most_common(5),
            "model_usage": dict(models),
            "avg_token_savings": round(sum(e["token_savings"] for e in entries)/len(entries), 3),
        }
