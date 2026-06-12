"""
Qwen GGUF Classifier — Drop-in replacement for SmolLM2 in MARP Router.

Uses llama-cpp-python with Qwen3.5-0.8B-Q6_K.gguf (639MB, quantized).
Runs on CPU or GPU via llama.cpp's CUDA backend.

Latency target: <5ms on GPU, <50ms on CPU
Accuracy target: 75-85% domain classification
"""

import time
import logging
from pathlib import Path
from typing import Optional

console = logging.getLogger("marp_console")


class QwenGGUFClassifier:
    """Domain classifier using Qwen 0.8B GGUF via llama-cpp-python.

    This is the recommended router model for MARP:
    - 639MB on disk (Q6_K quantization)
    - <5ms inference on GPU (RTX 3090)
    - <50ms inference on CPU
    - 10-domain classification with structured prompt
    """

    # Structured prompt for domain classification
    SYSTEM_PROMPT = """You are a query classifier. Classify the user's question into ONE or TWO domains.
Return ONLY the domain names separated by comma, nothing else.

Available domains: math, code, science, engineering, language, law, medical, business, philosophy, gaming, general

Examples:
"What is the derivative of x^2?" → math
"Write a Python function to sort a list" → code
"Explain quantum entanglement" → science
"How to deploy Docker to Kubernetes?" → code, engineering
"Draft a non-disclosure agreement" → law
"What are symptoms of diabetes?" → medical
"Calculate NPV of an investment" → business, math
"Is free will compatible with determinism?" → philosophy
"Design a roguelike death mechanic" → gaming
"Translate this poem to Spanish" → language

Rules:
- Use EXACT domain names from the list
- If a query spans multiple domains, use comma (max 2)
- If truly unsure, use "general"
- Never explain, never add text beyond domain names"""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = "J:/modelos_ia/Qwen3.5-0.8B-Q6_K.gguf"
        self.model_path = Path(model_path)
        self._llm: Optional[object] = None
        self._available = self.model_path.exists()
        self._load_attempted = False
        self._load_time_ms = 0.0
        self._total_calls = 0
        self._total_time_ms = 0.0

    @property
    def available(self) -> bool:
        return self._available

    def load(self, n_gpu_layers: int = -1, n_ctx: int = 512) -> bool:
        """Load the GGUF model via llama-cpp-python.

        Args:
            n_gpu_layers: Layers to offload to GPU (-1 = all, 0 = CPU only)
            n_ctx: Context window size
        """
        if self._load_attempted:
            return self._llm is not None
        self._load_attempted = True

        if not self._available:
            console.warning(f"Qwen GGUF not found at {self.model_path}")
            return False

        try:
            from llama_cpp import Llama
            t0 = time.perf_counter()
            console.info(f"Loading Qwen GGUF from {self.model_path}...")
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            self._load_time_ms = (time.perf_counter() - t0) * 1000
            console.info(f"Qwen GGUF loaded in {self._load_time_ms:.0f}ms "
                        f"(GPU layers: {n_gpu_layers})")
            return True
        except Exception as e:
            console.error(f"Failed to load Qwen GGUF: {e}")
            # Try CPU-only fallback
            try:
                from llama_cpp import Llama
                console.info("Retrying with CPU-only...")
                self._llm = Llama(
                    model_path=str(self.model_path),
                    n_ctx=512,
                    n_gpu_layers=0,
                    verbose=False,
                )
                self._load_time_ms = (time.perf_counter() - t0) * 1000
                console.info(f"Qwen GGUF loaded (CPU-only) in {self._load_time_ms:.0f}ms")
                return True
            except Exception as e2:
                console.error(f"CPU fallback also failed: {e2}")
                self._available = False
                return False

    def classify(self, query: str) -> tuple[list[str], float]:
        """Classify a query into domains.

        Returns:
            (domains, confidence) where domains is a list of 1-2 domain names
            and confidence is 0.0-1.0
        """
        if not self._llm:
            return ["general"], 0.15

        t0 = time.perf_counter()
        try:
            prompt = f"Query: {query[:300]}\nDomains:"

            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8,
                temperature=0.0,
                stop=["\n", ".", ":"],
            )

            elapsed = (time.perf_counter() - t0) * 1000
            self._total_calls += 1
            self._total_time_ms += elapsed

            # Parse response
            raw = response["choices"][0]["message"]["content"].strip().lower()
            # Clean up
            raw = raw.replace("domains:", "").replace("domain:", "").strip()

            # Split and validate
            domains = [d.strip() for d in raw.split(",")[:2]]
            valid = {'math','code','science','engineering','language',
                    'law','medical','business','philosophy','gaming','general'}
            domains = [d for d in domains if d in valid]

            if not domains:
                domains = ["general"]

            # Confidence based on how specific the classification was
            confidence = 0.85 if len(domains) == 1 else 0.70
            if domains == ["general"]:
                confidence = 0.25

            return domains, confidence

        except Exception as e:
            console.error(f"Qwen classify error: {e}")
            return ["general"], 0.10

    @property
    def stats(self) -> dict:
        return {
            "model": "Qwen3.5-0.8B-Q6_K",
            "size_mb": self.model_path.stat().st_size / 1e6 if self._available else 0,
            "loaded": self._llm is not None,
            "load_time_ms": round(self._load_time_ms, 0),
            "total_calls": self._total_calls,
            "avg_latency_ms": round(self._total_time_ms / max(self._total_calls, 1), 1),
        }


# ═══════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    console.setLevel(logging.INFO)

    clf = QwenGGUFClassifier()
    if not clf.available:
        print("Model not found at J:/modelos_ia/Qwen3.5-0.8B-Q6_K.gguf")
        exit(1)

    clf.load(n_gpu_layers=0)  # CPU first, safe
    print(f"Loaded: {clf.stats}")

    tests = [
        "What is the derivative of x squared?",
        "Write a Python function to sort a list",
        "Explain quantum entanglement simply",
        "How do I deploy Docker to Kubernetes?",
        "Draft a non-disclosure agreement for a startup",
        "What are the symptoms of type 2 diabetes?",
        "Calculate the net present value of an investment",
        "Is free will compatible with determinism?",
        "Design a roguelike death mechanic",
        "Translate this poem to Spanish",
    ]

    print("\nClassification tests:")
    print("-" * 60)
    for q in tests:
        domains, conf = clf.classify(q)
        print(f"  [{conf:.2f}] {domains} ← '{q[:60]}...'")

    print(f"\nStats: {clf.stats}")
