#!/usr/bin/env python3
"""
MARP Intelligent Pipeline v3.0 — Entry point.
Router + Worker con thinking dinámico por dominio.

FIX v3.0: Router usa /completion + GBNF grammar (elimina  tags).
- Keyword pre-filter para edge cases en español (0.1ms)
- GBNF grammar fuerza output válido (nunca retorna thinking tokens)
- Dominio "memory" añadido para preferencias/recuerdos
- Prompt bilingüe EN/ES con 17 few-shot examples

FIX v2.1: Detección dinámica de familia de modelo en :8084.
- Gemma 4 / GLM → /completion + prompt manual (evita loop infinito con Jinja)
- Qwen / otros  → /v1/chat/completions + chat_template_kwargs (comportamiento previo)
"""
import requests, time, re, json, sys
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────
RURL    = "http://127.0.0.1:8082/completion"   # v3: /completion + grammar
WBASE   = "http://127.0.0.1:8084"

REASONING_DOMAINS = {"math", "science", "philosophy", "analysis", "engineering", "code"}
SPEED_DOMAINS     = {"language", "business", "general", "gaming", "law", "medical", "memory"}

# GBNF grammar — forces router to output ONLY a valid domain name
GBNF_GRAMMAR = 'root ::= ("math" | "code" | "science" | "engineering" | "language" | "law" | "medical" | "business" | "philosophy" | "gaming" | "general" | "memory")'

# Keyword pre-filter for edge cases (Spanish gaming, economics, memory)
KEYWORD_RULES = [
    (re.compile(r'\b(juego|game|rpg|roguelike|minecraft|fortnite|steam|gaming|gamer|jugador|videojuego)\b', re.I), 'gaming'),
    (re.compile(r'\b(oferta|demanda|mercado|economia|economía|precio|inflacion|inflación|pib|gdp)\b', re.I), 'business'),
    (re.compile(r'\b(recuerda|remember|preferencia|preference|configuracion|configuración|perfil|profile|ajuste|setting)\b', re.I), 'memory'),
    (re.compile(r'\b(video|imagen|image|audio|foto|photo|editar|edit|generar|generate|dibuj|draw|pintar|paint)\b', re.I), 'general'),
]

# Tokens de control de Gemma 4 que generan loops si no se filtran
_GEMMA_CTRL_TOKENS = re.compile(
    r'<\|channel\|>.*?<channel\|>|'    # bloques de canal completos
    r'<\|channel\|>.*?$|'              # bloque de canal sin cerrar
    r'<channel\|>|'                    # cierre de canal suelto
    r'<\|im_start\|>|<\|im_end\|>|'   # tokens de Qwen que Gemma echo-ea a veces
    r'<start_of_turn>|<end_of_turn>|'  # tokens de plantilla propios de Gemma
    r'<bos>|<eos>',                    # tokens especiales de inicio/fin
    re.DOTALL
)

# Queries cortas/simples → nunca thinking aunque el dominio lo pida
SIMPLE_PATTERNS = re.compile(
    r'^(what is\s+\d+\s*[+\-*/]\s*\d+|'
    r'hi|hello|hey|thanks|thank you|good morning|good night|'
    r'who are you|what can you do|'
    r'translate\s+\w+\s+to\s+\w+|'
    r'define\s+\w+|'
    r'what\'?s?\s+the\s+(capital|population|weather|time|date)|'
    r'[?]\s*$|'
    r'^\w+\s+\w+\s*$)',
    re.IGNORECASE
)

ROUTER_PROMPT_HEAD = """Classify the query into EXACTLY one domain.
Domains: math, code, science, engineering, language, law, medical, business, philosophy, gaming, general, memory

Rules:
- video/image/audio editing, greetings, chat -> general
- programming, docker, servers, API, codigo -> code
- physics, biology, chemistry, CRISPR, gravity, ADN -> science
- legal, contracts, NDA, court, contrato -> law
- health, disease, medicine, symptoms, sintomas -> medical
- finance, NPV, ROI, startup, investment, oferta/demanda -> business
- remember, preference, settings, profile, recuerda -> memory
- translate, languages, translation, traducir -> language
- math, derivatives, equations, calculus, derivada -> math
- bridges, structures, mechanics, materials, puente -> engineering
- ethics, philosophy, free will, morality, etica -> philosophy
- games, RPG, roguelike, juego, diseno de juego -> gaming

Examples:
query: derivative of x squared
domain: math
query: python async function
domain: code
query: quantum entanglement
domain: science
query: bridge structural design
domain: engineering
query: translate hello to spanish
domain: language
query: NDA contract terms
domain: law
query: diabetes symptoms treatment
domain: medical
query: NPV calculation startup
domain: business
query: free will determinism
domain: philosophy
query: roguelike game design
domain: gaming
query: diseno de juego roguelike
domain: gaming
query: edit video transitions
domain: general
query: remember I prefer dark theme
domain: memory
query: search papers CRISPR
domain: science
query: hola como estas
domain: general
query: ley de oferta y demanda
domain: business

query: {query}
domain:"""


# ─── Worker Model Detector ────────────────────────────────────────
class WorkerModelDetector:
    """
    Detecta qué familia de modelo está corriendo en el puerto worker (:8084).
    Se consulta al arranque y se cachea. Puede refrescarse entre queries.

    Familias detectadas:
      - "gemma"  → /completion + formato <bos><start_of_turn>user…
      - "glm"    → /completion + formato <|system|>…<|user|>…<|assistant|>
      - "qwen"   → /v1/chat/completions + chat_template_kwargs
      - "unknown"→ /v1/chat/completions sin enable_thinking (safe fallback)
    """

    _FAMILY_PATTERNS = {
        "gemma": re.compile(r"gemma", re.IGNORECASE),
        "glm":   re.compile(r"glm|chatglm", re.IGNORECASE),
        "qwen":  re.compile(r"qwen", re.IGNORECASE),
    }

    def __init__(self, base_url: str):
        self._base_url = base_url
        self._family   = None   # cached result
        self._model_id = None

    def detect(self, force_refresh: bool = False) -> str:
        """Retorna la familia detectada. Cachea el resultado."""
        if self._family and not force_refresh:
            return self._family

        # Primero intentamos /v1/models (llama.cpp lo expone)
        try:
            r = requests.get(f"{self._base_url}/v1/models", timeout=3)
            if r.status_code == 200:
                data = r.json()
                model_id = data.get("data", [{}])[0].get("id", "")
                self._model_id = model_id
                for family, pattern in self._FAMILY_PATTERNS.items():
                    if pattern.search(model_id):
                        self._family = family
                        return family
        except Exception:
            pass

        # Fallback: /props  (llama.cpp b8000+)
        try:
            r = requests.get(f"{self._base_url}/props", timeout=3)
            if r.status_code == 200:
                props = r.json()
                model_id = (
                    props.get("model_path", "") or
                    props.get("model", "") or ""
                )
                self._model_id = model_id
                for family, pattern in self._FAMILY_PATTERNS.items():
                    if pattern.search(model_id):
                        self._family = family
                        return family
        except Exception:
            pass

        self._family = "unknown"
        return self._family

    @property
    def model_id(self) -> str:
        return self._model_id or "?"

    def is_gemma(self)  -> bool: return self.detect() == "gemma"
    def is_glm(self)    -> bool: return self.detect() == "glm"
    def is_qwen(self)   -> bool: return self.detect() == "qwen"


# ─── Payload builders ────────────────────────────────────────────
def _build_gemma_payload(system: str, prompt: str, max_tokens: int) -> tuple[str, dict]:
    """
    Gemma 4: usa /completion con plantilla de turno manual.
    SIN enable_thinking — Gemma 4 no tiene thinking nativo en llama.cpp.
    Los parámetros de temperatura y repeat_penalty son críticos para
    evitar que el modelo entre en colapso de probabilidad.
    """
    system_block = f"<start_of_turn>user\n{system}\n\n" if system else "<start_of_turn>user\n"
    gemma_prompt = (
        f"<bos>"
        f"{system_block}"
        f"{prompt}"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    payload = {
        "prompt":          gemma_prompt,
        "max_tokens":      max_tokens,
        "temperature":     0.35,
        "repeat_penalty":  1.1,   # evita colapso de tokens repetidos
        "stop":            ["<end_of_turn>", "<eos>", "<|im_end|>"],
    }
    return f"{WBASE}/completion", payload


def _build_glm_payload(system: str, prompt: str, max_tokens: int) -> tuple[str, dict]:
    """GLM-4: /completion con separadores de turno propios."""
    system_block = f"<|system|>{system}" if system else ""
    glm_prompt = f"{system_block}<|user|>{prompt}<|assistant|>"
    payload = {
        "prompt":      glm_prompt,
        "max_tokens":  max_tokens,
        "temperature": 0.3,
        "stop":        ["<|user|>", "<|endoftext|>"],
    }
    return f"{WBASE}/completion", payload


def _build_qwen_payload(system: str, prompt: str, max_tokens: int,
                         enable_thinking: bool) -> tuple[str, dict]:
    """Qwen y familia: /v1/chat/completions con Jinja nativo."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    return f"{WBASE}/v1/chat/completions", payload


def _build_unknown_payload(system: str, prompt: str, max_tokens: int) -> tuple[str, dict]:
    """Fallback seguro: chat/completions SIN enable_thinking."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.3,
    }
    return f"{WBASE}/v1/chat/completions", payload


def _clean_gemma_response(raw: str) -> str:
    """
    Elimina tokens de control de canal y de plantilla que Gemma 4 puede
    generar cuando el chat template no cierra correctamente los bloques.
    """
    cleaned = _GEMMA_CTRL_TOKENS.sub("", raw)
    # Limpiar <think>...</think> si aparece (no debería en Gemma, pero por seguridad)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


# ─── Router ──────────────────────────────────────────────────────
def query_router(query: str) -> tuple[str, float]:
    """Query Qwen0.8B router → returns (domain, latency_ms).
    
    v3: Uses /completion + GBNF grammar (eliminates  tags).
    Two-tier: keyword pre-filter (0.1ms) → GPU grammar (~250ms).
    """
    t0 = time.perf_counter()
    
    # Tier 1: keyword pre-filter for edge cases
    for pattern, domain in KEYWORD_RULES:
        if pattern.search(query):
            lat = (time.perf_counter() - t0) * 1000
            return domain, lat
    
    # Tier 2: GPU router with GBNF grammar
    full_prompt = ROUTER_PROMPT_HEAD.format(query=query[:300])
    try:
        r = requests.post(RURL, json={
            "prompt":      full_prompt,
            "n_predict":   12,
            "temperature": 0,
            "stop":        ["\n"],
            "grammar":     GBNF_GRAMMAR,
        }, timeout=15)
        lat = (time.perf_counter() - t0) * 1000
        
        if r.status_code != 200:
            return "general", lat
        
        content = r.json().get("content", "").strip().lower()
        valid = REASONING_DOMAINS | SPEED_DOMAINS
        return (content if content in valid else "general"), lat
    except Exception:
        lat = (time.perf_counter() - t0) * 1000
        return "general", lat


# ─── Worker ──────────────────────────────────────────────────────
_detector = WorkerModelDetector(WBASE)


def query_worker(prompt: str, domain: str, system: str = "",
                 max_tokens: int = 200) -> tuple[str, float, dict]:
    """
    Envía la query al worker (:8084) usando el endpoint y formato correcto
    para la familia de modelo activa. Detecta al primer llamado y cachea.
    """
    is_simple    = bool(SIMPLE_PATTERNS.search(prompt.strip()))
    needs_think  = domain in REASONING_DOMAINS and not is_simple
    family       = _detector.detect()

    if not system:
        style = "detallada paso a paso" if needs_think else "concisa y directa"
        system = f"Eres un experto en {domain}. Responde de forma {style}."

    # Gemma nunca tiene thinking nativo — siempre direct
    if family == "gemma":
        endpoint, payload = _build_gemma_payload(system, prompt, max_tokens)
        needs_think = False   # override: Gemma no soporta thinking en llama.cpp

    elif family == "glm":
        endpoint, payload = _build_glm_payload(system, prompt, max_tokens)
        needs_think = False   # GLM tampoco usa enable_thinking vía Jinja

    elif family == "qwen":
        mt = min(max_tokens * 3, 4096) if needs_think else max_tokens
        endpoint, payload = _build_qwen_payload(system, prompt, mt, needs_think)

    else:  # unknown — safe fallback sin Jinja
        endpoint, payload = _build_unknown_payload(system, prompt, max_tokens)
        needs_think = False

    t0 = time.perf_counter()
    try:
        timeout = 120 if needs_think else 60
        r = requests.post(endpoint, json=payload, timeout=timeout)
        lat = (time.perf_counter() - t0) * 1000
        data = r.json()
    except Exception as e:
        lat = (time.perf_counter() - t0) * 1000
        return f"[ERROR] {e}", lat, {
            "domain": domain, "thinking": needs_think, "latency_ms": round(lat),
            "tokens": 0, "finish": "error", "note": f"❌ request failed: {e}",
            "model": _detector.model_id,
        }

    # ── Extracción de contenido según endpoint ─────────────────────
    if "completion" in endpoint and "/v1/chat" not in endpoint:
        # /completion → "content" directo
        raw_content = data.get("content", "") or ""
        usage_tokens = data.get("tokens_predicted", 0) + data.get("tokens_evaluated", 0)
        finish = data.get("stop_type") or ("stop" if data.get("stop") else "length")
    else:
        # /v1/chat/completions
        choice      = data["choices"][0]
        msg         = choice["message"]
        raw_content = msg.get("content", "") or ""
        reasoning   = msg.get("reasoning_content", "") or ""
        usage_tokens = data.get("usage", {}).get("total_tokens", 0)
        finish      = choice.get("finish_reason", "?")

        # Thinking fallback: si el thinking no terminó, re-query directo
        if needs_think and not raw_content.strip() and finish == "length":
            try:
                _, fb_payload = _build_qwen_payload(system, prompt, max_tokens, False)
                fb = requests.post(endpoint, json=fb_payload, timeout=60)
                fb_data  = fb.json()
                fb_msg   = fb_data["choices"][0]["message"]
                raw_content = fb_msg.get("content", "") or reasoning
                finish      = "fallback"
            except Exception:
                raw_content = reasoning

    # ── Limpieza de respuesta ──────────────────────────────────────
    if family == "gemma":
        clean = _clean_gemma_response(raw_content)
    else:
        clean = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

    if not clean:
        clean = raw_content[:500]  # último recurso: texto crudo truncado

    # ── Nota de estado ────────────────────────────────────────────
    if family == "gemma":
        note = f"🔷 gemma-direct ({usage_tokens} tok)"
    elif needs_think:
        note = f"🧠 thinking ({usage_tokens} tok)" if finish != "fallback" else f"🔄 fallback ({usage_tokens} tok)"
    else:
        note = f"⚡ fast ({usage_tokens} tok)"

    return clean, lat, {
        "domain":     domain,
        "thinking":   needs_think,
        "latency_ms": round(lat),
        "tokens":     usage_tokens,
        "finish":     finish,
        "note":       note,
        "model":      _detector.model_id,
        "family":     family,
    }


# ─── Pipeline ─────────────────────────────────────────────────────
def process_query(query: str, system: str = "", max_tokens: int = 200) -> dict:
    t0 = time.perf_counter()
    try:
        domain, rt = query_router(query)
    except Exception as e:
        return {"error": f"Router: {e}", "query": query}
    try:
        answer, wt, meta = query_worker(query, domain, system, max_tokens)
    except Exception as e:
        return {"error": f"Worker: {e}", "domain": domain, "query": query}
    total = (time.perf_counter() - t0) * 1000
    return {
        "query":      query,
        "domain":     domain,
        "thinking":   meta["thinking"],
        "answer":     answer,
        "router_ms":  round(rt),
        "worker_ms":  round(wt),
        "total_ms":   round(total),
        "tokens":     meta["tokens"],
        "note":       meta["note"],
        "model":      meta["model"],
        "family":     meta.get("family", "?"),
        "finish":     meta["finish"],
    }


def print_result(r: dict):
    if "error" in r:
        print(f"  ❌ {r['error']}"); return
    icon = "🧠" if r["thinking"] else ("🔷" if r.get("family") == "gemma" else "⚡")
    fam  = r.get("family", "?")
    print(f"  📡 Router: {r['domain']:<12s} ({r['router_ms']}ms)")
    print(f"  {icon} Worker:  {r['worker_ms']}ms ({r['tokens']} tok) [{fam}] {r['note']}")
    print(f"  💬 {r['answer'][:250]}")
    if len(r['answer']) > 250:
        print(f"     ...({len(r['answer'])} chars)")
    print(f"  ⏱  Total: {r['total_ms']}ms [{r['model']}]")


# ─── Tests ────────────────────────────────────────────────────────
TETS = [
    ("derivative of sin(x)*cos(x) step by step",  "math"),
    ("Python async function with error handling",  "code"),
    ("explain quantum entanglement",               "science"),
    ("suspension bridge key structural elements",  "engineering"),
    ("translate good morning to Japanese",         "language"),
    ("key elements of an NDA agreement",           "law"),
    ("common symptoms of type 2 diabetes",         "medical"),
    ("calculate NPV for a startup 5yr projection", "business"),
    ("is free will compatible with determinism?",  "philosophy"),
    ("design a roguelike death penalty system",    "gaming"),
]


def cmd_test():
    family = _detector.detect(force_refresh=True)
    print(f"\n  ⚡ Quick test — 3 queries  [worker family: {family}]\n")
    for q in ["What is 2+2?", "Derivative of x^2 using chain rule", "Translate hello to spanish"]:
        r = process_query(q, max_tokens=50)
        icon = "🔷" if r.get("family") == "gemma" else ("🧠" if r.get("thinking") else "⚡")
        print(f"  {icon} {q[:35]:35s} → {r.get('domain','?'):<10s} {r.get('total_ms',0):>5}ms  \"{r.get('answer','')[:40]}\"")
    print()


def cmd_benchmark():
    family = _detector.detect(force_refresh=True)
    print(f"\n{'='*65}")
    print(f"  BENCHMARK: {len(TETS)} queries  [worker family: {family}]")
    print(f"  🧠 Thinking ON:  {sorted(REASONING_DOMAINS)}")
    print(f"{'='*65}\n")
    results = []
    for i, (q, exp) in enumerate(TETS):
        print(f"  [{i+1}/{len(TETS)}] {q[:45]:45s}", end=" ", flush=True)
        r = process_query(q, max_tokens=150)
        if "error" in r:
            print(f"❌ {r['error']}")
        else:
            ok   = "✅" if r['domain'] == exp else "❌"
            icon = "🧠" if r['thinking'] else ("🔷" if r.get("family") == "gemma" else "⚡")
            print(f"{ok} {r['domain']:<10s} {icon} {r['total_ms']:>5}ms {r['tokens']:>4}tok")
        results.append(r)
    correct = sum(1 for i, r in enumerate(results) if "error" not in r and r['domain'] == TETS[i][1])
    valid   = [r for r in results if "error" not in r]
    avg_ms  = sum(r['total_ms'] for r in valid) / len(valid) if valid else 0
    avg_tok = sum(r['tokens']   for r in valid) / len(valid) if valid else 0
    print(f"\n{'─'*65}")
    print(f"  📊 Router acc: {correct}/{len(TETS)} ({correct*100//len(TETS)}%)")
    print(f"  ⚡ Avg:        {avg_ms:.0f}ms, {avg_tok:.0f} tok")
    print(f"{'─'*65}\n")


# ─── Interactive ──────────────────────────────────────────────────
def cmd_interactive():
    family = _detector.detect(force_refresh=True)
    print("\n" + "=" * 65)
    print("  MARP Intelligent Pipeline v2.1")
    print(f"  Router: Qwen0.8B@:8082  |  Worker: {_detector.model_id} [{family}]@:8084")
    if family == "gemma":
        print("  🔷 Modo Gemma: /completion directo (sin loops Jinja)")
    elif family == "glm":
        print("  🔹 Modo GLM: /completion con separadores GLM")
    else:
        print(f"  🧠 Thinking ON:  {', '.join(sorted(REASONING_DOMAINS))}")
        print(f"  ⚡ Thinking OFF: {', '.join(sorted(SPEED_DOMAINS))}")
    print("=" * 65)
    print("  Commands: /quit  /test  /benchmark  /detect")
    print()
    while True:
        try:
            q = input("  >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!"); break
        if not q: continue
        if q in ("/quit", "/exit", "q"): break
        if q == "/test":      cmd_test();      continue
        if q == "/benchmark": cmd_benchmark(); continue
        if q == "/detect":
            f = _detector.detect(force_refresh=True)
            print(f"  Worker: {_detector.model_id} [{f}]")
            continue
        print()
        print_result(process_query(q, max_tokens=200))
        print()


# ─── CLI Dispatch ─────────────────────────────────────────────────
if __name__ == "__main__":
    if "--benchmark" in sys.argv or "-b" in sys.argv:
        cmd_benchmark()
    elif "--test" in sys.argv or "-t" in sys.argv:
        cmd_test()
    elif "--detect" in sys.argv:
        f = _detector.detect(force_refresh=True)
        print(f"Worker model: {_detector.model_id} [family: {f}]")
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        r = process_query(" ".join(sys.argv[1:]))
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        cmd_interactive()
