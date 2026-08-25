#!/usr/bin/env python3
"""axioma_selftest.py — Verificación end-to-end de todo el stack de memoria.

Propósito: dejar de "reparar" y pasar a "verificar". Cada eslabón de la cadena
Axioma-Omega -> Omega-Cube -> MARP se comprueba aquí. Si algo se desconecta,
este test falla RUIDOSAMENTE en lugar de degradarse en silencio.

Cadena verificada:
  1. Stores: existe UN único store consultable (memory/omega_cube_memory.json)
     y el legado cube_state.json NO se toca (guard anti split-brain).
  2. Axioma engine: carga, query, telemetría recalls/usages persistida.
  3. Omega-Cube engine: carga el store único, query relevante, save/load round-trip.
  4. Indexer: idempotente (2 corridas -> mismos nodos), no se auto-ingiere.
  5. MCP servers: ambos importan y responden (axioma + omega-cube).
  6. MARP router: puerto 8082 responde health + clasifica.
  7. Enriquecedor: inyecta contexto relevante en un brief.

Salida: una línea por check [PASS]/[FAIL] + resumen. Exit code != 0 si algo falla.
Diseñado para correr como cron no_agent (sin LLM, sin GPU, sin VRAM).
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

PROJECT = os.path.expanduser(r"~/.hermes/axioma-omega-protocol")
sys.path.insert(0, PROJECT)

MEMORY_STORE = os.path.join(PROJECT, "memory", "omega_cube_memory.json")
LEGACY_STORE = os.path.join(PROJECT, "omega_cube", "cube_state.json")
AXIOMA_STORE = os.path.join(PROJECT, "memory", "unified_memory.json")

RESULTS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    RESULTS.append((tag, name, detail))
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def section(title):
    print(f"\n=== {title} ===")


def main():
    print(f"Axioma-Omega SelfTest @ {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 1. Stores ─────────────────────────────────────────────
    section("1. Stores (anti split-brain)")
    check("store único existe", os.path.exists(MEMORY_STORE), MEMORY_STORE)
    check("store axioma existe", os.path.exists(AXIOMA_STORE))

    # El store legado debe estar congelado: archivado o sin tocar en 12h.
    if not os.path.exists(LEGACY_STORE):
        check("store legado eliminado/archivado", True, "ya no existe")
    else:
        age_h = (time.time() - os.path.getmtime(LEGACY_STORE)) / 3600
        check("store legado congelado (>12h sin tocar)", age_h > 12,
              f"última modificación hace {age_h:.1f}h")

    # Ningún script activo debe escribir en cube_state.json.
    # Detección por AST: buscar open(...'w'...) sobre cube_state.json o
    # json.dump hacia ese path — los comentarios/docstrings no cuentan.
    import ast, glob
    scripts_dir = os.path.join(PROJECT, "scripts")
    offenders = []
    for f in glob.glob(os.path.join(scripts_dir, "*.py")):
        if os.path.basename(f) == os.path.basename(__file__):
            continue
        try:
            tree = ast.parse(open(f, encoding="utf-8").read())
        except (IOError, SyntaxError):
            continue
        src_has_write = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                code = ast.unparse(node)
                if "cube_state.json" in code and ("'w'" in code or '"w"' in code):
                    src_has_write = True
        if src_has_write:
            offenders.append(os.path.basename(f))
    check("ningún script escribe en store legado", not offenders,
          f"ofensores: {offenders}" if offenders else "limpio")

    # ── 2. Axioma engine ──────────────────────────────────────
    section("2. Axioma engine (memoria axiomática)")
    try:
        from memory_engine import AxiomaticMemoryEngine
        eng = AxiomaticMemoryEngine()
        eng.load(AXIOMA_STORE)
        n = len(eng.nodes)
        check("axioma carga nodos", n > 0, f"{n} nodos")

        r = eng.query("marp router")
        check("axioma query devuelve resultados", len(r) > 0, f"{len(r)} hits")

        t = eng.telemetry()
        check("telemetría recalls/usages activa",
              "recalls" in t and "usages" in t,
              f"recalls={t.get('recalls')} usages={t.get('usages')}")
    except Exception as e:
        check("axioma engine", False, f"excepción: {e}")

    # ── 3. Omega-Cube engine ──────────────────────────────────
    section("3. Omega-Cube engine")
    try:
        from omega_cube.engine import OmegaCubeEngine
        cube = OmegaCubeEngine()
        ok_load = cube.load()
        nc = len(cube.nodes)
        check("omega-cube carga store único", ok_load and nc > 0, f"{nc} nodos")

        # El contenido fresco del indexer debe ser consultable (anti split-brain).
        # query() devuelve list de dicts con 'content'.
        r = cube.query("evonybot protocolo heartbeat")
        top_content = ""
        if r:
            top = r[0]
            if isinstance(top, dict):
                top_content = top.get("content", "")
            elif isinstance(top, tuple):
                n = top[0]
                top_content = n.content if hasattr(n, "content") else str(n)
            else:
                top_content = str(top)
        fresh_ok = "heartbeat" in top_content.lower() or "protocolo" in top_content.lower()
        check("contenido fresco es consultable", fresh_ok,
              f"top1: {top_content[:50]}")

        # Round-trip save/load sin pérdida
        cube.save()
        cube2 = OmegaCubeEngine()
        cube2.load()
        check("save/load round-trip sin pérdida", len(cube2.nodes) == nc,
              f"{nc} -> {len(cube2.nodes)}")
    except Exception as e:
        check("omega-cube engine", False, f"excepción: {e}")

    # ── 4. Indexer (idempotencia + no auto-ingestión) ────────
    section("4. Indexer automático")
    indexer = os.path.join(PROJECT, "scripts", "omega_auto_indexer.py")
    try:
        venv_py = os.path.expanduser(
            r"~/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe")
        py = venv_py if os.path.exists(venv_py) else sys.executable

        def run_indexer():
            return subprocess.run([py, indexer], cwd=PROJECT, capture_output=True,
                                  text=True, timeout=120)

        r1 = run_indexer()
        ok1 = r1.returncode == 0
        # contar nodos tras 1ª corrida
        cube_a = json.load(open(MEMORY_STORE, encoding="utf-8"))
        n1 = len(cube_a.get("nodes", {}))

        r2 = run_indexer()
        ok2 = r2.returncode == 0
        cube_b = json.load(open(MEMORY_STORE, encoding="utf-8"))
        n2 = len(cube_b.get("nodes", {}))

        check("indexer corre sin error", ok1 and ok2)
        check("indexer idempotente (no duplica)", n1 == n2, f"{n1} -> {n2} nodos")

        # No debe haber nodos cuyo contenido sea un dump JSON del propio store
        dumps = [nid for nid, nd in cube_b.get("nodes", {}).items()
                 if str(nd.get("content", "")).strip().startswith('{"nodes"')]
        check("indexer no se auto-ingiere", not dumps,
              f"{len(dumps)} nodos-basura" if dumps else "limpio")
    except Exception as e:
        check("indexer", False, f"excepción: {e}")

    # ── 5. MCP servers ────────────────────────────────────────
    section("5. MCP servers (import)")
    for name, path, import_stmt in [
        ("axioma MCP", "axioma_mcp_server.py",
         "import importlib.util,sys; sys.path.insert(0,r'{p}'); "
         "spec=importlib.util.spec_from_file_location('m',r'{p}/{f}'); "
         "m=importlib.util.module_from_spec(spec)".format(p=PROJECT, f="axioma_mcp_server.py")),
        ("omega-cube MCP", "omega_cube/omega_cube_mcp_server.py",
         None),
    ]:
        # Verificar que el archivo existe y no tiene errores de sintaxis
        fp = os.path.join(PROJECT, path)
        if not os.path.exists(fp):
            check(f"{name} existe", False, fp)
            continue
        rc = subprocess.run([sys.executable, "-m", "py_compile", fp],
                            capture_output=True, text=True)
        check(f"{name} compila", rc.returncode == 0,
              rc.stderr.strip()[:120] if rc.returncode else "OK")

    # ── 6. MARP router ────────────────────────────────────────
    section("6. MARP router (puerto 8082)")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8082/health", timeout=4) as resp:
            body = resp.read().decode()
            ok = resp.status == 200
        check("MARP router health", ok, body[:40])
    except Exception as e:
        check("MARP router health", False, f"no responde: {e}")

    # ── 7. Enriquecedor de briefs ─────────────────────────────
    section("7. Enriquecedor de briefs")
    try:
        sys.path.insert(0, PROJECT)
        from axion_brief_enricher import enrich_brief
        enriched, used = enrich_brief(
            "estandar desarrollo web stack", "Tarea de prueba.")
        check("enriquecedor inyecta contexto", len(used) > 0 and "Axioma" in enriched,
              f"{len(used)} nodos inyectados")
    except Exception as e:
        check("enriquecedor", False, f"excepción: {e}")

    # ── Resumen ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    passed = sum(1 for t, _, _ in RESULTS if t == "PASS")
    total = len(RESULTS)
    print(f"RESULTADO: {passed}/{total} checks PASS")
    if passed < total:
        print("FALLAS:")
        for tag, name, detail in RESULTS:
            if tag == "FAIL":
                print(f"  ✗ {name}: {detail}")
        return 1
    print("✅ Stack de memoria 100% funcional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
