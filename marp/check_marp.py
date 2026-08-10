#!/usr/bin/env python3
"""
Verificador y reiniciador del router MARP.
Verifica cada 10 minutos si el servicio está vivo en el puerto 8084.
"""
import subprocess
import time
import sys
from datetime import datetime

def is_marp_alive():
    """Verifica si el router MARP está respondiendo en el puerto 8084."""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://127.0.0.1:8084/health"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False

def start_marp():
    """Inicia el router MARP."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando MARP router...")
    try:
        process = subprocess.Popen(
            [
                "python",
                "C:/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/omega_cube_mcp_server.py",
                "--startup"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        # Esperar a que inicie
        time.sleep(3)
        return process
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR iniciando MARP: {e}")
        return None

def main():
    """Verifica el estado y reinicia si es necesario."""
    if is_marp_alive():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] MARP router está vivo ✅")
        return 0
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] MARP router inactivo ❌ Reiniciando...")
        process = start_marp()
        if process:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Esperando inicio...")
            # Verificar después de esperar
            time.sleep(5)
            if is_marp_alive():
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Router MARP reiniciado correctamente ✅")
                return 0
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: Router no arrancó después de reiniciar")
                return 1
        return 1

if __name__ == "__main__":
    sys.exit(main())
