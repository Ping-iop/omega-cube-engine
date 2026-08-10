#!/usr/bin/env python3
"""
Cron job para monitorizar y reiniciar el router MARP cada 10 minutos.
"""
import subprocess
import time
import sys
from datetime import datetime

MARP_SCRIPT = "C:/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/omega_cube_mcp_server.py"
MARP_PORT = 8084

def is_listening():
    """Verifica si algo está escuchando en el puerto."""
    try:
        result = subprocess.run(
            ["netstat", "-ano", "|", "findstr", f":{MARP_PORT}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return "LISTENING" in result.stdout
    except:
        return False

def health_check():
    """Verifica health endpoint."""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://127.0.0.1:8084/health"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

def start_marp():
    """Inicia el proceso MARP."""
    print(f"[{datetime.now()}] Iniciando router MARP...")
    subprocess.Popen(
        ["python", MARP_SCRIPT, "--startup"],
        shell=True
    )
    time.sleep(3)  # Esperar arranque

def main():
    """Ejecución única por cada invocación del cron."""
    # Verificar si está escuchando
    if not is_listening():
        print(f"[{datetime.now()}] Router MARP no activo en puerto {MARP_PORT}")
        start_marp()
        
        # Verificar health después de iniciar
        if health_check():
            print(f"[{datetime.now()}] ✅ Router MARP reiniciado correctamente")
            return True
        else:
            print(f"[{datetime.now()}] ❌ Router MARP no responde health check")
            return False
    else:
        if health_check():
            print(f"[{datetime.now()}] ✅ Router MARP activo y saludable")
            return False
        else:
            print(f"[{datetime.now()}] ⚠️  Health check fallido, reiniciando...")
            start_marp()
            return True

if __name__ == "__main__":
    main()
