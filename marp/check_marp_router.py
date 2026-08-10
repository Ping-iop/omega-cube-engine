#!/usr/bin/env python3
"""
MARP Router Health Check & Auto-restart
Verifica el router en puerto 8084 y lo reinicia si está caído.
"""
import subprocess
import time
import sys
import requests
from datetime import datetime

MARP_PORT = 8084
MARP_URL = f"http://127.0.0.1:{MARP_PORT}/health"
MARP_SCRIPT = "C:/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/omega_cube_mcp_server.py"

def is_process_running():
    """Verifica si el proceso MARP está activo en el puerto 8084."""
    try:
        result = subprocess.run(
            ["netstat", "-ano", "|", "findstr", ":8084"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if f":8084" in line and "LISTENING" in line:
                return True
    except Exception:
        pass
    return False

def health_check():
    """Realiza check de salud con curl."""
    try:
        result = subprocess.run(
            ["curl", "-s", MARP_URL],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False

def start_marp_router():
    """Inicia el router MARP."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando router MARP...")
    subprocess.Popen(
        ["python", MARP_SCRIPT, "--startup"],
        shell=True
    )
    time.sleep(3)  # Esperar a que inicie

def check_and_restart():
    """Verifica y reinicia si es necesario."""
    if not is_process_running():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Router MARP no está activo en puerto 8084")
        
        start_marp_router()
        
        if health_check():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Router MARP restaurado exitosamente")
            return True  # Se reinició
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Router MARP no responde después de reiniciar")
            return True  # Aun así intentó reiniciar
    else:
        if health_check():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Router MARP activo y saludable")
            return False  # No se reinició
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Router MARP está pero health check fallido, reiniciando...")
            start_marp_router()
            return True  # Se reinició por health check

if __name__ == "__main__":
    # Ejecutar una vez por cron
    result = check_and_restart()
    sys.exit(0)
