#!/usr/bin/env python3
"""MARP Router Monitor - Checks every 10 minutes, restarts if dead."""

import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime

LOGFILE = Path("/c/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp_monitor.log")
RESTART_FLAG = Path("/c/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp_last_restart.txt")

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}\n"
    LOGFILE.write_text(LOGFILE.read_text() + line, encoding='utf-8', errors='replace')
    print(line.strip())

def check_health():
    """Check if MARP router is responding on port 8084."""
    try:
        result = subprocess.run(
            ['curl', '-s', '--max-time', '5', 'http://127.0.0.1:8084/health'],
            capture_output=True, text=True, timeout=10
        )
        return "ok" in result.stdout.lower()
    except Exception as e:
        log(f"Health check error: {e}")
        return False

def restart_marp():
    """Kill existing process and restart MARP router."""
    # Kill any existing process on port 8084
    try:
        subprocess.run(['fuser', '-k', '8084/tcp'], capture_output=True, timeout=5)
    except Exception as e:
        log(f"Warning killing port: {e}")

    # Start new instance in background using nohup-style approach
    server_script = Path("/c/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/omega_cube_mcp_server.py")
    
    try:
        proc = subprocess.Popen(
            ['python3', str(server_script), '--startup'],
            stdout=open(LOGFILE, 'a'),
            stderr=subprocess.STDOUT,
            cwd=str(server_script.parent)
        )
        log(f"Started new process PID={proc.pid}")
        return True
    except Exception as e:
        log(f"Failed to start: {e}")
        return False

def main():
    log("MARP Monitor started")
    
    while True:
        if check_health():
            log("OK - MARP router alive on 8084")
        else:
            log("DEAD - Restarting...")
            if restart_marp():
                time.sleep(5)
                if check_health():
                    log("RESTARTED successfully")
                    RESTART_FLAG.write_text(datetime.now().isoformat())
                else:
                    log("FAILED - restart did not work")
        
        time.sleep(600)  # 10 minutes

if __name__ == '__main__':
    main()
