#!/usr/bin/env python3
import schedule
import time
import subprocess
import requests
import os

HEALTH_URL = "http://127.0.0.1:8084/health"
LOG_FILE = os.path.expanduser("~/.hermes/axioma-omega-protocol/omega_cube/marp/marp_status.log")

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def check_and_restart():
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        status = response.json().get("status", "")
        
        if status != "ok":
            log("MARP router dead - restarting...")
            subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq omega_cube_mcp_server", "/FI", "IMAGENAME eq python.exe"], stderr=subprocess.DEVNULL)
            
            log("Starting MARP server...")
            subprocess.Popen(["python", "C:/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/omega_cube_mcp_server.py", "--startup"])
            
            time.sleep(5)
            
            try:
                response2 = requests.get(HEALTH_URL, timeout=5)
                status2 = response2.json().get("status", "")
                if status2 == "ok":
                    log("MARP router restarted successfully")
                    print("MARP router restarted")
                else:
                    log("Failed to restart MARP router")
                    print("MARP router restart FAILED")
            except:
                log("Failed to verify restart")
                print("MARP router restart FAILED to verify")
        else:
            log(f"MARP router healthy (status: {status})")
    except Exception as e:
        log(f"Error checking MARP: {e}")
        print(f"Error: {e}")

# Schedule the job every 10 minutes
schedule.every(10).minutes.do(check_and_restart)

print("MARP Monitor started. Press Ctrl+C to stop.")
log("MARP Monitor started")

try:
    while True:
        schedule.run_pending()
        time.sleep(60)
except KeyboardInterrupt:
    log("MARP Monitor stopped")
