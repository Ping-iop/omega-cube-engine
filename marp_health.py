import requests
try:
    r = requests.get("http://127.0.0.1:8082/health", timeout=5)
    status = "OK" if r.json().get("status") == "ok" else "DOWN"
except Exception as e:
    status = "DOWN"
    print(f"  Error: {e}")

print(f"MARP GPU Router: {status}")
if status == "DOWN":
    print("WARNING: llama-server is not running. Start it with:")
    print(r"  C:\Users\GPAMD\.hermes\scripts\start_marp_gpu.bat")
