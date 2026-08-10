@echo off
curl -s http://127.0.0.1:8084/health | findstr /i "status:ok" >nul 2>&1
if %errorlevel% neq 0 (
    echo Router dead - restarting...
    taskkill /F /FI "WINDOWTITLE eq omega_cube_mcp_server" /FI "IMAGENAME eq python.exe" 2>nul
    start "" python "C:\Users\GPAMD\.hermes\axioma-omega-protocol\omega_cube\marp\omega_cube_mcp_server.py" --startup
) else (
    echo Router healthy
)

