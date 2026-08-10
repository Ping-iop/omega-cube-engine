@echo off
curl -s http://127.0.0.1:8084/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo %date% %time% - Router MARP alive
) else (
    echo %date% %time% - Reiniciando Router MARP...
    python C:/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/omega_cube_mcp_server.py --startup
    timeout /t 5 /nobreak >nul
    curl -s http://127.0.0.1:8084/health
)
