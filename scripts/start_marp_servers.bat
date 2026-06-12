@echo off
REM ============================================================
REM MARP Server Auto-Start for Windows
REM Instalar con Task Scheduler para auto-inicio al login
REM ============================================================
TITLE MARP Servers
echo [MARP] Starting servers...

set LLAMA_DIR=C:\Users\GPAMD\Downloads\Llama.cpp Cuda\llama-b9045-bin-win-cuda-13.1-x64
set MODELS_DIR=J:\modelos_ia
set LOG_DIR=%USERPROFILE%\.hermes\logs\marp
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Kill any existing llama-server instances
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 3 /nobreak >nul

REM 1. Router Qwen0.8B (rápido, ~700MB VRAM, -c 256)
echo [MARP] Starting Router Qwen0.8B on :8084...
start "MARP-Router" /B "%LLAMA_DIR%\llama-server.exe" ^
    -m "%MODELS_DIR%\qwen3.5-0.8b-instruct-Q4_K_M.gguf" ^
    -ngl 99 -c 256 --port 8084 --host 127.0.0.1 ^
    > "%LOG_DIR%\router.log" 2>&1

timeout /t 5 /nobreak >nul

REM 2. Worker Qwen 27B Omni (tarda ~30s en cargar)
echo [MARP] Starting Worker Qwen 27B Omni on :8082...
start "MARP-Worker" /B "%LLAMA_DIR%\llama-server.exe" ^
    -m "%MODELS_DIR%\Qwen3.6-27B-Omni-v4-Q4_K_M.gguf" ^
    -ngl 99 -c 4096 --port 8082 --host 127.0.0.1 --mlock --reasoning-format none ^
    > "%LOG_DIR%\worker.log" 2>&1

REM Esperar a que el worker esté listo
echo [MARP] Waiting for worker to load (up to 60s)...
set /a counter=0
:wait_loop
timeout /t 5 /nobreak >nul
set /a counter+=1
for /f "tokens=*" %%a in ('curl -s -o nul -w "%%{http_code}" --max-time 2 http://127.0.0.1:8082/health 2^>nul') do set status=%%a
if "%status%"=="200" goto ready
if %counter% geq 12 (
    echo [MARP] WARNING: Worker not ready after 60s. Check %LOG_DIR%\worker.log
    goto done
)
goto wait_loop

:ready
echo [MARP] ✅ Router :8084 — ready
echo [MARP] ✅ Worker :8082 — ready
echo [MARP] Both servers running.

:done
echo [MARP] Use intelligent_pipeline.py to interact:
echo   cd ~/.hermes/axioma-omega-protocol/omega_cube
echo   PYTHONPATH="$PWD" python marp/intelligent_pipeline.py
