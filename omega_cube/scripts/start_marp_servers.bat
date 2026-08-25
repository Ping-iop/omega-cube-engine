@echo off
setlocal
REM ============================================================
REM MARP Support Server - Qwen3.5-0.8B en puerto 8091
REM Rol: soporte interno de Axioma-Omega (search, indexacion, tags)
REM NO es router. El worker se carga por separado con los bat de modelo.
REM ============================================================
TITLE MARP Support Server

set LLAMA_DIR=C:\Users\GPAMD\Downloads\LLAMA~1.CPP\llama-b9045-bin-win-cuda-13.1-x64
set SUPPORT_MODEL=P:\AI_INFRA\custom_models\Qwen\Qwen3.5-0.8B-Q6_K.gguf
set SUPPORT_PORT=8091
set LOG_DIR=%USERPROFILE%\.hermes\logs\marp
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Verificar si el soporte ya esta activo
for /f %%i in ('curl -s -o NUL -w "%%{http_code}" --max-time 3 http://127.0.0.1:%SUPPORT_PORT%/health 2^>NUL') do set HTTP_CODE=%%i
if "%HTTP_CODE%"=="200" (
    echo [MARP] Support model ya activo en :%SUPPORT_PORT%.
    goto done
)

REM Verificar que el modelo existe
if not exist "%SUPPORT_MODEL%" (
    echo [ERROR] Modelo no encontrado: %SUPPORT_MODEL%
    exit /b 1
)

REM Matar solo proceso que ocupe el puerto 8091 (soporte), nunca el worker
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%SUPPORT_PORT% " ^| findstr LISTENING') do taskkill /F /PID %%p >nul 2>&1
timeout /t 2 /nobreak >nul

echo [MARP] Iniciando support model (Qwen3.5-0.8B) en :%SUPPORT_PORT%...
start "MARP-Support" /B "%LLAMA_DIR%\llama-server.exe" ^
    -m "%SUPPORT_MODEL%" ^
    -ngl 99 -c 512 --port %SUPPORT_PORT% --host 127.0.0.1 ^
    --log-disable ^
    > "%LOG_DIR%\support.log" 2>&1

REM Esperar hasta 30s
set /a counter=0
:wait_loop
timeout /t 2 /nobreak >nul
set /a counter+=1
for /f %%i in ('curl -s -o NUL -w "%%{http_code}" --max-time 2 http://127.0.0.1:%SUPPORT_PORT%/health 2^>NUL') do set HTTP_CODE=%%i
if "%HTTP_CODE%"=="200" (
    echo [MARP] Support model activo en :%SUPPORT_PORT%.
    goto done
)
if %counter% GEQ 15 (
    echo [ERROR] Support model no respondio en 30s. Ver %LOG_DIR%\support.log
    exit /b 1
)
goto wait_loop

:done
echo [MARP] Listo. El worker se carga por separado con los bat de modelo.
exit /b 0
