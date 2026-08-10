#!/bin/bash

MARP_SCRIPT="C:/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/omega_cube_mcp_server.py"
PORT=8084
HEALTH_URL="http://127.0.0.1:${PORT}/health"

# Verificar si el proceso está escuchando en el puerto
check_port() {
    ss -tlnp 2>/dev/null | grep -q ":${PORT} " || netstat -tlnp 2>/dev/null | grep -q ":${PORT} " || echo "no"
}

# Reiniciar el servidor
restart_marp() {
    echo "MARP router inactivo, reiniciando..."
    
    # Cerrar procesos existentes en el puerto
    pkill -f "omega_cube_mcp_server.py" 2>/dev/null || true
    sleep 2
    
    # Iniciar nuevo proceso en background
    python "$MARP_SCRIPT" --startup &
    MARP_PID=$!
    
    # Esperar a que se inicialice
    sleep 5
    
    # Verificar salud
    HEALTH_STATUS=$(curl -s "${HEALTH_URL}" 2>/dev/null)
    
    if [ -n "$HEALTH_STATUS" ] && echo "$HEALTH_STATUS" | grep -q "status.*:.*ok\|status.*ok"; then
        echo "✅ MARP router restarted successfully (PID: ${MARP_PID})"
        echo "Health check: ${HEALTH_STATUS}"
        return 0
    else
        echo "❌ MARP health check failed"
        echo "Response: ${HEALTH_STATUS}"
        return 1
    fi
}

# Main check
if check_port; then
    echo "MARP router is alive on port ${PORT}"
    exit 0
else
    restart_marp
    exit $?
fi
