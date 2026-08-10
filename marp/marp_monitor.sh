#!/usr/bin/env bash
# Monitor MARP router - verifica puerto 8084 y reinicia si está inactivo

LOG_FILE="/home/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/marp_monitor.log"
PID_FILE="/home/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/marp.pid"
SCRIPT_PATH="/home/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/omega_cube_mcp_server.py"

# Función para verificar salud
check_health() {
    curl -s http://127.0.0.1:8084/health | grep -q '"status":"ok"'
    return $?
}

# Función para reiniciar MARP
restart_marp() {
    local last_restart=""
    
    if [ -f "$LOG_FILE" ]; then
        last_restart=$(grep "RESTARTED" "$LOG_FILE" | tail -1 | cut -d: -f2)
    fi
    
    local current_time=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Solo reiniciar si es el primer inicio o han pasado más de 10 minutos
    if [ -z "$last_restart" ] || [ "$current_time" != "$last_restart" ]; then
        # Matar proceso si existe
        if [ -f "$PID_FILE" ]; then
            kill $(cat "$PID_FILE") 2>/dev/null
            rm -f "$PID_FILE"
        fi
        
        # Esperar 2 segundos y reiniciar
        sleep 2
        
        python3 "$SCRIPT_PATH" --startup &
        local pid=$!
        echo $pid > "$PID_FILE"
        
        # Esperar a que arranque
        sleep 3
        
        # Verificar que está vivo
        if check_health; then
            echo "$(date): RESTARTED" >> "$LOG_FILE"
            echo "MARP restarted at $(date)"
        else
            echo "$(date): FAILED_TO_START" >> "$LOG_FILE"
            echo "Failed to start MARP"
        fi
    fi
}

# Verificar salud y reiniciar si es necesario
if ! check_health; then
    restart_marp
fi

echo "$(date): CHECKED - Status: $(check_health && echo 'OK' || echo 'DOWN')" >> "$LOG_FILE"
