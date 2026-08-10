#!/usr/bin/env bash
LOG_FILE="/c/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/marp_monitor.log"
PID_FILE="/c/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/marp.pid"
SCRIPT_PATH="/c/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/omega_cube_mcp_server.py"
LAST_RESTART="/c/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/.last_restart"

check_health() {
    curl -s http://127.0.0.1:8084/health | grep -q '"status":"ok"'
    return $?
}

restart_marp() {
    local current_time=$(date "+%Y-%m-%d %H:%M:%S")
    if [ -f "$LAST_RESTART" ]; then
        local stored=$(cat "$LAST_RESTART")
        if [ "$stored" = "$current_time" ]; then
            return 1
        fi
    fi
    if [ -f "$PID_FILE" ]; then
        kill $(cat "$PID_FILE") 2>/dev/null
        rm -f "$PID_FILE"
    fi
    python3 "$SCRIPT_PATH" --startup &
    local pid=$!
    echo $pid > "$PID_FILE"
    echo $current_time > "$LAST_RESTART"
    sleep 3
    if check_health; then
        echo "$(date): RESTARTED" >> "$LOG_FILE"
        echo "[$(date)] MARP restarted"
        return 0
    else
        echo "$(date): FAILED_TO_START" >> "$LOG_FILE"
        echo "[$(date)] Failed to start MARP"
        return 1
    fi
}

if ! check_health; then
    restart_marp
fi

echo "$(date): CHECKED - $(check_health && echo 'OK' || echo 'DOWN')" >> "$LOG_FILE"

