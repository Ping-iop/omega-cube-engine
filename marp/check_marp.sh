#!/usr/bin/env bash

HEALTH_URL="http://127.0.0.1:8084/health"
LOG_FILE="$HOME/.hermes/axioma-omega-protocol/omega_cube/marp/marp_status.log"

check_and_restart() {
    RESPONSE=$(curl -s --max-time 5 "$HEALTH_URL" 2>/dev/null)
    STATUS=$(echo "$RESPONSE" | grep -o status:[^]*' | cut -d' -f4)
    
    if [ "$STATUS" != "ok" ]; then
        echo "[$(date \"+%Y-%m-%d %H:%M:%S\")] MARP router dead - restarting..." >> "$LOG_FILE"
        
        pkill -f omega_cube_mcp_server.py 2>/dev/null
        
        python C:/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/omega_cube_mcp_server.py --startup &
        PID=$!
        echo "[$(date \"+%Y-%m-%d %H:%M:%S\")] PID: $PID" >> "$LOG_FILE"
        
        sleep 5
        
        RESPONSE=$(curl -s --max-time 5 "$HEALTH_URL" 2>/dev/null)
        STATUS=$(echo "$RESPONSE" | grep -o status:[^]*' | cut -d' -f4)
        
        if [ "$STATUS" = "ok" ]; then
            echo "[$(date \"+%Y-%m-%d %H:%M:%S\")] MARP router restarted successfully" >> "$LOG_FILE"
            echo "MARP router restarted at $(date)"
            return 0
        else
            echo "[$(date \"+%Y-%m-%d %H:%M:%S\")] Failed to restart MARP router" >> "$LOG_FILE"
            echo "MARP router restart FAILED at $(date)"
            return 1
        fi
    else
        echo "[$(date \"+%Y-%m-%d %H:%M:%S\")] MARP router healthy (status: $STATUS)" >> "$LOG_FILE"
        return 0
    fi
}

check_and_restart
