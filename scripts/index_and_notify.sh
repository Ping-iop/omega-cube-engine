#!/bin/bash
# index_and_notify.sh — Wrapper para cron job de Omega-Cube indexer
# Ejecuta el indexer cada 6 horas y notifica cambios a la memoria de Hermes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_DIR/logs/indexer.log"
MEMORY_DIR="$PROJECT_DIR/memory"

# Crear directorio de logs si no existe
mkdir -p "$PROJECT_DIR/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iniciando Omega-Cube indexer..." >> "$LOG_FILE"

# Ejecutar el indexer (asumiendo que omega_auto_indexer.py está en scripts/)
cd "$PROJECT_DIR"
python3 scripts/omega_auto_indexer.py 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    NODE_COUNT=$(python3 -c "import json; d=json.load(open('$MEMORY_DIR/omega_cube_memory.json')); print(len(d.get('nodes',{})))" 2>/dev/null || echo "unknown")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Indexer completado. Nodos totales: $NODE_COUNT" >> "$LOG_FILE"
    
    # Notificar a la memoria de Hermes si hay cambios significativos (>5 nodos nuevos)
    if [ "$NODE_COUNT" -gt 10 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] → Memoria actualizada: $NODE_COUNT nodos indexados en Omega-Cube" >> "$LOG_FILE"
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Indexer falló con código $EXIT_CODE" >> "$LOG_FILE"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cron job finalizado." >> "$LOG_FILE"
