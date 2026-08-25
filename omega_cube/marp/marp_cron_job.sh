#!/usr/bin/env bash

LOG_FILE="/tmp/marp_cron.log"

echo "$(date): Verificando router MARP..." >> "$LOG_FILE"

# Correr el script de monitor y guardar el resultado
if [ -f /tmp/marp_restarted_*.txt ]; then
    # Listar archivos de reinicio
    ls -la /tmp/marp_restarted_* 2>/dev/null >> "$LOG_FILE"
    echo "$(date): Router MARP fue reiniciado (archivos de reinicio detectados)" >> "$LOG_FILE"
fi
