#!/bin/bash
# Configurar cron para chequeo cada 10 minutos
echo "*/10 * * * * C:/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/marp_health_check.bat >> C:/Users/GPAMD/.hermes/axioma-omega-protocol/omega_cube/marp/marp_cron.log 2>&1" | crontab -
crontab -l
