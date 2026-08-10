# Crear tarea de Task Scheduler para chequeo cada 10 minutos
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c C:\Users\GPAMD\.hermes\axioma-omega-protocol\omega_cube\marp\marp_health_check.bat >> C:\Users\GPAMD\.hermes\axioma-omega-protocol\omega_cube\marp\marp_cron.log"
$trigger = New-ScheduledTaskTrigger -OnceAt (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Hours 999)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet
try {
    Register-ScheduledTask -TaskName "MARP_Router_HealthCheck" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force
    Write-Host "Task Scheduler configured successfully"
} catch {
    Write-Host "Error: $_"
}
