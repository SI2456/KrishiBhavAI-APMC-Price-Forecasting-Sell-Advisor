# Register (or update) the Windows scheduled task that runs daily_prices.ps1.
# Run once from the project folder:
#   powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1
#
# It runs at 21:00 and again at 23:30 (to catch yards that report late), only
# while you are logged in, so git can use your saved GitHub login. If the
# laptop is off or asleep at those times, it runs as soon as it is back on.
# Remove it with:  Unregister-ScheduledTask -TaskName "KrishiBhav daily prices"

param([string[]]$At = @("21:00", "23:30"))

$TaskName = "KrishiBhav daily prices"
$Repo = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $PSScriptRoot "daily_prices.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Script`"" `
    -WorkingDirectory $Repo
$triggers = $At | ForEach-Object { New-ScheduledTaskTrigger -Daily -At $_ }
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings `
    -Description "Fetch Gujarat mandi prices from data.gov.in and push them to GitHub (KrishiBhav)." `
    -Force | Out-Null

$info = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Output "Scheduled '$TaskName' daily at $($At -join ' and '). Next run: $($info.NextRunTime)"
Write-Output "Log: $(Join-Path $Repo 'logs\daily_prices.log')"
