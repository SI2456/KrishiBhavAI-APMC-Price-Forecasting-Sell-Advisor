# Fetch today's Gujarat mandi prices from data.gov.in and push them to GitHub.
#
# data.gov.in does not answer GitHub's servers, so prices are fetched from this
# laptop. The Windows scheduled task "KrishiBhav daily prices" runs this script
# every evening (set it up once with scripts\install_task.ps1). Everything it
# does is written to logs\daily_prices.log.
#
# Safety rules:
# - commits only files in data\raw\datagov, never your other changes
# - only commits and pushes when the repo is on the main branch
# - does not push if you have your own unpushed commits (it logs it instead)

$Repo = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Repo "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir "daily_prices.log"
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$Git = (Get-Command git -ErrorAction SilentlyContinue).Source

function Say([string]$Message) {
    Add-Content -Path $Log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message" -Encoding UTF8
}

# Run a program, append its output to the log, return its exit code.
# (Start-Process keeps PowerShell 5.1 from turning stderr lines into errors.)
function Run([string]$Exe, [string[]]$Arguments) {
    $out = Join-Path $env:TEMP "krishibhav_out.txt"
    $err = Join-Path $env:TEMP "krishibhav_err.txt"
    $p = Start-Process -FilePath $Exe -ArgumentList $Arguments -WorkingDirectory $Repo -NoNewWindow `
        -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
    foreach ($f in @($out, $err)) {
        $text = Get-Content -Path $f -Raw -ErrorAction SilentlyContinue
        if ($text) { Add-Content -Path $Log -Value $text.TrimEnd() -Encoding UTF8 }
    }
    return $p.ExitCode
}

Say "==== start"
if (-not (Test-Path $Python)) { Say "ERROR: $Python not found. Create the .venv first (README step 1)."; exit 1 }
if (-not $Git) { Say "ERROR: git not found on PATH."; exit 1 }
Set-Location $Repo

$branch = (& $Git rev-parse --abbrev-ref HEAD).Trim()
$onMain = $branch -eq "main"
if ($onMain) {
    # Get the bot's weather-forecast commits first, so the push below is a fast-forward.
    if ((Run $Git @("pull", "--rebase", "--autostash", "origin", "main")) -ne 0) {
        Say "WARNING: git pull failed; will fetch prices but not push."
        $onMain = $false
    }
} else {
    Say "Repo is on branch '$branch', not main: prices will be saved but not committed."
}

if ((Run $Python @("-m", "krishibhav", "fetch-today")) -ne 0) {
    Say "ERROR: fetch-today failed (data.gov.in down or slow?). Next run will try again."
    exit 1
}
if (-not $onMain) { Say "done (not committed)"; exit 0 }

Run $Git @("add", "--", "data/raw/datagov") | Out-Null
$changed = @(& $Git diff --cached --name-only -- data/raw/datagov) | Where-Object { $_ }
if ($changed.Count -eq 0) { Say "No new prices since the last run."; Say "done"; exit 0 }

$days = ($changed | ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_) }) -join ", "
if ((Run $Git @("commit", "-m", "`"data: mandi prices $days`"", "--", "data/raw/datagov")) -ne 0) {
    Say "ERROR: git commit failed."; exit 1
}

$ahead = [int]((& $Git rev-list --count origin/main..HEAD).Trim())
if ($ahead -gt 1) {
    Say "You have $($ahead - 1) unpushed commit(s) of your own, so the price commit was NOT pushed. Push when ready."
    Say "done"; exit 0
}
if ((Run $Git @("push", "origin", "main")) -ne 0) {
    Say "Push rejected; pulling and trying once more."
    Run $Git @("pull", "--rebase", "--autostash", "origin", "main") | Out-Null
    if ((Run $Git @("push", "origin", "main")) -ne 0) { Say "ERROR: git push failed."; exit 1 }
}
Say "Pushed prices for $days"
Say "done"
exit 0
