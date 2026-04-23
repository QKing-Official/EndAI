# ─────────────────────────────────────────────────────────────────────────────
#  EndAI Setup — Windows Bootstrap
#  Run via:
#    irm https://raw.githubusercontent.com/QKing-Official/EndAI/main/setup.ps1 | iex
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Continue"
$RAW = "https://raw.githubusercontent.com/QKing-Official/EndAI/main"

function Log   { param($m) Write-Host "  -> $m" -ForegroundColor Cyan }
function Ok    { param($m) Write-Host "  + $m"  -ForegroundColor Green }
function Warn  { param($m) Write-Host "  ! $m"  -ForegroundColor Yellow }
function Err   { param($m) Write-Host "  x $m"  -ForegroundColor Red; exit 1 }
function Step  { param($m) Write-Host "  * $m"  -ForegroundColor Magenta }
function Ask {
    param([string]$Prompt, [string]$Default = "y")
    $hint = if ($Default -eq "y") { "[Y/n]" } else { "[y/N]" }
    $ans = Read-Host "  ? $Prompt $hint"
    if ([string]::IsNullOrWhiteSpace($ans)) { $ans = $Default }
    return $ans.ToLower() -eq "y"
}

function Banner {
    Write-Host ""
    Write-Host "  ###### #  # ####     ##  ### " -ForegroundColor Cyan
    Write-Host "  ##     ## # ##  #   ##  ##   " -ForegroundColor Cyan
    Write-Host "  ####   # ## ##  #  ######  # " -ForegroundColor Cyan
    Write-Host "  ##     #  # ##  #  ##  ##   ##" -ForegroundColor Cyan
    Write-Host "  ###### #  # ####   ##  # ###  " -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Universal Setup -- github.com/QKing-Official/EndAI" -ForegroundColor DarkGray
    Write-Host ""
}

# ── Try WSL first (best experience on Windows) ────────────────────────────────
function Try-WSL {
    if (-not (Get-Command "wsl" -ErrorAction SilentlyContinue)) { return $false }
    $distros = wsl --list --quiet 2>$null
    if (-not $distros) { return $false }
    Ok "WSL detected — running setup inside WSL for best compatibility"
    $cmd = "curl -fsSL $RAW/setup.sh | bash"
    wsl bash -c $cmd
    return $true
}

# ── GPU Detection ─────────────────────────────────────────────────────────────
function Detect-GPU {
    $script:GpuType = "cpu"; $script:GpuName = "CPU only"; $script:CudaTag = "cu121"
    if (Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue) {
        try {
            $name = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
            if ($name) {
                $script:GpuType = "nvidia"; $script:GpuName = $name.Trim()
                $driver = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1
                $major = [int]($driver.Trim().Split(".")[0])
                $script:CudaTag = if ($major -ge 545) { "cu121" } else { "cu118" }
                return
            }
        } catch {}
    }
    try {
        $gpu = Get-WmiObject Win32_VideoController | Select-Object -First 1
        if ($gpu) { $script:GpuName = "$($gpu.Name) (CPU inference)" }
    } catch {}
}

# ── Python ────────────────────────────────────────────────────────────────────
function Find-Python {
    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $minor = & $cmd -c "import sys; print(sys.version_info.minor)" 2>$null
            $major = & $cmd -c "import sys; print(sys.version_info.major)" 2>$null
            if ([int]$major -eq 3 -and [int]$minor -ge 10) {
                $script:Py = $cmd
                $script:PyVer = (& $cmd --version 2>&1).ToString().Split(" ")[1]
                return $true
            }
        } catch {}
    }
    return $false
}

function Install-Python {
    Warn "Python 3.10+ not found."
    if (Get-Command "winget" -ErrorAction SilentlyContinue) {
        Step "Installing Python 3.11 via winget..."
        winget install --id Python.Python.3.11 --silent --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path","User")
    } else {
        Err "Install Python 3.10+ from https://python.org (tick 'Add Python to PATH')"
    }
}

function Ensure-Git {
    if (Get-Command "git" -ErrorAction SilentlyContinue) { return }
    Warn "git not found — installing via winget..."
    if (Get-Command "winget" -ErrorAction SilentlyContinue) {
        winget install --id Git.Git --silent --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path","User")
    } else { Err "Install git from https://git-scm.com" }
}

function Pip { param([string[]]$pkgs, [string]$extra = "")
    $str = $pkgs -join " "
    Invoke-Expression "$script:Py -m pip install $str $extra --quiet" | Out-Null
    return $LASTEXITCODE -eq 0
}

# ── Native Windows Setup ──────────────────────────────────────────────────────
function Run-NativeSetup {
    Log "Running native Windows setup..."

    # Python
    Write-Host "`n[1/6] Python" -ForegroundColor White
    if (-not (Find-Python)) { Install-Python; Find-Python | Out-Null }
    if (-not (Find-Python)) { Err "Python 3.10+ required. Install from https://python.org" }
    Ok "Python $script:PyVer"

    # Repo
    Write-Host "`n[2/6] Repository" -ForegroundColor White
    if (Test-Path "server.py") {
        $script:Repo = (Get-Location).Path; Ok "Already in EndAI repo"
    } elseif (Test-Path "EndAI") {
        $script:Repo = (Resolve-Path "EndAI").Path; Ok "Found EndAI/"
    } else {
        Ensure-Git
        Step "Cloning EndAI..."
        git clone https://github.com/QKing-Official/EndAI.git
        $script:Repo = (Resolve-Path "EndAI").Path
        Ok "Cloned to $script:Repo"
    }

    # Base deps
    Write-Host "`n[3/6] Base Dependencies" -ForegroundColor White
    Invoke-Expression "$script:Py -m pip install --upgrade pip --quiet" | Out-Null; Ok "pip upgraded"
    foreach ($p in @("flask","numpy","psutil")) {
        Step "Installing $p..."; if (Pip @($p)) { Ok $p } else { Warn "Failed: $p" }
    }

    # llama-cpp-python
    Write-Host "`n[4/6] llama-cpp-python ($script:GpuType)" -ForegroundColor White
    switch ($script:GpuType) {
        "nvidia" {
            Log "NVIDIA — $script:CudaTag"
            $idx = "https://abetlen.github.io/llama-cpp-python/whl/$script:CudaTag"
            if (Pip @("llama-cpp-python") -extra "--extra-index-url $idx") { Ok "llama-cpp-python (CUDA)" }
            else { Warn "CUDA failed — CPU fallback"; Pip @("llama-cpp-python") | Out-Null }
        }
        default {
            if (Pip @("llama-cpp-python")) { Ok "llama-cpp-python (CPU)" } else { Warn "Install failed" }
        }
    }

    # Models
    Write-Host "`n[5/6] Models" -ForegroundColor White
    $modDir = Join-Path $script:Repo "models"
    New-Item -ItemType Directory -Force -Path $modDir | Out-Null
    Ok "models/ ready"
    if (-not (Get-ChildItem "$modDir\*.gguf" -ErrorAction SilentlyContinue)) {
        Warn "No .gguf files found."
        Write-Host "    Full : https://huggingface.co/QKing-Official/EndAI" -ForegroundColor Cyan
        Write-Host "    Small: https://huggingface.co/QKing-Official/EndAI-Small" -ForegroundColor Cyan
        if (Ask "Download EndAI-Small Q4_K_M (~700 MB)?" "n") {
            $url  = "https://huggingface.co/QKing-Official/EndAI-Small/resolve/main/EndAI-Small-Q4_K_M.gguf"
            $dest = Join-Path $modDir "EndAI-Small-Q4_K_M.gguf"
            Step "Downloading..."
            try { (New-Object System.Net.WebClient).DownloadFile($url, $dest); Ok "Done" }
            catch { Warn "Download failed: $_" }
        }
    } else {
        Get-ChildItem "$modDir\*.gguf" | ForEach-Object { Ok "Found: $($_.Name)" }
    }

    # Summary
    Write-Host ""
    Write-Host "  ======================================" -ForegroundColor Green
    Write-Host "   Setup Complete!" -ForegroundColor Green
    Write-Host "  ======================================" -ForegroundColor Green
    Write-Host "  GPU   : $script:GpuName" -ForegroundColor DarkGray
    Write-Host "  Python: $script:PyVer" -ForegroundColor DarkGray
    Write-Host "  Repo  : $script:Repo" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Start: cd `"$script:Repo`" && $script:Py server.py" -ForegroundColor Cyan
    Write-Host "  Open : http://localhost:5000" -ForegroundColor Cyan
    Write-Host ""

    if (Ask "Launch EndAI now?" "y") {
        $srv = Join-Path $script:Repo "server.py"
        if (Test-Path $srv) { Set-Location $script:Repo; Invoke-Expression "$script:Py server.py" }
        else { Warn "server.py not found" }
    }
}

# ── Entry Point ───────────────────────────────────────────────────────────────
Banner
Detect-GPU
Log "GPU: $script:GpuName [$script:GpuType]"
Write-Host ""
if (-not (Ask "Proceed?" "y")) { Write-Host "Aborted."; exit 0 }

if (-not (Try-WSL)) {
    Run-NativeSetup
}