$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = Split-Path -Parent $MyInvocation.MyCommand.Path
}

Set-Location -LiteralPath $Root

$VenvName = "lab_env"
$PythonVersion = "3.13"

$VenvDir = Join-Path $Root $VenvName
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsFile = Join-Path $Root "requirements.txt"

$ToolsDir = Join-Path $Root ".tools"
$UvDir = Join-Path $ToolsDir "uv"
$UvExe = Join-Path $UvDir "uv.exe"
$UvPythonDir = Join-Path $ToolsDir "python"
$UvCacheDir = Join-Path $ToolsDir "uv-cache"

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)

    Write-Host ""
    Write-Host "==> $Message"
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$File,
        [string[]]$Arguments = @()
    )

    & $File @Arguments
    $exitCode = $LASTEXITCODE
    if ($null -ne $exitCode -and $exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $File $($Arguments -join ' ')"
    }
}

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
}

function Test-LabEnvPath {
    $expected = Get-FullPath (Join-Path $Root $VenvName)
    $actual = Get-FullPath $VenvDir

    return $actual.Equals($expected, [System.StringComparison]::OrdinalIgnoreCase)
}

function Move-ExistingLabEnvAside {
    if (-not (Test-Path -LiteralPath $VenvDir)) {
        return
    }

    if (-not (Test-LabEnvPath)) {
        throw "Refusing to move unexpected virtual environment path: $VenvDir"
    }

    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupDir = Join-Path $Root "$VenvName.backup-$stamp"
    $index = 1

    while (Test-Path -LiteralPath $backupDir) {
        $backupDir = Join-Path $Root "$VenvName.backup-$stamp-$index"
        $index += 1
    }

    Move-Item -LiteralPath $VenvDir -Destination $backupDir
    Write-Host "Moved existing $VenvName to: $backupDir"
}

function Get-VenvPythonVersion {
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        return $null
    }

    $version = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($version)) {
        return $null
    }

    return ($version | Select-Object -First 1).Trim()
}

function Test-RequiredPythonVersion {
    param([string]$Version)

    if ([string]::IsNullOrWhiteSpace($Version)) {
        return $false
    }

    return ($Version -match "^3\.13(\.|$)")
}

function Initialize-UvEnvironment {
    New-Item -ItemType Directory -Force -Path $UvDir, $UvPythonDir, $UvCacheDir | Out-Null

    $env:UV_INSTALL_DIR = $UvDir
    $env:UV_NO_MODIFY_PATH = "1"
    $env:UV_PYTHON_INSTALL_DIR = $UvPythonDir
    $env:UV_CACHE_DIR = $UvCacheDir
}

function Ensure-LocalUv {
    Initialize-UvEnvironment

    if (Test-Path -LiteralPath $UvExe) {
        return $UvExe
    }

    Write-Step "Installing uv locally"
    try {
        $installScript = Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1"
        Invoke-Expression $installScript
    }
    catch {
        throw "Failed to install uv. Check your network connection, then rerun this script. $($_.Exception.Message)"
    }

    if (-not (Test-Path -LiteralPath $UvExe)) {
        throw "uv installer completed, but uv.exe was not found at: $UvExe"
    }

    return $UvExe
}

function Ensure-LabVenv {
    $existingVersion = Get-VenvPythonVersion

    if (Test-RequiredPythonVersion $existingVersion) {
        Write-Step "Found $VenvName with Python $existingVersion"
        return
    }

    if (Test-Path -LiteralPath $VenvDir) {
        if ([string]::IsNullOrWhiteSpace($existingVersion)) {
            Write-Step "$VenvName exists but is not a usable virtual environment"
        }
        else {
            Write-Step "$VenvName uses Python $existingVersion, not Python $PythonVersion"
        }

        Move-ExistingLabEnvAside
    }

    $uv = Ensure-LocalUv

    Write-Step "Installing managed Python $PythonVersion with uv"
    Invoke-Checked -File $uv -Arguments @("python", "install", $PythonVersion)

    Write-Step "Creating $VenvName with Python $PythonVersion"
    Invoke-Checked -File $uv -Arguments @("venv", "--python", $PythonVersion, "--seed", $VenvDir)

    $createdVersion = Get-VenvPythonVersion
    if (-not (Test-RequiredPythonVersion $createdVersion)) {
        throw "Expected Python $PythonVersion in $VenvName, but found: $createdVersion"
    }
}

function Test-PipAvailable {
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        return $false
    }

    & $PythonExe -m pip --version *> $null
    return ($LASTEXITCODE -eq 0)
}

function Install-Requirements {
    if (-not (Test-Path -LiteralPath $RequirementsFile)) {
        Write-Step "No requirements.txt found; skipping package installation"
        return
    }

    Write-Step "Checking packages from requirements.txt"

    if (Test-PipAvailable) {
        Invoke-Checked -File $PythonExe -Arguments @("-m", "pip", "install", "-r", $RequirementsFile)
        return
    }

    $uv = Ensure-LocalUv
    Invoke-Checked -File $uv -Arguments @("pip", "install", "--python", $PythonExe, "-r", $RequirementsFile)
}

function Remove-PathEntry {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $true)][string]$EntryToRemove
    )

    $entryFullPath = Get-FullPath $EntryToRemove
    $keptParts = New-Object System.Collections.Generic.List[string]

    foreach ($part in ($PathValue -split ';')) {
        if ([string]::IsNullOrWhiteSpace($part)) {
            continue
        }

        try {
            $partFullPath = Get-FullPath $part
            if ($partFullPath.Equals($entryFullPath, [System.StringComparison]::OrdinalIgnoreCase)) {
                continue
            }
        }
        catch {
            # Keep unusual PATH entries that are not normal filesystem paths.
        }

        $keptParts.Add($part)
    }

    return ($keptParts -join ';')
}

function Enable-LabEnv {
    $scriptsDir = Join-Path $VenvDir "Scripts"
    $activateScript = Join-Path $scriptsDir "Activate.ps1"

    Write-Step "Activating $VenvName"

    if (Test-Path -LiteralPath $activateScript) {
        try {
            & $activateScript
        }
        catch {
            Write-Warning "Activate.ps1 failed, applying PATH activation directly. $($_.Exception.Message)"
        }
    }

    if (Test-Path Env:PYTHONHOME) {
        Remove-Item Env:PYTHONHOME
    }

    $env:VIRTUAL_ENV = $VenvDir
    $env:PATH = "$(Get-FullPath $scriptsDir);$(Remove-PathEntry -PathValue $env:PATH -EntryToRemove $scriptsDir)"
}

Ensure-LabVenv
Install-Requirements
Enable-LabEnv

$activePython = Get-Command python -ErrorAction SilentlyContinue
$activeVersion = & $PythonExe --version

Write-Host ""
Write-Host "Setup complete. You can use python now."
Write-Host "Python: $activeVersion"
if ($null -ne $activePython) {
    Write-Host "Using: $($activePython.Source)"
}
