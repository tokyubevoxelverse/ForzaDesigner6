# GPU smoke runner - automates the steps in docs/CURSOR_GPU_SMOKE_TEST.md
#
# Usage (from anywhere on QUASAR):
#   powershell -NoProfile -ExecutionPolicy Bypass -File `
#     "\\QUASAR\ContentCreation\ForzaAbyssPainter_build\diagnostics\run_gpu_smoke.ps1"
#
# Also copies to diagnostics\smoke_results_upstream\ when run from Cursor.
#
# Produces:
#   ~\Desktop\smoke_results\
#     smoke_logo_1000.json          # generated output
#     smoke_logo_1000.stderr.log    # full IPC line stream
#     cursor_gpu_probe.json         # state probe (host + GPU + runtime)
#     smoke_diag.zip                # logs + marker + sysinfo bundle
#     smoke_summary.md              # human-readable summary report
#
# Steps 1 (fap-generate) + 4 (probe + diag) are automated; step 3
# (cancel mid-run) requires a human or a sleeper script and is left
# as an exercise - Cursor can run it manually or extend this script.

$ErrorActionPreference = 'Continue'   # don't bail on first non-zero exit

# --- paths ---
# Script lives on the QUASAR share under diagnostics\ - resolve build root from here.
$buildRoot = Split-Path $PSScriptRoot -Parent
$desktopBuild = Join-Path $env:USERPROFILE 'Desktop\ForzaAbyssPainter_build'
if (-not (Test-Path (Join-Path $buildRoot 'dist\ForzaAbyssPainter.exe')) -and (Test-Path (Join-Path $desktopBuild 'dist\ForzaAbyssPainter.exe'))) {
    $buildRoot = $desktopBuild
}

$resultsDir = Join-Path $env:USERPROFILE 'Desktop\smoke_results'
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

$py = Join-Path $env:LOCALAPPDATA 'ForzaAbyssPainter\runtime\python311\python.exe'
$marker = Join-Path $env:LOCALAPPDATA 'ForzaAbyssPainter\runtime\installed.json'
$source = Join-Path $buildRoot 'source'
$seed = Join-Path $source 'assets\forza_abyss_painter_logo.png'

$outJson = Join-Path $resultsDir 'smoke_logo_1000.json'
$outStderr = Join-Path $resultsDir 'smoke_logo_1000.stderr.log'
$probeJson = Join-Path $resultsDir 'cursor_gpu_probe.json'
$diagZip = Join-Path $resultsDir 'smoke_diag.zip'
$summary = Join-Path $resultsDir 'smoke_summary.md'

function Write-Section($title) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host " $title"
    Write-Host ("=" * 70)
}

# Initialize the summary report with the timestamp + machine info.
@"
# QUASAR GPU smoke run - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

Machine: $env:COMPUTERNAME
User: $env:USERNAME
PowerShell: $($PSVersionTable.PSVersion)

"@ | Set-Content -Path $summary -Encoding UTF8


# ============================================================ Pre-flight

Write-Section "Pre-flight"

if (-not (Test-Path $py)) {
    Write-Host "FAIL: embedded python not found at $py"
    Write-Host "      Install the GPU runtime first (Tools -> Install GPU runtime in the EXE)."
    "## Pre-flight`n- FAIL: embedded python missing at $py`n" | Add-Content $summary
    exit 2
}
if (-not (Test-Path $marker)) {
    Write-Host "FAIL: runtime marker missing - install not complete"
    "## Pre-flight`n- FAIL: marker missing at $marker`n" | Add-Content $summary
    exit 2
}
$markerJson = Get-Content $marker -Raw | ConvertFrom-Json
Write-Host "Marker: torch=$($markerJson.torch_version)  cuda=$($markerJson.cuda_available)  device=$($markerJson.cuda_device_name)"

if (-not (Test-Path $seed)) {
    Write-Host "FAIL: seed image not found at $seed"
    "## Pre-flight`n- FAIL: seed image missing`n" | Add-Content $summary
    exit 2
}

# Capture nvidia-smi snapshot (pre-run baseline).
$nvSmiPre = nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.used,memory.free `
    --format=csv,noheader,nounits 2>$null
Write-Host "nvidia-smi (pre-run): $nvSmiPre"

@"
## Pre-flight
- Marker torch: $($markerJson.torch_version)
- Marker cuda_available: $($markerJson.cuda_available)
- Marker device: $($markerJson.cuda_device_name)
- nvidia-smi pre-run: $nvSmiPre

"@ | Add-Content $summary


# ============================================================ Step 1: fap-generate

Write-Section "Step 1: fap-generate 1000 shapes on the app logo"

$t0 = Get-Date

# GPU shape-gen: CLI if present, else torch_runner (embedded runtime
# does not copy forza_abyss_painter.cli). Config UTF-8 no BOM.
$runnerCfg = Join-Path $resultsDir 'smoke_runner_config.json'
$genConfig = @{
    image_path = $seed; output_json_path = $outJson
    num_shapes = 1000; max_resolution = 720; random_samples = 8192
    sticker_mode = $false; device = 'cuda'; vram_budget_gib = 8.0
    preset_label = 'smoke-1000'; lock_alpha = $true
} | ConvertTo-Json -Compress
[System.IO.File]::WriteAllText($runnerCfg, $genConfig, (New-Object System.Text.UTF8Encoding $false))

$hasCli = & $py -c "import importlib.util; print(importlib.util.find_spec('forza_abyss_painter.cli') is not None)" 2>$null
if ($hasCli -eq 'True') {
    & $py -m forza_abyss_painter.cli.generate `
        --image $seed --output $outJson `
        --num-shapes 1000 --max-resolution 720 `
        --random-samples 8192 --vram-budget 8.0 `
        --preset-label 'smoke-1000' `
        2> $outStderr
} else {
    Write-Host "NOTE: cli not in embedded runtime; using torch_runner"
    & $py -m forza_abyss_painter.runtime.torch_runner --config $runnerCfg 2> $outStderr
}
$rc = $LASTEXITCODE
$elapsedS = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

Write-Host "Exit code: $rc"
Write-Host "Wall time: ${elapsedS}s"
if (Test-Path $outJson) {
    $outSize = (Get-Item $outJson).Length
    Write-Host "Output size: $outSize bytes"
}

# Capture pre-vs-post nvidia-smi delta as the proxy for peak VRAM.
$nvSmiPost = nvidia-smi --query-gpu=memory.used,memory.free `
    --format=csv,noheader,nounits 2>$null
Write-Host "nvidia-smi (post-run): $nvSmiPost"

# Look for chunked-K activation in the stderr log.
$chunkedLine = (Select-String -Path $outStderr -Pattern 'chunked-K' -SimpleMatch -ErrorAction SilentlyContinue) | Select-Object -First 1
if ($chunkedLine) {
    Write-Host "Chunked-K: $($chunkedLine.Line)"
}

# Validate output JSON structure.
$validation = "(not validated - output missing)"
if (Test-Path $outJson) {
    try {
        $j = Get-Content $outJson -Raw | ConvertFrom-Json
        $shapeCount = $j.shapes.Count
        $validation = "format=$($j.format) version=$($j.version) shape_count=$($j.shape_count) actual_shapes=$shapeCount"
        Write-Host $validation
        if ($j.format -ne 'fd6.shapes') { Write-Host "WARN: format != fd6.shapes" }
        if ($shapeCount -ne 1000) { Write-Host "WARN: shape_count != 1000 (got $shapeCount)" }
    } catch {
        $validation = "PARSE ERROR: $_"
        Write-Host $validation
    }
}

# Stderr tail (last 5 lines) for the summary.
$stderrTail = ""
if (Test-Path $outStderr) {
    $stderrTail = (Get-Content $outStderr -Tail 5) -join "`n"
}

@"
## Step 1: fap-generate 1000 shapes
- Exit code: $rc
- Wall time: ${elapsedS}s
- ms/shape (wall / shape_count): $([math]::Round($elapsedS * 1000.0 / 1000.0, 1))
- Output file size: $outSize bytes
- chunked-K log line: $($chunkedLine.Line)
- nvidia-smi post-run: $nvSmiPost
- Output JSON validation: $validation
- Stderr tail (last 5 lines):
$stderrTail

"@ | Add-Content $summary


# ============================================================ Step 4: probe + diag
# (Step 3 - interactive cancel - left for the human; see docs/CURSOR_GPU_SMOKE_TEST.md)

Write-Section "Step 4a: GPU + runtime probe"

$probeScript = Join-Path $PSScriptRoot 'probe_gpu_and_runtime.py'
if (Test-Path $probeScript) {
    py -3 $probeScript --output $probeJson --skip-kill-ladder
    if (Test-Path $probeJson) {
        $probeSize = (Get-Item $probeJson).Length
        Write-Host "Probe report: $probeJson ($probeSize bytes)"
    }
} else {
    Write-Host "SKIP: probe script not found at $probeScript"
}


Write-Section "Step 4b: fap-diag bundle"

& $py -m forza_abyss_painter.runtime.diagnostics --output $diagZip
if (Test-Path $diagZip) {
    $diagSize = (Get-Item $diagZip).Length
    Write-Host "Diagnostics bundle: $diagZip ($diagSize bytes)"
}

@"
## Step 4: probe + diagnostics
- probe JSON: $probeJson ($probeSize bytes)
- diagnostics bundle: $diagZip ($diagSize bytes)

"@ | Add-Content $summary


# ============================================================ Copy to SMB for upstream

$smbOut = Join-Path $buildRoot 'diagnostics\smoke_results_upstream'
New-Item -ItemType Directory -Force -Path $smbOut | Out-Null
Copy-Item -Path (Join-Path $resultsDir '*') -Destination $smbOut -Force -ErrorAction SilentlyContinue
Write-Host "Copied artifacts to: $smbOut"


# ============================================================ Done

Write-Section "Done"
Write-Host "Results in: $resultsDir"
Write-Host "Summary: $summary"
Write-Host ""
Write-Host "Share these files for Mac-side post-mortem:"
Write-Host "  - $summary"
Write-Host "  - $probeJson"
Write-Host "  - $diagZip"
Write-Host "  - $outStderr"
Write-Host ""
Write-Host "Step 3 (mid-run cancel) is interactive - see docs/CURSOR_GPU_SMOKE_TEST.md"
