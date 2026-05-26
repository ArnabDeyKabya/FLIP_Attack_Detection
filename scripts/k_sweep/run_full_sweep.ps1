# =============================================================================
# Deep k-NN defense — full thesis sweep (PowerShell)
# =============================================================================
# Mirror of run_full_sweep.sh for native Windows PowerShell.
#
# Usage (defaults):
#   pwsh -File scripts\k_sweep\run_full_sweep.ps1
#
# Common overrides:
#   pwsh -File scripts\k_sweep\run_full_sweep.ps1 -KValues 5,20,100 -Force
#   pwsh -File scripts\k_sweep\run_full_sweep.ps1 -SkipTrain
#   pwsh -File scripts\k_sweep\run_full_sweep.ps1 -Modes "remove"
# =============================================================================

[CmdletBinding()]
param(
    [int[]]   $KValues       = @(1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000),
    [string[]]$Modes         = @("none", "remove"),
    [string]  $Dataset       = "cifar",
    [string]  $UserModel     = "r32p",
    [string]  $Trainer       = "sgd",
    [string]  $Poisoner      = "1xs",
    [int]     $Budget        = 1500,
    [int]     $SourceLabel   = 9,
    [int]     $TargetLabel   = 4,
    [string]  $Encoder       = "dinov2_vits14",
    [string]  $Scoring       = "disagreement",
    [string]  $RemovalCount  = "auto",
    [double]  $Threshold     = 0.5,
    [int]     $Seed          = 0,
    [string]  $NamePrefix    = "",
    [string]  $Python        = "python",
    [switch]  $SkipTrain,
    [switch]  $SkipPlots,
    [switch]  $SkipTables,
    [switch]  $SkipAggregate,
    [switch]  $SkipNone,
    [switch]  $Force
)

$ErrorActionPreference = "Stop"

if ($SkipNone) { $Modes = @("remove") }

if ([string]::IsNullOrWhiteSpace($NamePrefix)) {
    $NamePrefix = "knn_defense_${Dataset}_${Poisoner}_${Budget}"
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$ReportDir = Join-Path "experiments" "_report_${NamePrefix}"
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

$LogFile = Join-Path $ReportDir "sweep.log"
Start-Transcript -Path $LogFile -Append | Out-Null

Write-Host "============================================================"
Write-Host "Deep k-NN defense sweep — starting"
Write-Host "  date           : $(Get-Date)"
Write-Host "  project root   : $Root"
Write-Host "  name prefix    : $NamePrefix"
Write-Host "  report dir     : $ReportDir"
Write-Host "  k values       : $($KValues -join ', ')"
Write-Host "  modes          : $($Modes -join ', ')"
Write-Host "  dataset        : $Dataset"
Write-Host "  user model     : $UserModel"
Write-Host "  trainer        : $Trainer"
Write-Host "  poisoner       : $Poisoner"
Write-Host "  budget         : $Budget"
Write-Host "  encoder        : $Encoder"
Write-Host "  scoring        : $Scoring"
Write-Host "  threshold      : $Threshold"
Write-Host "  seed           : $Seed"
Write-Host "  skip_train     : $SkipTrain"
Write-Host "  skip_plots     : $SkipPlots"
Write-Host "  skip_tables    : $SkipTables"
Write-Host "  force          : $Force"
Write-Host "============================================================"

function Invoke-Step {
    param([string]$Label, [scriptblock]$Body)
    Write-Host ""
    Write-Host "[$Label]"
    & $Body
    if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

# ---------------- 1. generate configs ----------------------------------------
Invoke-Step "1/6 generate configs" {
    & $Python "scripts/k_sweep/generate_config.py" `
        --k-values @KValues `
        --modes @Modes `
        --dataset $Dataset `
        --user-model $UserModel `
        --trainer $Trainer `
        --poisoner $Poisoner `
        --budget $Budget `
        --source-label $SourceLabel `
        --target-label $TargetLabel `
        --encoder $Encoder `
        --scoring $Scoring `
        --removal-count $RemovalCount `
        --threshold $Threshold `
        --seed $Seed `
        --name-prefix $NamePrefix
}

# ---------------- 2. run experiments -----------------------------------------
if ($SkipTrain) {
    Write-Host "[2/6] --SkipTrain set; not running run_experiment.py."
} else {
    Write-Host ""
    Write-Host "[2/6] Running experiments ..."
    foreach ($k in $KValues) {
        foreach ($mode in $Modes) {
            $expName = "${NamePrefix}_${mode}_k${k}"
            $summary = "experiments/${expName}/summary.json"
            if ((Test-Path $summary) -and -not $Force) {
                Write-Host "  [skip] $expName (summary.json exists; use -Force to override)"
                continue
            }
            Write-Host "  [run ] $expName"
            & $Python "run_experiment.py" $expName
            if ($LASTEXITCODE -ne 0) { throw "run_experiment.py failed on $expName" }
        }
    }
}

# ---------------- 3. aggregate -----------------------------------------------
if ($SkipAggregate) {
    Write-Host "[3/6] --SkipAggregate set."
} else {
    Invoke-Step "3/6 aggregate" {
        & $Python "scripts/k_sweep/aggregate_results.py" `
            --name-prefix $NamePrefix `
            --k-values @KValues `
            --modes @Modes `
            --report-dir $ReportDir
    }
}

# ---------------- 4-7. plots -------------------------------------------------
if ($SkipPlots) {
    Write-Host "[4-7/6] --SkipPlots set."
} else {
    Invoke-Step "4/6 detection plots" {
        & $Python "scripts/k_sweep/plot_detection.py" `
            --name-prefix $NamePrefix `
            --k-values @KValues `
            --modes @Modes `
            --report-dir $ReportDir `
            --threshold $Threshold
    }
    Invoke-Step "5/6 training plots" {
        & $Python "scripts/k_sweep/plot_training.py" `
            --name-prefix $NamePrefix `
            --k-values @KValues `
            --modes @Modes `
            --report-dir $ReportDir
    }
    Invoke-Step "6/6 feature plots" {
        & $Python "scripts/k_sweep/plot_features.py" `
            --name-prefix $NamePrefix `
            --k-values @KValues `
            --modes @Modes `
            --dataset $Dataset `
            --encoder $Encoder `
            --report-dir $ReportDir
    }
    Invoke-Step "6/6 sample-grid plots" {
        & $Python "scripts/k_sweep/plot_samples.py" `
            --name-prefix $NamePrefix `
            --k-values @KValues `
            --dataset $Dataset `
            --poisoner $Poisoner `
            --target-label $TargetLabel `
            --report-dir $ReportDir
    }
}

# ---------------- tables -----------------------------------------------------
if ($SkipTables) {
    Write-Host "[tables] --SkipTables set."
} else {
    Invoke-Step "tables" {
        & $Python "scripts/k_sweep/make_tables.py" `
            --name-prefix $NamePrefix `
            --report-dir $ReportDir
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Deep k-NN defense sweep — complete"
Write-Host "  report dir : $ReportDir"
Write-Host "  log        : $LogFile"
Write-Host "============================================================"

Stop-Transcript | Out-Null
