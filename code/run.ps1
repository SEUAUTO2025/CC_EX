$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $PSScriptRoot

$trainSeeds = @(0, 1, 2, 3, 4)
$evalSeeds = 0..19
$episodesPerSeed = 1
$device = "cuda"

# Main DAR-TD3BC run completed by resume training.
$darRunName = "paper_full_parallel"

# TD3BC was trained earlier. Keep both common names here because one previous
# script used a space in the run name.
$td3bcRunNameCandidates = @("paper_full_parallel", "paper_full parallel")

function Invoke-Python {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $Arguments
    )

    Write-Host ""
    Write-Host "python $($Arguments -join ' ')"
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "python $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Require-Path {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path not found: $Path"
    }
}

function Find-FirstCheckpointGlob {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Method,

        [Parameter(Mandatory = $true)]
        [string[]] $RunNames
    )

    foreach ($runName in $RunNames) {
        $glob = "results/runs/$Method/${runName}_seed*/checkpoint_best.pt"
        $matches = @(Get-ChildItem -Path $glob -ErrorAction SilentlyContinue)
        if ($matches.Count -gt 0) {
            Write-Host "Using $Method checkpoint glob: $glob"
            return $glob
        }
    }

    throw "No checkpoint_best.pt found for $Method with run names: $($RunNames -join ', ')"
}

Invoke-Python scripts/write_environment_txt.py `
    --output EXPERIMENT_ENVIRONMENT.txt

Invoke-Python scripts/import_reported_baselines.py `
    --output results/reported/neorl2_pipeline_official.csv

Invoke-Python scripts/check_next_operation.py `
    --root . `
    --output NEXT_OPERATIONS_before_postprocess.md

# 1. Evaluate main DAR-TD3BC rollout. TD3BC rollout is assumed complete.
foreach ($seed in $trainSeeds) {
    $checkpoint = "results/runs/dar_td3bc/${darRunName}_seed$seed/checkpoint_best.pt"
    Require-Path $checkpoint

    $darEvalArgs = @(
        "scripts/evaluate_rollout.py",
        "--checkpoint", $checkpoint,
        "--method", "dar_td3bc",
        "--task", "Pipeline",
        "--seeds"
    ) + $evalSeeds + @(
        "--episodes-per-seed", $episodesPerSeed,
        "--device", $device
    )
    Invoke-Python @darEvalArgs
}

# 2. Run robustness for the main TD3BC and DAR-TD3BC checkpoints.
$td3bcCheckpointGlob = Find-FirstCheckpointGlob `
    -Method "td3bc_mlp" `
    -RunNames $td3bcRunNameCandidates

Invoke-Python scripts/run_robustness.py `
    --checkpoint-glob $td3bcCheckpointGlob `
    --seeds $evalSeeds `
    --episodes-per-seed $episodesPerSeed `
    --device $device

$darCheckpointGlob = "results/runs/dar_td3bc/${darRunName}_seed*/checkpoint_best.pt"
Invoke-Python scripts/run_robustness.py `
    --checkpoint-glob $darCheckpointGlob `
    --seeds $evalSeeds `
    --episodes-per-seed $episodesPerSeed `
    --device $device

# 3. Aggregate every rollout and robustness metric currently under results/runs.
Invoke-Python scripts/aggregate_results.py `
    --run-root results/runs `
    --output results/aggregated `
    --exclude-method-prefixes dar_td3bc_no_ dar_td3bc_residual_ dar_td3bc_full

Invoke-Python scripts/check_next_operation.py `
    --root . `
    --output NEXT_OPERATIONS.md

Write-Host "noop"
