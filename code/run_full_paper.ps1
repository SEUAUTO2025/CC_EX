$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $PSScriptRoot

$trainSeeds = @(0, 1, 2, 3, 4)
$evalSeeds = 0..19
$steps = 500000
$maxWorkers = 5
$episodesPerSeed = 1
$device = "cuda"
$runName = "paper_full_parallel"

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

Invoke-Python scripts/write_environment_txt.py `
    --output EXPERIMENT_ENVIRONMENT.txt

Invoke-Python scripts/import_reported_baselines.py `
    --output results/reported/neorl2_pipeline_official.csv

Invoke-Python scripts/acquire_data.py `
    --task Pipeline `
    --output data/raw/neorl2

Invoke-Python scripts/prepare_pipeline_data.py `
    --train data/raw/neorl2/pipeline_train.npz `
    --val data/raw/neorl2/pipeline_val.npz `
    --output-dir data/metadata

Invoke-Python scripts/check_next_operation.py `
    --root . `
    --output NEXT_OPERATIONS_before_training.md

$td3bcTrainArgs = @(
    "scripts/train_td3bc.py",
    "--seeds"
) + $trainSeeds + @(
    "--max-workers", $maxWorkers,
    "--steps", $steps,
    "--run-name", $runName,
    "--device", $device
)
Invoke-Python @td3bcTrainArgs

$darTrainArgs = @(
    "scripts/train_dar_td3bc.py",
    "--seeds"
) + $trainSeeds + @(
    "--max-workers", $maxWorkers,
    "--steps", $steps,
    "--run-name", $runName,
    "--device", $device
)
Invoke-Python @darTrainArgs

foreach ($seed in $trainSeeds) {
    $td3bcEvalArgs = @(
        "scripts/evaluate_rollout.py",
        "--checkpoint", "results/runs/td3bc_mlp/${runName}_seed$seed/checkpoint_best.pt",
        "--method", "td3bc_mlp",
        "--task", "Pipeline",
        "--seeds"
    ) + $evalSeeds + @(
        "--episodes-per-seed", $episodesPerSeed,
        "--device", $device
    )
    Invoke-Python @td3bcEvalArgs

    $darEvalArgs = @(
        "scripts/evaluate_rollout.py",
        "--checkpoint", "results/runs/dar_td3bc/${runName}_seed$seed/checkpoint_best.pt",
        "--method", "dar_td3bc",
        "--task", "Pipeline",
        "--seeds"
    ) + $evalSeeds + @(
        "--episodes-per-seed", $episodesPerSeed,
        "--device", $device
    )
    Invoke-Python @darEvalArgs
}

$ablationArgs = @(
    "scripts/run_ablation.py",
    "--seeds"
) + $trainSeeds + @(
    "--eval-seeds"
) + $evalSeeds + @(
    "--steps", $steps,
    "--episodes-per-seed", $episodesPerSeed,
    "--max-workers", $maxWorkers,
    "--device", $device
)
Invoke-Python @ablationArgs

$robustnessArgs = @(
    "scripts/run_robustness.py",
    "--checkpoint-glob", "results/runs/*/${runName}_seed*/checkpoint_best.pt",
    "--seeds"
) + $evalSeeds + @(
    "--episodes-per-seed", $episodesPerSeed,
    "--device", $device
)
Invoke-Python @robustnessArgs

Invoke-Python scripts/aggregate_results.py `
    --run-root results/runs `
    --output results/aggregated

Invoke-Python scripts/check_next_operation.py `
    --root . `
    --output NEXT_OPERATIONS.md

Write-Host "noop"
