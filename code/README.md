# DAR-TD3BC Code Scaffold

This directory contains the research code scaffold for DAR-TD3BC on NeoRL-2
Pipeline. It is intentionally split into tested core utilities and experiment
entry points that require the real NeoRL-2 environment/data.

## What Is Implemented

- Pipeline observation parsing for 52-dimensional observations.
- Episode-safe future-flow targets for horizons such as 1, 5, 10, and 20.
- Normalization utilities fitted only from training arrays.
- Control metrics: RMSE, MAE, IAE, action total variation, action energy, and
  target-switch settling events.
- PyTorch modules: causal TCN delay encoder, behavior policy, residual actor,
  twin critic, multi-horizon prediction head, and critic-disagreement gate.
- TD3+BC helper losses: Polyak target update, masked prediction loss, and
  weighted actor loss components.
- Runnable training entry points for behavior cloning, encoder pretraining,
  TD3BC-MLP, and DAR-TD3BC.
- Offline checkpoint evaluation and result aggregation utilities. These produce
  `offline_validation_npz` metrics and do not claim official NeoRL-2 rollout
  normalized return.
- Provenance utilities for reported-not-rerun official baselines.
- Data acquisition entry point that clearly reports required data fields when
  NeoRL-2 dependencies or data are unavailable.

## Local Smoke Test

```bash
cd code
python -m pytest tests -q
```

The current local test suite does not require NeoRL-2 or Gymnasium.

## Environment Record

The target experiment environment is documented in:

```bash
EXPERIMENT_ENVIRONMENT.txt
```

This is a planning record, not evidence that full experiments were run on this
computer.

Regenerate it with:

```bash
python scripts/write_environment_txt.py --output EXPERIMENT_ENVIRONMENT.txt
```

## Reported Official Baselines

Generate the reported-not-rerun baseline seed CSV:

```bash
python scripts/import_reported_baselines.py \
  --output results/reported/neorl2_pipeline_official.csv
```

Rows are marked `reported_not_rerun` and `unverified_error_bar`. Before any
manuscript use, verify table number, error-bar type, and seed count from the
original NeoRL-2 paper.

## Data Acquisition

When NeoRL-2 and Gymnasium are installed in a suitable Python 3.10 environment:

```bash
python scripts/acquire_data.py --task Pipeline --output data/raw/neorl2
```

If dependencies or official data are unavailable, the script exits with code 2
and prints the exact required fields and acceptable data formats.

## Minimum Runnable Experiment Chain

After `data/raw/neorl2/pipeline_train.npz` and
`data/raw/neorl2/pipeline_val.npz` exist:

```bash
python scripts/prepare_pipeline_data.py \
  --train data/raw/neorl2/pipeline_train.npz \
  --val data/raw/neorl2/pipeline_val.npz \
  --output-dir data/metadata

python scripts/train_behavior_policy.py --seed 0 --steps 1000
python scripts/pretrain_encoder.py --seed 0 --steps 1000
python scripts/train_td3bc.py --seed 0 --steps 1000
python scripts/train_dar_td3bc.py --seed 0 --steps 1000
```

Evaluate a produced checkpoint on the validation NPZ:

```bash
python scripts/evaluate.py \
  --checkpoint results/runs/td3bc_mlp/<run_dir>/checkpoint_best.pt \
  --val data/raw/neorl2/pipeline_val.npz \
  --method td3bc_mlp
```

Aggregate all runs that contain `metrics_eval.csv`:

```bash
python scripts/aggregate_results.py \
  --run-root results/runs \
  --output results/aggregated
```

For full paper results, replace the 1000-step smoke settings with the configured
multi-seed runs and add official NeoRL-2 rollout evaluation once the local
NeoRL-2 environment is available.
