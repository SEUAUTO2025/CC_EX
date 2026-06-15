# DAR-TD3BC Next Operations

Status: Cannot produce final paper results yet.

## Readiness

- BLOCKED `data`: Pipeline train/validation NPZ files are missing.
  Next command/action: `python scripts/acquire_data.py --task Pipeline --output data/raw/neorl2`
- BLOCKED `metadata`: Dataset validation metadata has not been generated.
  Next command/action: `python scripts/prepare_pipeline_data.py --train data/raw/neorl2/pipeline_train.npz --val data/raw/neorl2/pipeline_val.npz --output-dir data/metadata`
- OK `training_scripts`: Training, evaluation, and aggregation scripts are present.
  Next command/action: `python -m pytest tests -q`
- BLOCKED `results`: No aggregated final summary exists.
  Next command/action: `python scripts/aggregate_results.py --output results/aggregated`

## Immediate Next Step

`python scripts/acquire_data.py --task Pipeline --output data/raw/neorl2`
