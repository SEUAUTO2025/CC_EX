# DAR-TD3BC Next Operations

Status: Cannot produce final paper results yet.

## Readiness

- OK `data`: Pipeline train/validation NPZ files are present.
  Next command/action: `python scripts/acquire_data.py --task Pipeline --output data/raw/neorl2`
- OK `metadata`: Dataset validation metadata exists.
  Next command/action: `python scripts/prepare_pipeline_data.py --train data/raw/neorl2/pipeline_train.npz --val data/raw/neorl2/pipeline_val.npz --output-dir data/metadata`
- OK `training_scripts`: Training, evaluation, and aggregation scripts are present.
  Next command/action: `python -m pytest tests -q`
- OK `experiment_scripts`: Rollout, ablation, and robustness scripts are present.
  Next command/action: `python -m pytest tests -q`
- BLOCKED `paper_grade_results`: No rollout_neorl2 normalized_score rows are present; offline diagnostics are not paper-grade results.
  Next command/action: `python scripts/evaluate_rollout.py --checkpoint results/runs/<method>/<run_dir>/checkpoint_best.pt --method <method> --task Pipeline --eval-seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --episodes-per-seed 1 && python scripts/aggregate_results.py --run-root results/runs --output results/aggregated`

## Immediate Next Step

`python scripts/evaluate_rollout.py --checkpoint results/runs/<method>/<run_dir>/checkpoint_best.pt --method <method> --task Pipeline --eval-seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --episodes-per-seed 1 && python scripts/aggregate_results.py --run-root results/runs --output results/aggregated`
