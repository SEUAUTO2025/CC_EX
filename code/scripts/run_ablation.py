from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


ABLATIONS = {
    "dar_td3bc_full": {},
    "dar_td3bc_no_gate": {"model": {"gate_kappa": 0.0}},
    "dar_td3bc_no_prediction": {"loss": {"alpha_pred": 0.0}},
    "dar_td3bc_no_tv": {"loss": {"alpha_tv": 0.0}},
    "dar_td3bc_no_residual_reg": {"loss": {"alpha_res": 0.0}},
    "dar_td3bc_residual_010": {"model": {"residual_scale": 0.10}},
    "dar_td3bc_residual_050": {"model": {"residual_scale": 0.50}},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/model/dar_td3bc.yaml")
    parser.add_argument("--train", default="data/raw/neorl2/pipeline_train.npz")
    parser.add_argument("--val", default="data/raw/neorl2/pipeline_val.npz")
    parser.add_argument("--task", default="Pipeline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-root", default="results/runs")
    parser.add_argument("--config-output", default="results/configs/ablation")
    parser.add_argument("--skip-rollout", action="store_true")
    args = parser.parse_args()

    base_config = _load_yaml(PROJECT_ROOT / args.base_config)
    config_output = PROJECT_ROOT / args.config_output
    config_output.mkdir(parents=True, exist_ok=True)

    for variant, override in ABLATIONS.items():
        config_path = config_output / f"{variant}.yaml"
        config = copy.deepcopy(base_config)
        _deep_update(config, override)
        if args.device is not None:
            config["device"] = args.device
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        for seed in args.seeds:
            run_name = variant
            _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_dar_td3bc.py"),
                    "--config",
                    str(config_path),
                    "--train",
                    args.train,
                    "--val",
                    args.val,
                    "--seed",
                    str(seed),
                    "--steps",
                    str(args.steps),
                    "--device",
                    str(config.get("device", "auto")),
                    "--output-root",
                    args.output_root,
                    "--run-name",
                    run_name,
                ]
            )
            if not args.skip_rollout:
                checkpoint = (
                    PROJECT_ROOT
                    / args.output_root
                    / "dar_td3bc"
                    / f"{run_name}_seed{seed}"
                    / "checkpoint_best.pt"
                )
                _run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "evaluate_rollout.py"),
                        "--checkpoint",
                        str(checkpoint),
                        "--method",
                        variant,
                        "--task",
                        args.task,
                        "--seeds",
                        *(str(value) for value in args.eval_seeds),
                        "--episodes-per-seed",
                        str(args.episodes_per_seed),
                        "--max-steps",
                        str(args.max_steps),
                        "--device",
                        str(config.get("device", "auto")),
                        "--output-dir",
                        str(checkpoint.parent),
                    ]
                )
    return 0


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if data is not None else {}


def _deep_update(target: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict):
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                target[key] = child = {}
            _deep_update(child, value)
        else:
            target[key] = value


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
