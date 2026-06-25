from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dar_td3bc.envs.robustness_wrappers import apply_robustness_wrappers
from dar_td3bc.evaluation.rollout import evaluate_checkpoint_rollout

from evaluate_rollout import _make_neorl2_env_factory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-glob",
        default="results/runs/*/*/checkpoint_best.pt",
    )
    parser.add_argument("--task", default="Pipeline")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--noise-stds", type=float, nargs="+", default=[0.01, 0.03, 0.05])
    parser.add_argument("--delay-steps", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    base_factory = _make_neorl2_env_factory(args.task, args.env_id)
    checkpoints = sorted(PROJECT_ROOT.glob(args.checkpoint_glob))
    if not checkpoints:
        raise SystemExit(f"No checkpoints matched {args.checkpoint_glob}")

    for checkpoint in checkpoints:
        method = _infer_method_from_path(checkpoint)
        for noise_std in args.noise_stds:
            for delay_steps in args.delay_steps:
                condition = f"noise{noise_std:g}_delay{delay_steps}"
                output_name = f"metrics_rollout_robustness_{condition}.csv"

                def factory(noise_std=noise_std, delay_steps=delay_steps):
                    return apply_robustness_wrappers(
                        base_factory(),
                        observation_noise_std=noise_std,
                        observation_delay_steps=delay_steps,
                    )

                output_path = evaluate_checkpoint_rollout(
                    checkpoint_path=checkpoint,
                    env_factory=factory,
                    output_dir=checkpoint.parent,
                    method=method,
                    task=args.task,
                    seeds=args.seeds,
                    episodes_per_seed=args.episodes_per_seed,
                    max_steps=args.max_steps,
                    output_name=output_name,
                    metadata={
                        "condition": condition,
                        "observation_noise_std": noise_std,
                        "observation_delay_steps": delay_steps,
                    },
                    device=args.device,
                )
                print(f"Wrote {output_path}")
    return 0


def _infer_method_from_path(checkpoint: Path) -> str | None:
    known = {"td3bc_mlp", "dar_td3bc"}
    for parent in checkpoint.parents:
        if parent.name in known:
            return parent.name
    return None


if __name__ == "__main__":
    raise SystemExit(main())
