from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dar_td3bc.evaluation.rollout import evaluate_checkpoint_rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--task", default="Pipeline")
    parser.add_argument("--env-id", default=None)
    parser.add_argument(
        "--seeds",
        "--eval-seeds",
        dest="seeds",
        type=int,
        nargs="+",
        default=list(range(20)),
    )
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default="metrics_rollout.csv")
    parser.add_argument("--condition", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint.parent
    output_path = evaluate_checkpoint_rollout(
        checkpoint_path=checkpoint,
        env_factory=_make_neorl2_env_factory(args.task, args.env_id),
        output_dir=output_dir,
        method=args.method,
        task=args.task,
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        max_steps=args.max_steps,
        output_name=args.output_name,
        metadata={"condition": args.condition} if args.condition else None,
        device=args.device,
    )
    print(f"Wrote {output_path}")
    return 0


def _make_neorl2_env_factory(task: str, env_id: str | None) -> Callable[[], object]:
    try:
        import neorl2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "NeoRL-2 is not installed. Install it with: "
            "git clone https://github.com/polixir/NeoRL2.git ../NeoRL2 && "
            "pip install -e ../NeoRL2"
        ) from exc

    try:
        import gymnasium as gym
    except ImportError:
        import gym

    candidates = [env_id] if env_id else [task, f"{task}-v0"]

    def factory() -> object:
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                return gym.make(candidate)
            except Exception as exc:
                last_error = exc
        joined = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(f"Could not create NeoRL-2 environment from: {joined}") from last_error

    return factory


if __name__ == "__main__":
    raise SystemExit(main())
