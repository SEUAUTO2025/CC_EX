from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dar_td3bc.training.supervised import pretrain_encoder
from dar_td3bc.utils.run import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/dar_td3bc.yaml")
    parser.add_argument("--train", default="data/raw/neorl2/pipeline_train.npz")
    parser.add_argument("--val", default="data/raw/neorl2/pipeline_val.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default="results/runs")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.device is not None:
        config["device"] = args.device
    print(
        f"Starting encoder pretraining: seed={args.seed}, steps={args.steps}, "
        f"device={config.get('device', 'cpu')}"
    )
    run_dir = pretrain_encoder(
        train_path=args.train,
        val_path=args.val,
        config=config,
        seed=args.seed,
        steps=args.steps,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    print(f"Finished encoder pretraining. Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
