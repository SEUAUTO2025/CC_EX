from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dar_td3bc.data.acquisition import missing_pipeline_data_request
from dar_td3bc.data.acquisition import get_pipeline_dataset, save_dataset_npz


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Pipeline")
    parser.add_argument("--output", default="data/raw/neorl2")
    args = parser.parse_args()

    if args.task != "Pipeline":
        print("Only Pipeline is supported by this project scaffold.", file=sys.stderr)
        return 2

    try:
        train_data, val_data = get_pipeline_dataset(args.task)
    except RuntimeError as exc:
        print(exc)
        return 2

    output_dir = Path(args.output)
    save_dataset_npz(train_data, output_dir / "pipeline_train.npz")
    save_dataset_npz(val_data, output_dir / "pipeline_val.npz")
    print(f"Wrote Pipeline data to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
