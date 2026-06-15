from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dar_td3bc.evaluation.offline import evaluate_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val", default="data/raw/neorl2/pipeline_val.npz")
    parser.add_argument("--method", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint.parent
    output_path = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        val_path=args.val,
        output_dir=output_dir,
        method=args.method,
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
