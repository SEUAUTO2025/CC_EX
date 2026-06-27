from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dar_td3bc.results.aggregate import aggregate_run_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="results/runs")
    parser.add_argument("--output", default="results/aggregated")
    parser.add_argument("--exclude-method-prefixes", nargs="*", default=[])
    args = parser.parse_args()

    output_dir = aggregate_run_metrics(
        run_root=args.run_root,
        output_dir=args.output,
        exclude_method_prefixes=args.exclude_method_prefixes,
    )
    print(f"Wrote {output_dir / 'final_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
