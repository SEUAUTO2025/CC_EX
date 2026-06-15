from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dar_td3bc.utils.provenance import write_environment_txt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="EXPERIMENT_ENVIRONMENT.txt",
        help="Text file describing the target experiment environment.",
    )
    args = parser.parse_args()
    path = write_environment_txt(Path(args.output))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
