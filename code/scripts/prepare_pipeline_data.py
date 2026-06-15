from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dar_td3bc.data.pipeline_dataset import PipelineArrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument(
        "--output-dir", default="data/metadata", help="Metadata output directory."
    )
    args = parser.parse_args()

    train = PipelineArrays.from_npz(args.train)
    val = PipelineArrays.from_npz(args.val)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "train_path": str(Path(args.train)),
        "val_path": str(Path(args.val)),
        "train": train.validation_summary(),
        "val": val.validation_summary(),
    }
    (output_dir / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "pipeline_validation_report.md").write_text(
        _render_report(manifest), encoding="utf-8"
    )
    print(f"Wrote metadata to {output_dir}")
    return 0


def _render_report(manifest: dict) -> str:
    return (
        "# Pipeline Dataset Validation Report\n\n"
        f"- Train path: `{manifest['train_path']}`\n"
        f"- Validation path: `{manifest['val_path']}`\n"
        f"- Train transitions: {manifest['train']['num_transitions']}\n"
        f"- Validation transitions: {manifest['val']['num_transitions']}\n"
        f"- Observation shape: {manifest['train']['observation_shape']}\n"
        f"- Action shape: {manifest['train']['action_shape']}\n"
        f"- Train episode ends: {manifest['train']['episode_end_count']}\n"
        f"- Validation episode ends: {manifest['val']['episode_end_count']}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
