from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReadinessItem:
    key: str
    status: str
    message: str
    command: str


def assess_result_readiness(project_root: str | Path) -> list[ReadinessItem]:
    root = Path(project_root)
    items: list[ReadinessItem] = []

    train_npz = root / "data" / "raw" / "neorl2" / "pipeline_train.npz"
    val_npz = root / "data" / "raw" / "neorl2" / "pipeline_val.npz"
    data_ok = train_npz.exists() and val_npz.exists()
    items.append(
        ReadinessItem(
            key="data",
            status="ok" if data_ok else "missing",
            message=(
                "Pipeline train/validation NPZ files are present."
                if data_ok
                else "Pipeline train/validation NPZ files are missing."
            ),
            command=(
                "python scripts/acquire_data.py --task Pipeline "
                "--output data/raw/neorl2"
            ),
        )
    )

    metadata = root / "data" / "metadata" / "pipeline_manifest.json"
    metadata_ok = metadata.exists()
    items.append(
        ReadinessItem(
            key="metadata",
            status="ok" if metadata_ok else "missing",
            message=(
                "Dataset validation metadata exists."
                if metadata_ok
                else "Dataset validation metadata has not been generated."
            ),
            command=(
                "python scripts/prepare_pipeline_data.py "
                "--train data/raw/neorl2/pipeline_train.npz "
                "--val data/raw/neorl2/pipeline_val.npz "
                "--output-dir data/metadata"
            ),
        )
    )

    required_scripts = (
        "train_behavior_policy.py",
        "pretrain_encoder.py",
        "train_td3bc.py",
        "train_dar_td3bc.py",
        "evaluate.py",
        "aggregate_results.py",
    )
    missing_scripts = [
        script for script in required_scripts if not (root / "scripts" / script).exists()
    ]
    scripts_ok = not missing_scripts
    items.append(
        ReadinessItem(
            key="training_scripts",
            status="ok" if scripts_ok else "missing",
            message=(
                "Training, evaluation, and aggregation scripts are present."
                if scripts_ok
                else "Missing runnable scripts: " + ", ".join(missing_scripts)
            ),
            command=(
                "python -m pytest tests -q"
                if scripts_ok
                else (
                    "Implement train_behavior_policy.py, pretrain_encoder.py, "
                    "train_td3bc.py, train_dar_td3bc.py, evaluate.py, and "
                    "aggregate_results.py"
                )
            ),
        )
    )

    final_summary = root / "results" / "aggregated" / "final_summary.csv"
    results_ok = final_summary.exists()
    items.append(
        ReadinessItem(
            key="results",
            status="ok" if results_ok else "missing",
            message=(
                "Aggregated final summary exists."
                if results_ok
                else "No aggregated final summary exists."
            ),
            command="python scripts/aggregate_results.py --output results/aggregated",
        )
    )
    return items


def render_next_operations(project_root: str | Path) -> str:
    items = assess_result_readiness(project_root)
    ready = all(item.status == "ok" for item in items)
    lines = [
        "# DAR-TD3BC Next Operations",
        "",
        (
            "Status: results-ready."
            if ready
            else "Status: Cannot produce final paper results yet."
        ),
        "",
        "## Readiness",
        "",
    ]
    for item in items:
        mark = "OK" if item.status == "ok" else "BLOCKED"
        lines.extend(
            [
                f"- {mark} `{item.key}`: {item.message}",
                f"  Next command/action: `{item.command}`",
            ]
        )
    first_blocker = next((item for item in items if item.status != "ok"), None)
    if first_blocker is not None:
        lines.extend(
            [
                "",
                "## Immediate Next Step",
                "",
                f"`{first_blocker.command}`",
            ]
        )
    return "\n".join(lines)
