from __future__ import annotations

import csv
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

    experiment_scripts = (
        "evaluate_rollout.py",
        "run_ablation.py",
        "run_robustness.py",
    )
    missing_experiment_scripts = [
        script
        for script in experiment_scripts
        if not (root / "scripts" / script).exists()
    ]
    experiment_scripts_ok = not missing_experiment_scripts
    items.append(
        ReadinessItem(
            key="experiment_scripts",
            status="ok" if experiment_scripts_ok else "missing",
            message=(
                "Rollout, ablation, and robustness scripts are present."
                if experiment_scripts_ok
                else "Missing experiment scripts: "
                + ", ".join(missing_experiment_scripts)
            ),
            command=(
                "python -m pytest tests -q"
                if experiment_scripts_ok
                else "Implement evaluate_rollout.py, run_ablation.py, and run_robustness.py"
            ),
        )
    )

    results_status, results_message = _paper_grade_results_status(root)
    items.append(
        ReadinessItem(
            key="paper_grade_results",
            status=results_status,
            message=results_message,
            command=(
                "python scripts/evaluate_rollout.py --checkpoint "
                "results/runs/<method>/<run_dir>/checkpoint_best.pt "
                "--method <method> --task Pipeline "
                "--eval-seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 "
                "--episodes-per-seed 1 && "
                "python scripts/aggregate_results.py --run-root results/runs "
                "--output results/aggregated"
            ),
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


def _paper_grade_results_status(root: Path) -> tuple[str, str]:
    complete = root / "results" / "aggregated" / "complete_results.csv"
    if not complete.exists():
        return "missing", "No aggregated complete_results.csv exists."

    rows = _read_result_rows(complete)
    if not rows:
        return "missing", "Aggregated results are empty."

    required_methods = {"td3bc_mlp", "dar_td3bc"}
    required_seeds = {0, 1, 2, 3, 4}
    rollout_rows = [
        row
        for row in rows
        if row.get("metric") == "normalized_score"
        and row.get("provenance") == "rollout_neorl2"
    ]
    if not rollout_rows:
        return (
            "missing",
            "No rollout_neorl2 normalized_score rows are present; offline diagnostics are not paper-grade results.",
        )

    present_methods = {row.get("method", "") for row in rollout_rows}
    missing_methods = sorted(required_methods - present_methods)
    if missing_methods:
        return (
            "missing",
            "Missing rollout normalized_score rows for methods: "
            + ", ".join(missing_methods),
        )

    incomplete: list[str] = []
    for method in sorted(required_methods):
        method_rows = [row for row in rollout_rows if row.get("method") == method]
        seeds = {_parse_int(row.get("seed")) for row in method_rows}
        missing_seeds = required_seeds - {seed for seed in seeds if seed is not None}
        if missing_seeds:
            incomplete.append(
                f"{method} missing seeds {', '.join(str(seed) for seed in sorted(missing_seeds))}"
            )
            continue
        for seed in sorted(required_seeds):
            rollout_episodes = {
                (_parse_int(row.get("eval_seed")), _parse_int(row.get("episode")))
                for row in method_rows
                if _parse_int(row.get("seed")) == seed
            }
            rollout_episodes = {
                item
                for item in rollout_episodes
                if item[0] is not None and item[1] is not None
            }
            if len(rollout_episodes) < 20:
                incomplete.append(
                    f"{method} seed {seed} has {len(rollout_episodes)} normalized_score episodes, expected 20"
                )
                break
    if incomplete:
        return "missing", "; ".join(incomplete)
    return (
        "ok",
        "Paper-grade rollout normalized_score rows exist for td3bc_mlp and dar_td3bc across seeds 0-4 with at least 20 episodes per seed.",
    )


def _read_result_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_int(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None
