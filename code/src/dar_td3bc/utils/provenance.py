from __future__ import annotations

import csv
import platform
import sys
from datetime import date
from pathlib import Path
from typing import Iterable


BASELINE_COLUMNS = (
    "method",
    "mean",
    "error",
    "error_type",
    "metric",
    "num_seeds",
    "source_title",
    "source_url",
    "source_table",
    "source_commit",
    "retrieved_date",
    "provenance",
    "notes",
)


_PIPELINE_BASELINES = (
    ("Data policy", 69.25, ""),
    ("BC", 68.61, 13.42),
    ("CQL", 81.08, 8.25),
    ("EDAC", 72.88, 4.64),
    ("MCQ", 49.70, 7.39),
    ("TD3+BC", 81.95, 7.46),
    ("MOPO", -26.33, 92.65),
    ("COMBO", 55.50, 4.28),
    ("RAMBO", 24.06, 74.39),
    ("MOBILE", 65.51, 4.05),
)


def official_pipeline_baselines(
    retrieved_date: str | None = None,
) -> list[dict[str, str]]:
    """Return official NeoRL-2 Pipeline baselines with explicit provenance."""
    checked = retrieved_date or date.today().isoformat()
    rows: list[dict[str, str]] = []
    for method, mean, error in _PIPELINE_BASELINES:
        rows.append(
            {
                "method": method,
                "mean": str(mean),
                "error": str(error),
                "error_type": "standard_error" if error != "" else "",
                "metric": "normalized_return",
                "num_seeds": "3" if error != "" else "",
                "source_title": (
                    "NeoRL-2: Near Real-World Benchmarks for Offline "
                    "Reinforcement Learning with Extended Realistic Scenarios"
                ),
                "source_url": "https://arxiv.org/abs/2503.19267",
                "source_table": "Table 2",
                "source_commit": "polixir/NeoRL2 main benchmark/task_score.csv",
                "retrieved_date": checked,
                "provenance": "reported_not_rerun",
                "notes": (
                    "Official NeoRL-2 reported Pipeline normalized score; "
                    "error column is standard error over three random seeds."
                ),
            }
        )
    return rows


def reported_pipeline_followups(
    retrieved_date: str | None = None,
) -> list[dict[str, str]]:
    """Return later reported Pipeline scores that use NeoRL-2 normalized score."""
    checked = retrieved_date or date.today().isoformat()
    rows = [
        {
            "method": "PIQL",
            "mean": "89.3",
            "error": "14.8",
            "error_type": "reported_dispersion",
            "metric": "normalized_return",
            "num_seeds": "",
            "source_title": (
                "PIQL: Projective Implicit Q-Learning with Support Constraint "
                "for Offline Reinforcement Learning"
            ),
            "source_url": "https://arxiv.org/abs/2501.08907",
            "source_table": "Table 3",
            "source_commit": "",
            "retrieved_date": checked,
            "provenance": "reported_not_rerun",
            "notes": "NeoRL2 Pipeline normalized score reported by follow-up paper.",
        },
        {
            "method": "ACPO",
            "mean": "95.2",
            "error": "4.62",
            "error_type": "standard_deviation",
            "metric": "normalized_return",
            "num_seeds": "3",
            "source_title": (
                "Automatic Constraint Policy Optimization based on Continuous "
                "Constraint Interpolation Framework for Offline Reinforcement Learning"
            ),
            "source_url": "https://arxiv.org/abs/2601.23010",
            "source_table": "Table 3",
            "source_commit": "",
            "retrieved_date": checked,
            "provenance": "reported_not_rerun",
            "notes": "NeoRL2 Pipeline normalized score reported by follow-up paper.",
        },
        {
            "method": "TD3PA",
            "mean": "82.31",
            "error": "",
            "error_type": "",
            "metric": "normalized_return",
            "num_seeds": "3",
            "source_title": "Pessimistic Auxiliary Policy for Offline Reinforcement Learning",
            "source_url": "https://arxiv.org/abs/2602.23974",
            "source_table": "Table II",
            "source_commit": "",
            "retrieved_date": checked,
            "provenance": "reported_not_rerun",
            "notes": "NeoRL-2 Pipeline normalized score reported by follow-up paper.",
        },
    ]
    return rows


def write_reported_baselines_csv(
    path: str | Path, rows: Iterable[dict[str, str]] | None = None
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows if rows is not None else official_pipeline_baselines())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASELINE_COLUMNS)
        writer.writeheader()
        writer.writerows(materialized)
    return output


def render_environment_txt() -> str:
    """Render target experiment environment notes without claiming local runs."""
    return "\n".join(
        [
            "DAR-TD3BC target experiment environment",
            "=====================================",
            "",
            "Status: planning record only; real NeoRL-2 experiments do not need to",
            "run on this computer.",
            "",
            "Recommended runtime:",
            "- Python: 3.10",
            "- PyTorch: CUDA-enabled build matching the selected GPU driver",
            "- Gymnasium / NeoRL-2: use versions required by the checked NeoRL-2 commit",
            "- OS: Linux is preferred for long multi-seed training runs",
            "- Seeds: 0, 1, 2, 3, 4 for paper-complete experiments",
            "- Evaluation: at least 20 episodes per checkpoint",
            "",
            "Current authoring machine snapshot:",
            f"- Python executable: {sys.executable}",
            f"- Python version: {sys.version.split()[0]}",
            f"- Platform: {platform.platform()}",
            "",
            "Required provenance to fill before submission:",
            "- NeoRL-2 commit SHA",
            "- Dataset SHA256 or Hugging Face revision",
            "- GPU model and CUDA version",
            "- Training command line for every run",
            "- Random seed and resolved config for every run",
        ]
    )


def write_environment_txt(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_environment_txt() + "\n", encoding="utf-8")
    return output
