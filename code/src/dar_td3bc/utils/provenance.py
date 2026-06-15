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
    """Return reported NeoRL-2 Pipeline baselines with explicit provenance.

    These rows are intentionally marked as not rerun and still requiring
    original-paper verification of table number, error-bar type, and seed count.
    """
    checked = retrieved_date or date.today().isoformat()
    rows: list[dict[str, str]] = []
    for method, mean, error in _PIPELINE_BASELINES:
        rows.append(
            {
                "method": method,
                "mean": str(mean),
                "error": str(error),
                "error_type": "unverified_error_bar",
                "metric": "normalized_return",
                "num_seeds": "",
                "source_title": "NeoRL2: A Near Real-World Benchmark for Offline Reinforcement Learning",
                "source_url": "https://arxiv.org/abs/2503.19267",
                "source_table": "TODO_verify_original_table",
                "source_commit": "",
                "retrieved_date": checked,
                "provenance": "reported_not_rerun",
                "notes": (
                    "Seeded from local project plan; verify against the "
                    "original paper before manuscript use."
                ),
            }
        )
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
