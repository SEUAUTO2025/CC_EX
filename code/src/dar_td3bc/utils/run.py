from __future__ import annotations

from pathlib import Path
from typing import Any

import csv
from datetime import datetime

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if data is not None else {}


def make_run_dir(
    root: str | Path,
    method: str,
    seed: int,
    *,
    run_name: str | None = None,
) -> Path:
    root_path = Path(root)
    name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_path / method / f"{name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
