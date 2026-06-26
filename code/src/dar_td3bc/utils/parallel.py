from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess


def run_parallel_commands(
    commands: Sequence[Sequence[str]],
    *,
    max_workers: int | None = None,
    label: str = "job",
    cwd: str | Path | None = None,
) -> int:
    if not commands:
        return 0
    workers = max_workers or len(commands)
    if workers <= 0:
        raise ValueError("max_workers must be positive")
    workers = min(workers, len(commands))
    print(f"Starting {len(commands)} {label}(s) with {workers} worker(s).", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_command, list(command), cwd): list(command)
            for command in commands
        }
        first_failure = 0
        for future in as_completed(futures):
            command = futures[future]
            return_code = future.result()
            if return_code != 0 and first_failure == 0:
                first_failure = return_code
            print(
                f"Finished {label}: return_code={return_code} command={' '.join(command)}",
                flush=True,
            )
    return first_failure


def _run_command(command: list[str], cwd: str | Path | None) -> int:
    print("Running: " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    return int(result.returncode)
