from __future__ import annotations

from pathlib import Path

import pandas as pd


def aggregate_run_metrics(*, run_root: str | Path, output_dir: str | Path) -> Path:
    run_root = Path(run_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        pd.read_csv(path).assign(source_file=str(path))
        for path in sorted(run_root.glob("**/metrics_eval.csv"))
    ]
    if frames:
        metrics = pd.concat(frames, ignore_index=True)
        summary = _summarize(metrics)
    else:
        metrics = pd.DataFrame(
            columns=[
                "method",
                "metric",
                "value",
                "provenance",
                "checkpoint",
                "dataset",
                "source_file",
            ]
        )
        summary = pd.DataFrame(
            columns=[
                "method",
                "metric",
                "n",
                "mean",
                "std",
                "median",
                "ci95_low",
                "ci95_high",
                "provenance",
            ]
        )

    metrics.to_csv(output_dir / "complete_results.csv", index=False)
    summary.to_csv(output_dir / "final_summary.csv", index=False)
    _write_empty_paired_tests(output_dir / "paired_tests.csv")
    _write_latex_table(summary, output_dir / "main_results.tex")
    return output_dir


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, metric), group in metrics.groupby(["method", "metric"], sort=True):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        n = int(values.shape[0])
        if n == 0:
            continue
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        sem_margin = 1.96 * std / (n**0.5) if n > 1 else 0.0
        rows.append(
            {
                "method": method,
                "metric": metric,
                "n": n,
                "mean": float(values.mean()),
                "std": std,
                "median": float(values.median()),
                "ci95_low": float(values.mean() - sem_margin),
                "ci95_high": float(values.mean() + sem_margin),
                "provenance": ";".join(sorted(set(group["provenance"].astype(str)))),
            }
        )
    return pd.DataFrame(rows)


def _write_empty_paired_tests(path: Path) -> None:
    pd.DataFrame(
        columns=[
            "metric",
            "method_a",
            "method_b",
            "test",
            "statistic",
            "p_value",
            "effect_size",
            "notes",
        ]
    ).to_csv(path, index=False)


def _write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        path.write_text("% No evaluation metrics found.\n", encoding="utf-8")
        return
    columns = ["method", "metric", "n", "mean", "std", "provenance"]
    latex = summary[columns].to_latex(index=False, float_format="%.4f")
    path.write_text(latex, encoding="utf-8")
