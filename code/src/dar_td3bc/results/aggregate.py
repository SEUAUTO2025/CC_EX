from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_run_metrics(*, run_root: str | Path, output_dir: str | Path) -> Path:
    run_root = Path(run_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _read_metric_frames(run_root)
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
                "task",
                "seed",
                "episode",
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
    _write_paired_tests(metrics, output_dir / "paired_tests.csv")
    _write_latex_table(summary, output_dir / "main_results.tex")
    return output_dir


def _read_metric_frames(run_root: Path) -> list[pd.DataFrame]:
    frames = []
    for pattern in ("**/metrics_eval.csv", "**/metrics_rollout*.csv"):
        for path in sorted(run_root.glob(pattern)):
            frame = pd.read_csv(path)
            frame["source_file"] = str(path)
            frames.append(frame)
    return frames


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


def _write_paired_tests(metrics: pd.DataFrame, path: Path) -> None:
    columns = [
        "metric",
        "method_a",
        "method_b",
        "test",
        "statistic",
        "p_value",
        "effect_size",
        "n_pairs",
        "notes",
    ]
    if metrics.empty or "seed" not in metrics.columns:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return

    if "provenance" not in metrics.columns:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    rollout = metrics[metrics["provenance"] == "rollout_neorl2"].copy()
    if rollout.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return

    rollout["value"] = pd.to_numeric(rollout["value"], errors="coerce")
    rollout["seed"] = pd.to_numeric(rollout["seed"], errors="coerce")
    rows = []
    for metric, group in rollout.dropna(subset=["value", "seed"]).groupby("metric"):
        per_seed = (
            group.groupby(["method", "seed"], as_index=False)["value"]
            .mean()
            .pivot(index="seed", columns="method", values="value")
        )
        if {"dar_td3bc", "td3bc_mlp"} - set(per_seed.columns):
            continue
        paired = per_seed[["dar_td3bc", "td3bc_mlp"]].dropna()
        n_pairs = int(paired.shape[0])
        if n_pairs < 2:
            rows.append(
                {
                    "metric": metric,
                    "method_a": "dar_td3bc",
                    "method_b": "td3bc_mlp",
                    "test": "paired_t",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "effect_size": np.nan,
                    "n_pairs": n_pairs,
                    "notes": "insufficient paired seeds",
                }
            )
            continue
        diff = paired["dar_td3bc"].to_numpy() - paired["td3bc_mlp"].to_numpy()
        statistic, p_value = _paired_t(diff)
        rows.append(
            {
                "metric": metric,
                "method_a": "dar_td3bc",
                "method_b": "td3bc_mlp",
                "test": "paired_t",
                "statistic": statistic,
                "p_value": p_value,
                "effect_size": _cohen_dz(diff),
                "n_pairs": n_pairs,
                "notes": "paired by random seed after averaging episodes",
            }
        )
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _paired_t(diff: np.ndarray) -> tuple[float, float]:
    std = float(np.std(diff, ddof=1))
    if std == 0.0:
        return float("inf") if float(np.mean(diff)) != 0.0 else 0.0, 0.0
    statistic = float(np.mean(diff) / (std / np.sqrt(diff.size)))
    try:
        from scipy import stats

        p_value = float(stats.ttest_1samp(diff, popmean=0.0).pvalue)
    except Exception:
        p_value = float("nan")
    return statistic, p_value


def _cohen_dz(diff: np.ndarray) -> float:
    std = float(np.std(diff, ddof=1))
    if std == 0.0:
        return float("inf") if float(np.mean(diff)) != 0.0 else 0.0
    return float(np.mean(diff) / std)


def _write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        path.write_text("% No evaluation metrics found.\n", encoding="utf-8")
        return
    columns = ["method", "metric", "n", "mean", "std", "provenance"]
    latex = summary[columns].to_latex(index=False, float_format="%.4f")
    path.write_text(latex, encoding="utf-8")
