from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def missing_pipeline_data_request() -> str:
    return (
        "当前代码和环境已准备完成，但 NeoRL-2 Pipeline 数据未能从官方接口下载。\n"
        "请提供以下任一种：\n\n"
        "A. Pipeline 训练集和验证集文件（Parquet、NPZ、PKL 均可）；\n"
        "B. 已下载的 Hugging Face NeoRL-2 缓存目录；\n"
        "C. 可联网下载的运行环境或代理方式；\n"
        "D. 你本地执行 env.get_dataset() 后导出的数据目录。\n\n"
        "数据至少需要以下字段：\n"
        "observations、actions、rewards、next_observations、terminals/dones。\n\n"
        "预期维度：\n"
        "observations: [N, 52]\n"
        "actions: [N, 1]\n"
        "rewards: [N] 或 [N, 1]\n"
        "next_observations: [N, 52]\n"
        "terminals/dones: [N] 或 [N, 1]\n"
    )


def get_pipeline_dataset(task: str = "Pipeline") -> tuple[Any, Any]:
    try:
        import neorl2  # noqa: F401
        import gymnasium as gym
    except ModuleNotFoundError as exc:
        raise RuntimeError(missing_pipeline_data_request()) from exc

    env = gym.make(task)
    return env.get_dataset()


def save_dataset_npz(split: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    field_map = {
        "obs": "observations",
        "observation": "observations",
        "observations": "observations",
        "next_obs": "next_observations",
        "next_observation": "next_observations",
        "next_observations": "next_observations",
        "action": "actions",
        "actions": "actions",
        "reward": "rewards",
        "rewards": "rewards",
        "done": "dones",
        "dones": "dones",
        "terminal": "terminals",
        "terminals": "terminals",
        "timeout": "timeouts",
        "timeouts": "timeouts",
        "truncated": "truncated",
    }
    arrays: dict[str, np.ndarray] = {}
    for key, value in split.items():
        canonical = field_map.get(key)
        if canonical is not None:
            arrays[canonical] = np.asarray(value)
    if not arrays:
        raise ValueError("dataset split did not contain supported array fields")
    np.savez_compressed(output, **arrays)
    return output
