from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PipelineArrays:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> "PipelineArrays":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as data:
            required = {
                "observations",
                "actions",
                "rewards",
                "next_observations",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"Missing required dataset fields: {missing}")
            observations = np.asarray(data["observations"], dtype=np.float32)
            actions = _as_2d(np.asarray(data["actions"], dtype=np.float32))
            rewards = _as_2d(np.asarray(data["rewards"], dtype=np.float32))
            next_observations = np.asarray(
                data["next_observations"], dtype=np.float32
            )
            dones = _load_optional_bool(
                data, ("terminals", "dones"), observations.shape[0]
            )
            timeouts = _load_optional_bool(
                data, ("timeouts", "truncated"), observations.shape[0]
            )

        arrays = cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            timeouts=timeouts,
        )
        arrays._validate()
        return arrays

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    def validation_summary(self) -> dict[str, Any]:
        return {
            "num_transitions": self.size,
            "observation_shape": list(self.observations.shape),
            "action_shape": list(self.actions.shape),
            "reward_shape": list(self.rewards.shape),
            "next_observation_shape": list(self.next_observations.shape),
            "reward_min": float(np.min(self.rewards)),
            "reward_max": float(np.max(self.rewards)),
            "action_min": float(np.min(self.actions)),
            "action_max": float(np.max(self.actions)),
            "episode_end_count": int(np.logical_or(self.dones, self.timeouts).sum()),
            "has_nan": bool(
                np.isnan(self.observations).any()
                or np.isnan(self.actions).any()
                or np.isnan(self.rewards).any()
                or np.isnan(self.next_observations).any()
            ),
        }

    def _validate(self) -> None:
        n = self.observations.shape[0]
        if self.observations.ndim != 2 or self.observations.shape[1] != 52:
            raise ValueError(
                "observations must have shape [N, 52], got "
                f"{self.observations.shape}"
            )
        if self.next_observations.shape != self.observations.shape:
            raise ValueError(
                "next_observations must match observations shape, got "
                f"{self.next_observations.shape} and {self.observations.shape}"
            )
        _check_shape("actions", self.actions, n, 1)
        _check_shape("rewards", self.rewards, n, 1)
        _check_shape("dones", self.dones, n, 1)
        _check_shape("timeouts", self.timeouts, n, 1)
        if self.validation_summary()["has_nan"]:
            raise ValueError("dataset contains NaN values")


def _as_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _load_optional_bool(
    data: Any, names: tuple[str, ...], length: int
) -> np.ndarray:
    for name in names:
        if name in data.files:
            return _as_2d(np.asarray(data[name], dtype=bool))
    return np.zeros((length, 1), dtype=bool)


def _check_shape(name: str, array: np.ndarray, rows: int, cols: int) -> None:
    if array.shape != (rows, cols):
        raise ValueError(
            f"{name} must have shape [{rows}, {cols}], got {array.shape}"
        )
