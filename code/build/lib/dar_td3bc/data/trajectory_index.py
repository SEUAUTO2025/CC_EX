from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FutureTargets:
    values: np.ndarray
    mask: np.ndarray
    horizons: tuple[int, ...]


class TrajectoryIndex:
    """Episode-aware index over transition-style offline data."""

    def __init__(
        self,
        terminals: Iterable[bool],
        timeouts: Iterable[bool] | None = None,
        horizon: int | None = None,
    ) -> None:
        terminals_array = np.asarray(terminals, dtype=bool).reshape(-1)
        if timeouts is None:
            timeouts_array = np.zeros_like(terminals_array, dtype=bool)
        else:
            timeouts_array = np.asarray(timeouts, dtype=bool).reshape(-1)
            if timeouts_array.shape != terminals_array.shape:
                raise ValueError(
                    "timeouts must have the same length as terminals, got "
                    f"{timeouts_array.shape[0]} and {terminals_array.shape[0]}"
                )
        if terminals_array.size == 0:
            raise ValueError("terminals must contain at least one transition")

        self.terminals = terminals_array
        self.timeouts = timeouts_array
        self.ends = np.logical_or(terminals_array, timeouts_array)
        if horizon is not None and horizon <= 0:
            raise ValueError("horizon must be positive when provided")
        self.horizon = horizon

        if horizon is not None:
            forced_ends = np.zeros_like(self.ends)
            forced_ends[horizon - 1 :: horizon] = True
            self.ends = np.logical_or(self.ends, forced_ends)

        if not self.ends[-1]:
            self.ends = self.ends.copy()
            self.ends[-1] = True

        self.episode_end_index = np.empty_like(self.ends, dtype=np.int64)
        start = 0
        for end in np.flatnonzero(self.ends):
            self.episode_end_index[start : end + 1] = end
            start = end + 1

    def future_flow_targets(
        self, flow: Iterable[float], horizons: Iterable[int]
    ) -> FutureTargets:
        flow_array = np.asarray(flow, dtype=float).reshape(-1)
        if flow_array.shape[0] != self.ends.shape[0]:
            raise ValueError(
                "flow must have the same length as terminals, got "
                f"{flow_array.shape[0]} and {self.ends.shape[0]}"
            )

        horizon_tuple = tuple(int(h) for h in horizons)
        if not horizon_tuple:
            raise ValueError("horizons must not be empty")
        if any(h <= 0 for h in horizon_tuple):
            raise ValueError(f"horizons must be positive, got {horizon_tuple}")

        values = np.zeros((flow_array.shape[0], len(horizon_tuple)), dtype=float)
        mask = np.zeros_like(values)
        indices = np.arange(flow_array.shape[0])

        for col, horizon in enumerate(horizon_tuple):
            future_indices = indices + horizon
            valid = future_indices <= self.episode_end_index
            values[valid, col] = flow_array[future_indices[valid]]
            mask[valid, col] = 1.0

        return FutureTargets(values=values, mask=mask, horizons=horizon_tuple)
