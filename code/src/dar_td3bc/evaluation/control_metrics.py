from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SettlingEvent:
    switch_index: int
    previous_target: float
    target: float
    settled_index: int | None
    settling_time: float | None
    max_overshoot: float
    censored: bool


def rmse(y: Iterable[float], r: Iterable[float]) -> float:
    y_array, r_array = _paired_arrays(y, r)
    return float(np.sqrt(np.mean((y_array - r_array) ** 2)))


def mae(y: Iterable[float], r: Iterable[float]) -> float:
    y_array, r_array = _paired_arrays(y, r)
    return float(np.mean(np.abs(y_array - r_array)))


def iae(y: Iterable[float], r: Iterable[float], dt: float = 1.0) -> float:
    y_array, r_array = _paired_arrays(y, r)
    return float(np.sum(np.abs(y_array - r_array)) * dt)


def action_total_variation(action: Iterable[float]) -> float:
    action_array = np.asarray(action, dtype=float).reshape(-1)
    if action_array.size < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(action_array))))


def action_energy(action: Iterable[float]) -> float:
    action_array = np.asarray(action, dtype=float)
    return float(np.sum(action_array**2))


def target_switch_settling_events(
    flow: Iterable[float],
    target: Iterable[float],
    *,
    hold_steps: int = 10,
    dt: float = 1.0,
) -> list[SettlingEvent]:
    flow_array, target_array = _paired_arrays(flow, target)
    if hold_steps <= 0:
        raise ValueError("hold_steps must be positive")
    if flow_array.size < 2:
        return []

    switch_indices = np.flatnonzero(target_array[1:] != target_array[:-1]) + 1
    events: list[SettlingEvent] = []

    for pos, switch_index in enumerate(switch_indices):
        next_switch = (
            int(switch_indices[pos + 1])
            if pos + 1 < len(switch_indices)
            else flow_array.size
        )
        previous_target = float(target_array[switch_index - 1])
        new_target = float(target_array[switch_index])
        segment_flow = flow_array[switch_index:next_switch]
        segment_target = target_array[switch_index:next_switch]

        band = max(0.02 * abs(new_target), 2.0)
        errors = np.abs(segment_flow - segment_target)
        settled_index: int | None = None
        last_start = errors.size - hold_steps
        if last_start >= 0:
            for offset in range(last_start + 1):
                if np.all(errors[offset : offset + hold_steps] <= band):
                    settled_index = int(switch_index + offset)
                    break

        direction = np.sign(new_target - previous_target)
        if direction == 0.0:
            max_overshoot = 0.0
        else:
            overshoot = direction * (segment_flow - new_target)
            max_overshoot = float(max(0.0, np.max(overshoot)))

        events.append(
            SettlingEvent(
                switch_index=int(switch_index),
                previous_target=previous_target,
                target=new_target,
                settled_index=settled_index,
                settling_time=(
                    None
                    if settled_index is None
                    else float((settled_index - switch_index) * dt)
                ),
                max_overshoot=max_overshoot,
                censored=settled_index is None,
            )
        )

    return events


def _paired_arrays(
    y: Iterable[float], r: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    y_array = np.asarray(y, dtype=float).reshape(-1)
    r_array = np.asarray(r, dtype=float).reshape(-1)
    if y_array.shape != r_array.shape:
        raise ValueError(
            f"Expected arrays with matching shape, got {y_array.shape} and "
            f"{r_array.shape}"
        )
    if y_array.size == 0:
        raise ValueError("metric inputs must not be empty")
    return y_array, r_array
