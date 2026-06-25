from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class ObservationNoiseWrapper:
    """Add Gaussian observation noise during evaluation without changing rewards."""

    def __init__(self, env: Any, std: float, *, seed_offset: int = 0) -> None:
        if std < 0.0:
            raise ValueError("std must be non-negative")
        self.env = env
        self.std = float(std)
        self.seed_offset = int(seed_offset)
        self._rng = np.random.default_rng(seed_offset)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed) + self.seed_offset)
        result = _reset(self.env, seed=seed, options=options)
        if isinstance(result, tuple):
            obs, info = result
            return self._noise(obs), info
        return self._noise(result)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return self._noise(obs), reward, terminated, truncated, info
        obs, reward, done, info = result
        return self._noise(obs), reward, done, info

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _noise(self, observation: Any) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if self.std == 0.0:
            return array.copy()
        return (array + self._rng.normal(0.0, self.std, size=array.shape)).astype(
            np.float32
        )


class ObservationDelayWrapper:
    """Delay policy observations while stepping the original environment."""

    def __init__(self, env: Any, delay_steps: int) -> None:
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        self.env = env
        self.delay_steps = int(delay_steps)
        self._buffer: deque[np.ndarray] = deque(maxlen=max(self.delay_steps + 1, 1))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        result = _reset(self.env, seed=seed, options=options)
        if isinstance(result, tuple):
            obs, info = result
            return self._reset_buffer(obs), info
        return self._reset_buffer(result)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return self._delay(obs), reward, terminated, truncated, info
        obs, reward, done, info = result
        return self._delay(obs), reward, done, info

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _reset_buffer(self, observation: Any) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        self._buffer.clear()
        for _ in range(self.delay_steps + 1):
            self._buffer.append(obs.copy())
        return self._buffer[0].copy()

    def _delay(self, observation: Any) -> np.ndarray:
        self._buffer.append(np.asarray(observation, dtype=np.float32).copy())
        return self._buffer[0].copy()


def apply_robustness_wrappers(
    env: Any,
    *,
    observation_noise_std: float = 0.0,
    observation_delay_steps: int = 0,
) -> Any:
    wrapped = env
    if observation_delay_steps > 0:
        wrapped = ObservationDelayWrapper(wrapped, observation_delay_steps)
    if observation_noise_std > 0.0:
        wrapped = ObservationNoiseWrapper(wrapped, observation_noise_std)
    return wrapped


def _reset(env: Any, *, seed: int | None, options: dict | None):
    try:
        return env.reset(seed=seed, options=options)
    except TypeError:
        if options is not None:
            return env.reset(seed=seed)
        try:
            return env.reset(seed=seed)
        except TypeError:
            return env.reset()
