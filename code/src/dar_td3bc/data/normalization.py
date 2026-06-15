from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def transform_numpy(self, value: np.ndarray) -> np.ndarray:
        return (np.asarray(value, dtype=float) - self.mean) / self.std

    def inverse_numpy(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=float) * self.std + self.mean

    def transform_torch(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.std, device=value.device, dtype=value.dtype)
        return (value - mean) / std


def fit_normalization(values: np.ndarray, eps: float = 1e-6) -> NormalizationStats:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0 or array.shape[0] == 0:
        raise ValueError("values must contain at least one row")
    mean = array.mean(axis=0)
    std = np.maximum(array.std(axis=0), eps)
    return NormalizationStats(mean=mean, std=std)
