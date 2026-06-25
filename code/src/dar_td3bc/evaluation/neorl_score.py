from __future__ import annotations

from typing import Any

import numpy as np


def normalized_score(env: Any, raw_return: float) -> float:
    """Compute the environment-defined NeoRL-style normalized score."""
    scorer = _find_scorer(env)
    score = scorer(float(raw_return))
    value = float(np.asarray(score, dtype=np.float64).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError(f"normalized score must be finite, got {value}")
    return value


def _find_scorer(env: Any):
    for candidate in (env, getattr(env, "unwrapped", None)):
        if candidate is not None and hasattr(candidate, "get_normalized_score"):
            return candidate.get_normalized_score
    raise AttributeError("environment does not expose get_normalized_score")
