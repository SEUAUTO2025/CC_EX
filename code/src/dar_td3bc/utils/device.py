from __future__ import annotations

from typing import Any

import torch


def resolve_device(config: dict[str, Any] | None = None) -> torch.device:
    config = config or {}
    requested = str(config.get("device", "cpu")).lower()
    if requested in {"auto", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device
