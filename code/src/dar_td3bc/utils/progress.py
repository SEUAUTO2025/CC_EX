from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def progress_range(
    steps: int,
    *,
    desc: str,
    enabled: bool = True,
) -> Iterator[int]:
    return progress_iter(range(1, steps + 1), desc=desc, total=steps, enabled=enabled)


def progress_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    enabled: bool = True,
) -> Iterator[T]:
    if not enabled:
        yield from iterable
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from iterable
        return
    yield from tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def progress_enabled(config: dict | None = None) -> bool:
    if config is None:
        return True
    return bool(config.get("progress", True))
