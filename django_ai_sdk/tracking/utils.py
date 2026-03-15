import time
from collections.abc import Callable
from functools import wraps
from typing import Any


class OptimisticTracker:
    _instance = None

    DEFAULTS = {
        "llm": 100,
        "image": 50,
        "embedding": 1000,
    }

    def __new__(cls) -> "OptimisticTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._counts = {k: 0 for k in cls.DEFAULTS}
            cls._instance._window_start = time.time()
        return cls._instance

    def track(self, category: str) -> None:
        self._reset_if_needed()
        self._counts[category] = self._counts.get(category, 0) + 1

    def _reset_if_needed(self) -> None:
        now = time.time()
        if now - self._window_start >= 60:
            self._counts = {k: 0 for k in self.DEFAULTS}
            self._window_start = now

    def get_state(self) -> dict[str, Any]:
        self._reset_if_needed()
        return {
            "window_start": self._window_start,
            "counts": dict(self._counts),
            "limits": dict(self.DEFAULTS),
        }


tracker = OptimisticTracker()


def track_llm[F: Callable[..., Any]](func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tracker.track("llm")
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def track_image[F: Callable[..., Any]](func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tracker.track("image")
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def track_embedding[F: Callable[..., Any]](func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tracker.track("embedding")
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
