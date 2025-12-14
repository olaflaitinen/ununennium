"""Profiling utilities for benchmarking."""

from __future__ import annotations

import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch


class Profiler:
    """Hierarchical profiler for timing code sections."""

    def __init__(self):
        self.timings: dict[str, list[float]] = defaultdict(list)
        self._stack: list[str] = []

    @contextmanager
    def section(self, name: str):
        """Time a code section.

        Args:
            name: Section name.

        Yields:
            Context manager.
        """
        full_name = "/".join([*self._stack, name])
        self._stack.append(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self._stack.pop()
            self.timings[full_name].append(elapsed)

    def report(self) -> dict[str, dict[str, float]]:
        """Generate timing report.

        Returns:
            Dictionary with statistics per section.
        """
        return {
            name: {
                "mean_ms": float(np.mean(times)),
                "std_ms": float(np.std(times)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
                "count": len(times),
            }
            for name, times in self.timings.items()
        }

    def reset(self) -> None:
        """Clear all timings."""
        self.timings.clear()


class MemoryProfiler:
    """Track memory usage during operations."""

    def __init__(self, track_cuda: bool = True):
        self.track_cuda = track_cuda
        self.snapshots: list[dict[str, Any]] = []

    def snapshot(self, label: str) -> dict[str, Any]:
        """Take memory snapshot.

        Args:
            label: Snapshot label.

        Returns:
            Memory stats dictionary.
        """

        result: dict[str, Any] = {"label": label, "timestamp": time.time()}

        # CPU memory
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            result["cpu_current_mb"] = current / 1024**2
            result["cpu_peak_mb"] = peak / 1024**2

        # GPU memory
        if self.track_cuda and torch.cuda.is_available():
            result["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            result["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            result["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2

        self.snapshots.append(result)
        return result

    @contextmanager
    def track(self, label: str):
        """Context manager for memory tracking.

        Args:
            label: Section label.
        """

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        tracemalloc.start()
        self.snapshot(f"{label}_start")

        try:
            yield
        finally:
            self.snapshot(f"{label}_end")
            tracemalloc.stop()

    @property
    def peak_memory(self) -> float | None:
        """Get peak memory usage in MB."""
        if not self.snapshots:
            return None
        # Find max peak from snapshots
        peaks = [s.get("cpu_peak_mb", 0) for s in self.snapshots]
        cuda_peaks = [s.get("cuda_max_allocated_mb", 0) for s in self.snapshots]
        return max(peaks + cuda_peaks)

    def __enter__(self):
        # Start tracking manually

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tracemalloc.start()
        self.snapshot("start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.snapshot("end")
        tracemalloc.stop()
