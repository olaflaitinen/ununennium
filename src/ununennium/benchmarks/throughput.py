"""Throughput benchmarking functions."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from ununennium.benchmarks.profiler import Profiler, MemoryProfiler


def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    n_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cuda",
    mixed_precision: bool = True,
) -> Dict[str, float]:
    """Benchmark model inference throughput.

    Args:
        model: Model to benchmark.
        input_shape: Input tensor shape (B, C, H, W).
        n_iterations: Number of benchmark iterations.
        warmup_iterations: Number of warmup iterations.
        device: Device to run on.
        mixed_precision: Whether to use AMP.

    Returns:
        Dictionary with throughput metrics.
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(warmup_iterations):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    profiler = Profiler()
    memory = MemoryProfiler()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with profiler.section("inference"):
        for _ in range(n_iterations):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=mixed_precision):
                    _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()
        memory.snapshot("post_benchmark")

    total_time_ms = sum(profiler.timings["inference"])
    batch_size = input_shape[0]

    return {
        "iterations": n_iterations,
        "batch_size": batch_size,
        "total_time_ms": total_time_ms,
        "avg_latency_ms": total_time_ms / n_iterations,
        "samples_per_second": (n_iterations * batch_size) / (total_time_ms / 1000),
        "p50_latency_ms": float(np.percentile(profiler.timings["inference"], 50)),
        "p95_latency_ms": float(np.percentile(profiler.timings["inference"], 95)),
        "p99_latency_ms": float(np.percentile(profiler.timings["inference"], 99)),
        "gpu_memory_peak_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device == "cuda"
            else 0
        ),
    }


def benchmark_training(
    model: nn.Module,
    loss_fn: nn.Module,
    input_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
    n_iterations: int = 50,
    warmup_iterations: int = 5,
    device: str = "cuda",
    mixed_precision: bool = True,
) -> Dict[str, float]:
    """Benchmark training iteration throughput.

    Args:
        model: Model to benchmark.
        loss_fn: Loss function.
        input_shape: Input tensor shape.
        target_shape: Target tensor shape.
        n_iterations: Number of benchmark iterations.
        warmup_iterations: Number of warmup iterations.
        device: Device to run on.
        mixed_precision: Whether to use AMP.

    Returns:
        Dictionary with throughput metrics.
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

    dummy_input = torch.randn(*input_shape, device=device)
    dummy_target = torch.randint(0, 10, target_shape, device=device)

    # Warmup
    for _ in range(warmup_iterations):
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            output = model(dummy_input)
            loss = loss_fn(output, dummy_target)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    profiler = Profiler()

    with profiler.section("training"):
        for _ in range(n_iterations):
            with profiler.section("forward"):
                with torch.cuda.amp.autocast(enabled=mixed_precision):
                    output = model(dummy_input)
                    loss = loss_fn(output, dummy_target)

            with profiler.section("backward"):
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            with profiler.section("optimizer"):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()

    total_time_ms = sum(profiler.timings["training"])
    batch_size = input_shape[0]

    return {
        "iterations": n_iterations,
        "batch_size": batch_size,
        "total_time_ms": total_time_ms,
        "avg_iteration_ms": total_time_ms / n_iterations,
        "samples_per_second": (n_iterations * batch_size) / (total_time_ms / 1000),
        "forward_ms": float(np.mean(profiler.timings.get("training/forward", [0]))),
        "backward_ms": float(np.mean(profiler.timings.get("training/backward", [0]))),
        "optimizer_ms": float(np.mean(profiler.timings.get("training/optimizer", [0]))),
        "gpu_memory_peak_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device == "cuda"
            else 0
        ),
    }
