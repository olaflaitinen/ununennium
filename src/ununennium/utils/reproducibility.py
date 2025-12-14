"""Reproducibility utilities for deterministic training."""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, enables PyTorch deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True)


def get_seed_state() -> dict[str, Any]:
    """Capture current random state for later restoration.

    Returns:
        Dictionary with state for each RNG.
    """
    state: dict[str, Any] = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_seed_state(state: dict[str, Any]) -> None:
    """Restore random state from captured state.

    Args:
        state: Dictionary from get_seed_state().
    """
    random.setstate(state["random"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]
